import cv2
import mediapipe as mp
import numpy as np

# --- CONFIGURACIÓN DE PUNTOS PARA CARICATURA ---
# Índices de MediaPipe para los rasgos que vamos a exagerar.
# Puedes encontrar estos índices en la documentación de MediaPipe Face Mesh.
CARTOON_CONFIG = {
    'left_eye': [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173
    ],
    'right_eye': [
        263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398
    ],
    'mouth_outer': [61, 291, 0, 17, 405, 181, 91, 146, 321, 375],
    'nose_tip': [1]  # Usado como punto de referencia central
}

# --- INICIALIZACIÓN DE MEDIAPIPE ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
)


# --- FUNCIONES DE DEFORMACIÓN DE MALLA (WARPING) ---
# (Se mantienen tus funciones originales, son perfectas para esto)
def warp_triangle(src_img, dest_img, tri_src, tri_dest):
    rect_src = cv2.boundingRect(np.float32(tri_src))
    rect_dest = cv2.boundingRect(np.float32(tri_dest))

    tri_src_cropped = [(tri_src[i][0] - rect_src[0], tri_src[i][1] - rect_src[1]) for i in range(3)]
    tri_dest_cropped = [(tri_dest[i][0] - rect_dest[0], tri_dest[i][1] - rect_dest[1]) for i in range(3)]

    src_cropped = src_img[rect_src[1]:rect_src[1] + rect_src[3], rect_src[0]:rect_src[0] + rect_src[2]]
    warp_mat = cv2.getAffineTransform(np.float32(tri_src_cropped), np.float32(tri_dest_cropped))
    dest_cropped = cv2.warpAffine(src_cropped, warp_mat, (rect_dest[2], rect_dest[3]), None,
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    mask = np.zeros((rect_dest[3], rect_dest[2]), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri_dest_cropped), 1.0, 16, 0)

    if rect_dest[0] >= 0 and rect_dest[1] >= 0 and \
            rect_dest[0] + rect_dest[2] <= dest_img.shape[1] and \
            rect_dest[1] + rect_dest[3] <= dest_img.shape[0]:

        dest_area = dest_img[rect_dest[1]:rect_dest[1] + rect_dest[3], rect_dest[0]:rect_dest[0] + rect_dest[2]]

        mask_3ch = mask[:, :, np.newaxis]
        if dest_area.shape == dest_cropped.shape and dest_area.shape[:2] == mask.shape:
            dest_area *= (1.0 - mask_3ch)
            dest_area += dest_cropped * mask_3ch


def apply_mesh_deformation(image, original_landmarks, new_landmarks):
    dest_img = image.copy().astype(np.float32)
    points_orig = np.array(original_landmarks, dtype=np.float32)
    points_new = np.array(new_landmarks, dtype=np.float32)
    rect = (0, 0, image.shape[1], image.shape[0])
    subdiv = cv2.Subdiv2D(rect)

    valid_points = []
    for p in points_orig.tolist():
        p_int = (int(p[0]), int(p[1]))
        if 0 <= p_int[0] < rect[2] and 0 <= p_int[1] < rect[3] and p_int not in valid_points:
            valid_points.append(p_int)

    if len(valid_points) < 3: return image

    subdiv.insert(valid_points)
    triangle_list = subdiv.getTriangleList()

    for t in triangle_list:
        pts = np.array([(t[0], t[1]), (t[2], t[3]), (t[4], t[5])], dtype=np.float32)
        indices = []
        for i in range(3):
            dist = np.linalg.norm(points_orig - pts[i], axis=1)
            min_dist_idx = np.argmin(dist)
            if dist[min_dist_idx] < 1.0:
                indices.append(min_dist_idx)
            else:
                indices = []
                break
        if len(set(indices)) == 3:
            tri_src = points_orig[indices]
            tri_dest = points_new[indices]
            warp_triangle(image.astype(np.float32), dest_img, tri_src, tri_dest)

    return dest_img.astype(np.uint8)


# --- NUEVAS FUNCIONES PARA CARICATURIZACIÓN ---

def exaggerate_landmarks(landmarks, eye_scale=1.3, mouth_scale=1.4):
    """
    Modifica los landmarks para exagerar los rasgos.
    - eye_scale: Factor para agrandar los ojos (1.0 = sin cambio).
    - mouth_scale: Factor para agrandar la boca.
    """
    new_landmarks = list(landmarks)

    # Exagerar ojos
    for eye_indices in [CARTOON_CONFIG['left_eye'], CARTOON_CONFIG['right_eye']]:
        eye_points = np.array([landmarks[i] for i in eye_indices])
        center = np.mean(eye_points, axis=0)

        for i in eye_indices:
            vec = np.array(landmarks[i]) - center
            new_landmarks[i] = tuple((center + vec * eye_scale).astype(int))

    # Exagerar boca (hacerla más ancha)
    mouth_points = np.array([landmarks[i] for i in CARTOON_CONFIG['mouth_outer']])
    center = np.mean(mouth_points, axis=0)

    for i in CARTOON_CONFIG['mouth_outer']:
        vec = np.array(landmarks[i]) - center
        # Exagerar más en horizontal que en vertical para una sonrisa ancha
        scaled_vec = np.array([vec[0] * mouth_scale, vec[1] * (1 + (mouth_scale - 1) * 0.5)])
        new_landmarks[i] = tuple((center + scaled_vec).astype(int))

    return new_landmarks


def add_cartoon_style(image):
    """Aplica el estilo visual de caricatura: bordes y colores simplificados."""
    # 1. Suavizado fuerte para eliminar texturas
    # El filtro bilateral preserva los bordes mientras suaviza el resto.
    smoothed = cv2.bilateralFilter(image, d=15, sigmaColor=80, sigmaSpace=80)

    # 2. Detección de bordes para el contorno negro
    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=7
    )
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 3. Combinar la imagen suavizada con los bordes
    cartoon_image = cv2.bitwise_and(smoothed, edges)

    return cartoon_image


# --- FUNCIÓN PRINCIPAL DE CARICATURIZACIÓN ---

def cartoonify_face(image_path, eye_scale=1.25, mouth_scale=1.35):
    """
    Función principal que carga una imagen, detecta el rostro,
    lo deforma y le aplica un estilo de caricatura.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    original_image = image.copy()
    h, w, _ = image.shape

    # 1. Detección de landmarks con MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        raise Exception("No se detectó rostro en la imagen.")

    landmarks_norm = results.multi_face_landmarks[0].landmark
    original_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_norm]

    # Añadir puntos de anclaje en las esquinas para estabilizar la deformación
    anchor_points = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    stable_original_landmarks = original_landmarks + anchor_points

    # 2. Exagerar los rasgos faciales
    exaggerated_landmarks = exaggerate_landmarks(original_landmarks, eye_scale, mouth_scale)
    stable_exaggerated_landmarks = exaggerated_landmarks + anchor_points

    # 3. Aplicar la deformación de malla
    morphed_image = apply_mesh_deformation(image, stable_original_landmarks, stable_exaggerated_landmarks)

    # 4. Aplicar el estilo visual de caricatura (bordes y suavizado)
    cartoon_result = add_cartoon_style(morphed_image)

    return original_image, cartoon_result


# --- EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    try:
        # Reemplaza "ruta/a/tu/imagen.jpg" con la ruta de tu foto.
        # Puedes usar la misma foto que usaste para rejuvenecer.
        image_file = "ProyectoRostro/rostro.jpg"

        # Juega con estos valores para diferentes estilos de caricatura
        EYE_SIZE = 1.3  # 1.0 es normal, 1.5 es muy grande
        MOUTH_SIZE = 1.4  # 1.0 es normal, 1.5 es muy grande

        original, caricatura = cartoonify_face(image_file, eye_scale=EYE_SIZE, mouth_scale=MOUTH_SIZE)

        # Mostrar los resultados
        combined_result = np.hstack([original, caricatura])

        # Redimensionar si es muy grande para la pantalla
        max_width = 1600
        h, w, _ = combined_result.shape
        if w > max_width:
            scale = max_width / w
            new_w, new_h = int(w * scale), int(h * scale)
            combined_result = cv2.resize(combined_result, (new_w, new_h))

        cv2.imshow("Original vs. Caricatura", combined_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Guardar el resultado
        cv2.imwrite("resultado_caricatura.jpg", caricatura)
        print("¡Imagen de caricatura guardada como 'resultado_caricatura.jpg'!")

    except (FileNotFoundError, Exception) as e:
        print(f"Ocurrió un error: {e}")