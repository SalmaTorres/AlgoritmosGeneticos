import cv2
import numpy as np
import random
import mediapipe as mp

# ---------------------------
# Configuración MediaPipe
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh
# >>> CAMBIO: landmarks más estables con refine_landmarks y leves umbrales de confianza
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,              # antes: False por defecto
    min_detection_confidence=0.6,       # extra
    min_tracking_confidence=0.6         # extra
)

def get_landmarks(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = []
    for lm in results.multi_face_landmarks[0].landmark:
        x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
        landmarks.append([x, y])
    return np.array(landmarks, np.int32)

# >>> CAMBIO: utilidad pequeña para normalizar por escala facial
def interpupillary(lm):
    # puntos laterales de los ojos: 33 y 263 (FaceMesh 468)
    return float(np.linalg.norm(lm[33] - lm[263])) + 1e-6

def draw_landmarks_points(image, landmarks):
    img_copy = image.copy()
    for (x, y) in landmarks:
        cv2.circle(img_copy, (x, y), 1, (0, 255, 0), -1)
    return img_copy

def warp_image(src, src_points, dst_points, size):
    src_points = np.float32(src_points)
    dst_points = np.float32(dst_points)
    matrix, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
    # >>> CAMBIO: evitar bordes negros al deformar
    warped = cv2.warpAffine(
        src, matrix, (size[1], size[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )
    return warped

# ---------------------------
# Funciones de efectos (caricatura)
# ---------------------------
def apply_effects_caricature(image, params):
    """Aplica deformaciones tipo caricatura según los parámetros"""
    eyes_scale, mouth_scale, chin_scale, head_scale, sat_scale = params
    img = image.copy()
    landmarks = get_landmarks(img)
    if landmarks is None:
        return img, None   # >>> CAMBIO: devolver también None en landmarks para coherencia

    modified_landmarks = landmarks.copy()

    # >>> CAMBIO: limitar desplazamientos por-punto y suavizar radialmente (evita "derretir")
    ipd = interpupillary(landmarks)
    sigma = 0.9 * ipd          # radio de atenuación en cada región
    max_px = 0.15 * ipd        # tope de desplazamiento por punto

    def _apply_region(idxs, center, scale):
        for i in idxs:
            direction = landmarks[i] - center
            # peso gaussiano: menos efecto lejos del centro de la región
            w = np.exp(-(np.linalg.norm(direction) ** 2) / (2 * (sigma ** 2)))
            disp = direction * (scale - 1.0) * w
            n = np.linalg.norm(disp)
            if n > max_px:
                disp = disp * (max_px / n)
            modified_landmarks[i] = (landmarks[i] + disp).astype(np.int32)

    # --- Ojos ---
    eyes_idx = [33, 133, 362, 263]
    center_eyes = np.mean(landmarks[eyes_idx], axis=0)
    _apply_region(eyes_idx, center_eyes, eyes_scale)  # >>> CAMBIO: usa helper con suavizado y tope

    # --- Boca ---
    mouth_idx = list(range(61, 81)) + list(range(291, 309))
    center_mouth = np.mean(landmarks[mouth_idx], axis=0)
    _apply_region(mouth_idx, center_mouth, mouth_scale)  # >>> CAMBIO

    # --- Mentón ---
    chin_idx = list(range(152, 178))
    center_chin = np.mean(landmarks[chin_idx], axis=0)
    _apply_region(chin_idx, center_chin, chin_scale)  # >>> CAMBIO

    # --- Cabeza (aproximada) ---
    head_idx = [10, 338, 297, 332]
    center_head = np.mean(landmarks[head_idx], axis=0)
    _apply_region(head_idx, center_head, head_scale)  # >>> CAMBIO

    # --- Warping geométrico ---
    warped = warp_image(img, landmarks, modified_landmarks, img.shape)

    # --- Saturación para efecto “toon” ---
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * sat_scale, 0, 255).astype(np.uint8)
    cartoon = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # --- Filtro estilo caricatura (bordes) ---
    # >>> CAMBIO: toon más nítido y limpio (mean-shift + Canny invertido)
    smooth = cv2.bilateralFilter(cartoon, 9, 120, 120)
    quant = cv2.pyrMeanShiftFiltering(smooth, sp=8, sr=20)
    gray = cv2.cvtColor(quant, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.bitwise_not(edges)
    edges3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    final_img = cv2.bitwise_and(quant, edges3)

    return final_img, modified_landmarks

# ---------------------------
# Fitness caricatura
# ---------------------------
def compute_fitness_caricature(landmarks_orig, landmarks_mod):
    # Exageración ojos y boca
    eyes_idx = [33, 133, 362, 263]
    mouth_idx = [61, 291]

    dist_eyes_orig = np.linalg.norm(landmarks_orig[33] - landmarks_orig[263])
    dist_eyes_mod = np.linalg.norm(landmarks_mod[33] - landmarks_mod[263])
    exaggeration_eyes = dist_eyes_mod / (dist_eyes_orig + 1e-6)

    dist_mouth_orig = np.linalg.norm(landmarks_orig[61] - landmarks_orig[291])
    dist_mouth_mod = np.linalg.norm(landmarks_mod[61] - landmarks_mod[291])
    exaggeration_mouth = dist_mouth_mod / (dist_mouth_orig + 1e-6)

    E = np.clip((exaggeration_eyes + exaggeration_mouth) / 2.0, 0.0, 2.5)

    # Distorsión global (normalizada por escala facial)
    # >>> CAMBIO: normalizar por distancia interpupilar para que no dependa del tamaño en pixeles
    ipd = interpupillary(landmarks_orig)
    D = np.mean(np.linalg.norm(landmarks_orig - landmarks_mod, axis=1)) / ipd

    alpha, beta = 0.7, 0.3
    fitness = alpha * E - beta * D
    return float(fitness)

# >>> CAMBIO: componente de nitidez (Tenengrad/Sobel) para premiar bordes limpios
def edge_nitidez_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag2 = gx * gx + gy * gy
    s = float(np.mean(mag2)) / (255.0 * 255.0)
    # compresión suave a [0,1]
    return float(np.tanh(3.0 * s))

# ---------------------------
# Algoritmo Genético
# ---------------------------
def initialize_population(pop_size, param_bounds):
    population = []
    for _ in range(pop_size):
        indiv = [random.uniform(low, high) for (low, high) in param_bounds]
        population.append(indiv)
    return population

def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]

def arithmetic_crossover(parent1, parent2):
    lambda_ = random.random()
    child = [lambda_ * p1 + (1 - lambda_) * p2 for p1, p2 in zip(parent1, parent2)]
    return child

def mutate(individual, param_bounds, mutation_rate=0.1):
    new_indiv = individual.copy()
    for i, (low, high) in enumerate(param_bounds):
        if random.random() < mutation_rate:
            perturb = (high - low) * random.uniform(-0.1, 0.1)
            new_indiv[i] = float(np.clip(new_indiv[i] + perturb, low, high))
    return new_indiv

def genetic_algorithm_caricature(image_orig, param_bounds, pop_size=15, generations=50):
    landmarks_orig = get_landmarks(image_orig)
    if landmarks_orig is None:
        raise ValueError("No se detectaron landmarks en la imagen original")

    population = initialize_population(pop_size, param_bounds)
    best_individual, best_fitness, best_image = None, -np.inf, None

    for gen in range(generations):
        fitnesses, modified_images, landmarks_list = [], [], []

        for indiv in population:
            img_mod, landmarks_mod = apply_effects_caricature(image_orig, indiv)
            # >>> CAMBIO: fitness compuesta = landmarks + nitidez (ponderado suave)
            f_lm = compute_fitness_caricature(landmarks_orig, landmarks_mod)
            f_edge = edge_nitidez_score(img_mod)
            fitness = 0.85 * f_lm + 0.15 * f_edge
            fitnesses.append(float(fitness))
            modified_images.append(img_mod)
            landmarks_list.append(landmarks_mod)

        max_idx = int(np.argmax(fitnesses))
        if fitnesses[max_idx] > best_fitness:
            best_fitness = fitnesses[max_idx]
            best_individual = population[max_idx]
            best_image = modified_images[max_idx]

        # >>> CAMBIO: elitismo 1 — preserva el mejor de la generación
        elite = population[max_idx]

        # Nueva generación
        new_population = [elite]  # arranca con el mejor
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = arithmetic_crossover(parent1, parent2)
            child = mutate(child, param_bounds)
            new_population.append(child)
        population = new_population

        print(f"Generación {gen+1}: Mejor fitness = {best_fitness:.4f}")

    return best_individual, best_fitness, best_image


def resize_image_keep_ratio(image, max_height=400, max_width=400):
    """
    Redimensiona la imagen manteniendo la proporción.
    max_height, max_width: límites máximos de alto y ancho
    """
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


# ---------------------------
# Ejecutar
# ---------------------------
if __name__ == "__main__":
    image_orig = cv2.imread("MJ.jpg")  # Cambia la ruta
    image_orig = resize_image_keep_ratio(image_orig, max_height=400, max_width=400)
    param_bounds = [
        (1.0, 1.5),  # ojos
        (0.8, 1.3),  # boca
        (0.8, 1.2),  # mentón
        (0.9, 1.2),  # cabeza
        (0.8, 1.5)   # saturación
    ]

    best_params, best_fit, best_img = genetic_algorithm_caricature(
        image_orig, param_bounds, pop_size=15, generations=50
    )

    landmarks_orig = get_landmarks(image_orig)
    image_with_landmarks = draw_landmarks_points(image_orig, landmarks_orig)

    h = image_orig.shape[0]
    imgs_to_show = [image_orig, image_with_landmarks, best_img]
    resized_imgs = [cv2.resize(im, (int(im.shape[1]*h/im.shape[0]), h)) for im in imgs_to_show]
    combined = np.hstack(resized_imgs)

    cv2.imshow("Original | Landmarks | Caricatura", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
