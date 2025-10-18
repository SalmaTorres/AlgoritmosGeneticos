import cv2
import numpy as np
import random
import mediapipe as mp

# ---------------------------
# Configuración MediaPipe
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

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

def draw_landmarks_points(image, landmarks):
    img_copy = image.copy()
    for (x, y) in landmarks:
        cv2.circle(img_copy, (x, y), 1, (0, 255, 0), -1)
    return img_copy

def warp_image(src, src_points, dst_points, size):
    src_points = np.float32(src_points)
    dst_points = np.float32(dst_points)
    matrix, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
    warped = cv2.warpAffine(src, matrix, (size[1], size[0]), flags=cv2.INTER_LINEAR)
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
        return img

    modified_landmarks = landmarks.copy()

    # --- Ojos ---
    eyes_idx = [33, 133, 362, 263]
    center_eyes = np.mean(landmarks[eyes_idx], axis=0)
    for i in eyes_idx:
        direction = landmarks[i] - center_eyes
        modified_landmarks[i] = landmarks[i] + direction * (eyes_scale - 1)

    # --- Boca ---
    mouth_idx = list(range(61, 81)) + list(range(291, 309))
    center_mouth = np.mean(landmarks[mouth_idx], axis=0)
    for i in mouth_idx:
        direction = landmarks[i] - center_mouth
        modified_landmarks[i] = landmarks[i] + direction * (mouth_scale - 1)

    # --- Mentón y cabeza ---
    chin_idx = list(range(152, 178))
    center_chin = np.mean(landmarks[chin_idx], axis=0)
    for i in chin_idx:
        direction = landmarks[i] - center_chin
        modified_landmarks[i] = landmarks[i] + direction * (chin_scale - 1)

    # Cabeza general (aproximada)
    head_idx = [10, 338, 297, 332]
    center_head = np.mean(landmarks[head_idx], axis=0)
    for i in head_idx:
        direction = landmarks[i] - center_head
        modified_landmarks[i] = landmarks[i] + direction * (head_scale - 1)

    # --- Warping geométrico ---
    warped = warp_image(img, landmarks, modified_landmarks, img.shape)

    # --- Saturación para efecto “toon” ---
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    cartoon = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # --- Filtro estilo caricatura (bordes) ---
    gray = cv2.cvtColor(cartoon, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(cartoon, 9, 250, 250)
    final_img = cv2.bitwise_and(color, color, mask=edges)

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
    exaggeration_eyes = dist_eyes_mod / dist_eyes_orig

    dist_mouth_orig = np.linalg.norm(landmarks_orig[61] - landmarks_orig[291])
    dist_mouth_mod = np.linalg.norm(landmarks_mod[61] - landmarks_mod[291])
    exaggeration_mouth = dist_mouth_mod / dist_mouth_orig

    E = np.clip((exaggeration_eyes + exaggeration_mouth)/2, 0, 2)

    # Distorsión global
    D = np.mean(np.linalg.norm(landmarks_orig - landmarks_mod, axis=1))

    alpha, beta = 0.7, 0.3
    fitness = alpha * E - beta * D
    return fitness

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
            new_indiv[i] = np.clip(new_indiv[i] + perturb, low, high)
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
            fitness = compute_fitness_caricature(landmarks_orig, landmarks_mod)
            fitnesses.append(fitness)
            modified_images.append(img_mod)
            landmarks_list.append(landmarks_mod)

        max_idx = np.argmax(fitnesses)
        if fitnesses[max_idx] > best_fitness:
            best_fitness = fitnesses[max_idx]
            best_individual = population[max_idx]
            best_image = modified_images[max_idx]

        # Nueva generación
        new_population = []
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
    image_orig = cv2.imread("maluma.jpg")  # Cambia la ruta
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
