import cv2
import numpy as np
import random
import mediapipe as mp
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
    """Dibuja los landmarks sobre la imagen."""
    img_copy = image.copy()
    for (x, y) in landmarks:
        cv2.circle(img_copy, (x, y), 1, (0, 255, 0), -1)
    return img_copy


def warp_image(src, src_points, dst_points, size):
    """Warp facial según puntos de control"""
    src_points = np.float32(src_points)
    dst_points = np.float32(dst_points)
    matrix, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
    warped = cv2.warpAffine(src, matrix, (size[1], size[0]), flags=cv2.INTER_LINEAR)
    return warped


def apply_effects(image, params):
    s, lift_m, lift_p, vol_l = params
    img = image.copy()
    landmarks = get_landmarks(img)
    if landmarks is None:
        return img

    # --- Copiamos landmarks para modificarlos ---
    modified_landmarks = landmarks.copy()

    # Levantar mejillas (landmarks 50–54, 280–285)
    cheek_idx = list(range(50, 55)) + list(range(280, 286))
    modified_landmarks[cheek_idx, 1] -= int(lift_m * 20)

    # Levantar párpados (landmarks 159–161 y 386–388)
    eyelid_idx = [159, 160, 161, 386, 387, 388]
    modified_landmarks[eyelid_idx, 1] -= int(lift_p * 15)

    # Aumentar labios (landmarks 61–80 y 291–308)
    lips_idx = list(range(61, 81)) + list(range(291, 309))
    center_lips = np.mean(landmarks[lips_idx], axis=0)
    for i in lips_idx:
        direction = landmarks[i] - center_lips
        modified_landmarks[i] = landmarks[i] + direction * vol_l * 0.1

    # --- Warping geométrico simple ---
    warped = warp_image(img, landmarks, modified_landmarks, img.shape)

    # --- Suavizado para reducir arrugas ---
    ksize = int(5 + s * 10)
    sigma = 50 * s
    final_img = cv2.bilateralFilter(warped, ksize, sigma, sigma)

    return final_img


# ---------------------------
# Funcion fitness
# ---------------------------

def compute_fitness(image_orig, image_mod, landmarks_orig, landmarks_mod, alpha=0.5, beta=0.5):
    """Calcula fitness basado en varianza de textura y preservación de rasgos"""
    # 1. Varianza de textura (Laplaciano)
    gray = cv2.cvtColor(image_mod, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    var_texture = np.var(lap)
    var_texture_norm = np.clip(var_texture / 1000, 0, 1)  # normalizar arbitrariamente

    # 2. Distancia de landmarks
    dist_landmarks = np.mean(np.linalg.norm(landmarks_orig - landmarks_mod, axis=1))
    interocular = np.linalg.norm(landmarks_orig[33] - landmarks_orig[263])  # índices MediaPipe ojos
    dist_norm = np.clip(dist_landmarks / interocular, 0, 1)

    fitness = alpha * (1 - var_texture_norm) + beta * (1 - dist_norm)
    return fitness


# ---------------------------
# Algoritmo genético
# ---------------------------

def initialize_population(pop_size, param_bounds):
    """Inicializa individuos aleatorios"""
    population = []
    for _ in range(pop_size):
        indiv = [random.uniform(low, high) for (low, high) in param_bounds]
        population.append(indiv)
    return population


def tournament_selection(population, fitnesses, k=3):
    """Selecciona un individuo por torneo"""
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]


def arithmetic_crossover(parent1, parent2):
    """Crossover aritmético"""
    lambda_ = random.random()
    child = [lambda_ * p1 + (1 - lambda_) * p2 for p1, p2 in zip(parent1, parent2)]
    return child


def mutate(individual, param_bounds, mutation_rate=0.1):
    """Mutación ligera"""
    new_indiv = individual.copy()
    for i, (low, high) in enumerate(param_bounds):
        if random.random() < mutation_rate:
            perturb = (high - low) * random.uniform(-0.1, 0.1)
            new_indiv[i] = np.clip(new_indiv[i] + perturb, low, high)
    return new_indiv


def genetic_algorithm(image_orig, param_bounds, pop_size=15, generations=50):
    # Extraer landmarks originales **una sola vez**
    landmarks_orig = get_landmarks(image_orig)
    if landmarks_orig is None:
        raise ValueError("No se detectaron landmarks en la imagen original")

    # Inicializar población
    population = initialize_population(pop_size, param_bounds)

    best_individual = None
    best_fitness = -np.inf

    # Pre-calcular interocular para normalizar distancia
    interocular = np.linalg.norm(landmarks_orig[33] - landmarks_orig[263])

    for gen in range(generations):
        fitnesses = []
        modified_images = []

        # Evaluar fitness
        for indiv in population:
            img_mod = apply_effects(image_orig, indiv)

            landmarks_mod = get_landmarks(img_mod)
            if landmarks_mod is not None:
                fitness = compute_fitness(image_orig, img_mod, landmarks_orig, landmarks_mod, alpha=0.5, beta=0.5)
            else:
                fitness = 0  # penalización si no se detecta rostro
            fitnesses.append(fitness)
            modified_images.append(img_mod)

        # Guardar el mejor
        max_idx = np.argmax(fitnesses)
        if fitnesses[max_idx] > best_fitness:
            best_fitness = fitnesses[max_idx]
            best_individual = population[max_idx]
            best_image = modified_images[max_idx]

        # Nueva población
        new_population = []
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = arithmetic_crossover(parent1, parent2)
            child = mutate(child, param_bounds)
            new_population.append(child)

        population = new_population
        print(f"Generación {gen + 1}: Mejor fitness = {best_fitness:.4f}")

    return best_individual, best_fitness, best_image


if __name__ == "__main__":
    image_orig = cv2.imread("V2.jpg")  # ruta de la imagen
    param_bounds = [(0, 1), (-0.05, 0.05), (-0.05, 0.05), (0.9, 1.1)]

    # Ejecutar el algoritmo genético
    best_params, best_fit, best_img = genetic_algorithm(image_orig, param_bounds)

    # Extraer landmarks para dibujar sobre la original
    landmarks_orig = get_landmarks(image_orig)
    image_with_landmarks = draw_landmarks_points(image_orig, landmarks_orig)

    # Concatenar imágenes: original | con landmarks | resultado final
    # Asegurarse de que tengan la misma altura
    h = image_orig.shape[0]
    imgs_to_show = [image_orig, image_with_landmarks, best_img]
    resized_imgs = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in imgs_to_show]
    combined = np.hstack(resized_imgs)

    cv2.imshow("Original | Con Landmarks | Rejuvenecido", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
