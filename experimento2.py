import os
import cv2
import json
import time
import random
import numpy as np
import concurrent.futures as cf
from collections import deque

import mediapipe as mp

# ----------------------------------------------------------------------
# 1. Configuración MediaPipe y Utilidades Comunes
# ----------------------------------------------------------------------

# FaceMesh para una sola imagen
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def get_landmarks(image):
    """Detecta y retorna los landmarks de la cara."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = []
    for lm in results.multi_face_landmarks[0].landmark:
        x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
        landmarks.append([x, y])
    return np.array(landmarks, np.int32)

def interpupillary(lm):
    """Distancia interpupilar para normalización."""
    return float(np.linalg.norm(lm[33] - lm[263])) + 1e-6

def resize_image_keep_ratio(image, max_height=400, max_width=400):
    """Redimensiona manteniendo proporción."""
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def resize_max_side(image, max_side=256):
    """Redimensiona para que el lado mayor sea max_side. Devuelve imagen y factor escala relativo al original."""
    h, w = image.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return image.copy(), 1.0
    scale = max_side / float(s)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

def warp_image(src, src_points, dst_points, size):
    """Deformación global usando affine parcial ajustada con muchos puntos."""
    src_points = np.float32(src_points)
    dst_points = np.float32(dst_points)
    matrix, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
    warped = cv2.warpAffine(
        src, matrix, (size[1], size[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )
    return warped

# ----------------------------------------------------------------------
# 2. Pipeline de efectos rápido y vectorizado
# ----------------------------------------------------------------------

def apply_effects_caricature(image, landmarks_orig, params):
    """
    Versión rápida:
      - deformación por regiones vectorizada
      - filtros toon más baratos
    """
    eyes_scale, mouth_scale, chin_scale, head_scale, sat_scale = params
    img = image
    landmarks = landmarks_orig
    modified_landmarks = landmarks.copy()

    ipd = interpupillary(landmarks)
    sigma = np.float32(0.9 * ipd)
    max_px = np.float32(0.15 * ipd)

    def deform_region(idxs, center, scale):
        idxs_arr = np.array(idxs, dtype=np.int32)
        P = landmarks[idxs_arr].astype(np.float32)
        C = center.astype(np.float32)
        D = P - C
        w = np.exp(-np.sum(D * D, axis=1) / (2.0 * sigma * sigma)).astype(np.float32)
        w = w[:, None]
        disp = D * (np.float32(scale) - 1.0) * w
        nrm = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-6
        clip_factor = np.minimum(1.0, (max_px / nrm)).astype(np.float32)
        disp = disp * clip_factor
        modified = (P + disp).astype(np.int32)
        modified_landmarks[idxs_arr] = modified

    # Índices y centros
    eyes_idx = [33, 133, 362, 263]
    mouth_idx = list(range(61, 81)) + list(range(291, 309))
    chin_idx  = list(range(152, 178))
    head_idx  = [10, 338, 297, 332]

    center_eyes  = np.mean(landmarks[eyes_idx], axis=0)
    center_mouth = np.mean(landmarks[mouth_idx], axis=0)
    center_chin  = np.mean(landmarks[chin_idx], axis=0)
    center_head  = np.mean(landmarks[head_idx], axis=0)

    deform_region(eyes_idx,  center_eyes,  eyes_scale)
    deform_region(mouth_idx, center_mouth, mouth_scale)
    deform_region(chin_idx,  center_chin,  chin_scale)
    deform_region(head_idx,  center_head,  head_scale)

    warped = warp_image(img, landmarks, modified_landmarks, img.shape)

    # Saturación
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * np.float32(sat_scale), 0, 255).astype(np.uint8)
    tmp = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Toon barato
    toon = cv2.edgePreservingFilter(tmp, flags=1, sigma_s=30, sigma_r=0.35)
    toon = cv2.bilateralFilter(toon, 5, 50, 50)
    gray = cv2.cvtColor(toon, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.bitwise_not(edges)
    edges3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    final_img = cv2.bitwise_and(toon, edges3)

    return final_img, modified_landmarks

# ----------------------------------------------------------------------
# 3. Funciones de Aptitud
# ----------------------------------------------------------------------

def edge_nitidez_score(img):
    """Puntaje de nitidez (Tenengrad/Sobel)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag2 = gx * gx + gy * gy
    s = float(np.mean(mag2)) / (255.0 * 255.0)
    return float(np.tanh(3.0 * s))

def fitness_F1_dist_abs(landmarks_orig, landmarks_mod, image_shape, img_mod=None):
    """Fitness = Exageración - Distorsión en píxeles sin normalizar."""
    dist_eyes_orig  = np.linalg.norm(landmarks_orig[33] - landmarks_orig[263])
    dist_eyes_mod   = np.linalg.norm(landmarks_mod[33]  - landmarks_mod[263])
    exaggeration_eyes = dist_eyes_mod / (dist_eyes_orig + 1e-6)

    dist_mouth_orig = np.linalg.norm(landmarks_orig[61] - landmarks_orig[291])
    dist_mouth_mod  = np.linalg.norm(landmarks_mod[61]  - landmarks_mod[291])
    exaggeration_mouth = dist_mouth_mod / (dist_mouth_orig + 1e-6)

    E = np.clip((exaggeration_eyes + exaggeration_mouth) / 2.0, 0.0, 2.5)
    D = np.mean(np.linalg.norm(landmarks_orig - landmarks_mod, axis=1))
    alpha, beta = 0.7, 0.3
    return float(alpha * E - beta * D)

def fitness_F2_normalizada_nitidez(landmarks_orig, landmarks_mod, image_shape, img_mod):
    """Fitness = (Exageración - Distorsión normalizada) + Nitidez."""
    dist_eyes_orig  = np.linalg.norm(landmarks_orig[33] - landmarks_orig[263])
    dist_eyes_mod   = np.linalg.norm(landmarks_mod[33]  - landmarks_mod[263])
    exaggeration_eyes = dist_eyes_mod / (dist_eyes_orig + 1e-6)

    dist_mouth_orig = np.linalg.norm(landmarks_orig[61] - landmarks_orig[291])
    dist_mouth_mod  = np.linalg.norm(landmarks_mod[61]  - landmarks_mod[291])
    exaggeration_mouth = dist_mouth_mod / (dist_mouth_orig + 1e-6)

    E = np.clip((exaggeration_eyes + exaggeration_mouth) / 2.0, 0.0, 2.5)
    ipd = interpupillary(landmarks_orig)
    D = np.mean(np.linalg.norm(landmarks_orig - landmarks_mod, axis=1)) / ipd
    f_lm = 0.85 * E - 0.15 * D
    f_edge = edge_nitidez_score(img_mod)
    fitness = 0.9 * f_lm + 0.1 * f_edge
    return float(fitness)

def fitness_F3_normalizada_saturacion(landmarks_orig, landmarks_mod, image_shape, img_mod):
    """Fitness = (Exageración - Distorsión normalizada) - Penalización por saturación extrema."""
    dist_eyes_orig  = np.linalg.norm(landmarks_orig[33] - landmarks_orig[263])
    dist_eyes_mod   = np.linalg.norm(landmarks_mod[33]  - landmarks_mod[263])
    exaggeration_eyes = dist_eyes_mod / (dist_eyes_orig + 1e-6)

    dist_mouth_orig = np.linalg.norm(landmarks_orig[61] - landmarks_orig[291])
    dist_mouth_mod  = np.linalg.norm(landmarks_mod[61]  - landmarks_mod[291])
    exaggeration_mouth = dist_mouth_mod / (dist_mouth_orig + 1e-6)

    E = np.clip((exaggeration_eyes + exaggeration_mouth) / 2.0, 0.0, 2.5)
    ipd = interpupillary(landmarks_orig)
    D = np.mean(np.linalg.norm(landmarks_orig - landmarks_mod, axis=1)) / ipd
    f_lm = 0.85 * E - 0.15 * D

    hsv = cv2.cvtColor(img_mod, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1].astype(np.float32) / 255.0
    S_mean = np.mean(S)
    P_low  = max(0, 0.4 - S_mean) * 2.0
    P_high = max(0, S_mean - 0.8) * 1.5
    P_sat  = P_low + P_high

    fitness = f_lm - 0.2 * P_sat
    return float(fitness)

# ----------------------------------------------------------------------
# 4. Operadores Genéticos
# ----------------------------------------------------------------------

def arithmetic_crossover(parent1, parent2):
    """Cruce aritmético."""
    lambda_ = random.random()
    return [lambda_ * p1 + (1 - lambda_) * p2 for p1, p2 in zip(parent1, parent2)]

def simple_mutate(individual, param_bounds, mutation_rate=0.1):
    """Mutación simple limitada al 10% del rango."""
    new_indiv = individual.copy()
    for i, (low, high) in enumerate(param_bounds):
        if random.random() < mutation_rate:
            perturb = (high - low) * random.uniform(-0.1, 0.1)
            new_indiv[i] = float(np.clip(new_indiv[i] + perturb, low, high))
    return new_indiv

OPERATOR_SET_1 = (arithmetic_crossover, simple_mutate)

def uniform_crossover(parent1, parent2):
    """Cruce uniforme."""
    return [p1 if random.random() < 0.5 else p2 for p1, p2 in zip(parent1, parent2)]

def gaussian_mutate(individual, param_bounds, mutation_rate=0.2, sigma_factor=0.05):
    """Mutación gaussiana con sigma proporcional al rango."""
    new_indiv = individual.copy()
    for i, (low, high) in enumerate(param_bounds):
        if random.random() < mutation_rate:
            rng = high - low
            sigma = rng * sigma_factor
            perturb = random.gauss(0, sigma)
            new_indiv[i] = float(np.clip(new_indiv[i] + perturb, low, high))
    return new_indiv

OPERATOR_SET_2 = (uniform_crossover, gaussian_mutate)

def blend_crossover(parent1, parent2, alpha=0.5):
    """Cruce BLX-alpha."""
    child = []
    for p1, p2 in zip(parent1, parent2):
        d = abs(p1 - p2)
        min_val = min(p1, p2) - alpha * d
        max_val = max(p1, p2) + alpha * d
        child.append(random.uniform(min_val, max_val))
    return child

def abarcador_mutate(individual, param_bounds, mutation_rate=0.1):
    """Mutación que reinicia el gen dentro de su rango."""
    new_indiv = individual.copy()
    for i, (low, high) in enumerate(param_bounds):
        if random.random() < mutation_rate:
            new_indiv[i] = random.uniform(low, high)
    return new_indiv

OPERATOR_SET_3 = (blend_crossover, abarcador_mutate)

# ----------------------------------------------------------------------
# 5. Algoritmo Genético con paralelismo, downscale y early stopping
# ----------------------------------------------------------------------

def initialize_population(pop_size, param_bounds):
    return [[random.uniform(low, high) for (low, high) in param_bounds] for _ in range(pop_size)]

def tournament_selection(population, fitnesses, k=3):
    indexed = list(zip(population, fitnesses))
    selected = random.sample(indexed, k)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]

def genetic_algorithm_caricature(
        image_orig,
        landmarks_orig,
        param_bounds,
        pop_size,
        generations,
        fitness_func,
        crossover_func,
        mutation_func,
        mutation_rate=0.1,
        ga_max_side=256,
        patience=15,
        epsilon_improve=1e-4,
        n_workers=None
):
    start_time = time.time()

    # Imagen de trabajo pequeña para evaluar rápido
    work_img, scale_img = resize_max_side(image_orig, max_side=ga_max_side)
    # Escala homogénea por mantener proporción
    s_w = work_img.shape[1] / image_orig.shape[1]
    work_lm = (landmarks_orig.astype(np.float32) * s_w).astype(np.int32)

    population = initialize_population(pop_size, param_bounds)
    best_individual, best_fitness = None, -np.inf
    best_fitness_history = []
    diversity_history = []

    if n_workers is None:
        try:
            n_workers = min(32, (os.cpu_count() or 4) + 4)
        except:
            n_workers = 8

    no_improve_count = 0

    def eval_individual(indiv):
        img_mod, landmarks_mod = apply_effects_caricature(work_img, work_lm, indiv)
        if landmarks_mod is None:
            return -1.0, img_mod
        fit = fitness_func(work_lm, landmarks_mod, work_img.shape, img_mod)
        return float(fit), img_mod

    for gen in range(generations):
        # Evaluación paralela
        with cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(eval_individual, population))

        fitnesses = [r[0] for r in results]

        # Diversidad
        diversity = float(np.mean(np.std(np.array(population, dtype=np.float32), axis=0)))
        diversity_history.append(diversity)

        # Mejor de la generación
        max_idx = int(np.argmax(fitnesses))
        current_best_individual = population[max_idx]
        current_best_fitness = float(fitnesses[max_idx])

        improved = current_best_fitness > best_fitness + epsilon_improve
        if improved:
            best_fitness = current_best_fitness
            best_individual = current_best_individual
            no_improve_count = 0
        else:
            no_improve_count += 1

        best_fitness_history.append(best_fitness)

        # Early stopping
        if no_improve_count >= patience:
            break

        # Nueva población con elitismo
        new_population = [current_best_individual]
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover_func(parent1, parent2)
            child = mutation_func(child, param_bounds, mutation_rate)
            # Clip a límites
            for i, (low, high) in enumerate(param_bounds):
                v = child[i]
                if v < low: v = low
                if v > high: v = high
                child[i] = float(v)
            new_population.append(child)
        population = new_population

    total_time = time.time() - start_time

    # Convergencia por historial
    convergence_gen = len(best_fitness_history)
    for i in range(convergence_gen - 1, 0, -1):
        if abs(best_fitness_history[i] - best_fitness_history[i - 1]) > epsilon_improve:
            convergence_gen = i + 1
            break

    return best_individual, best_fitness, best_fitness_history, total_time, float(np.mean(diversity_history)), convergence_gen

# ----------------------------------------------------------------------
# 6. Tablas de funciones y operadores
# ----------------------------------------------------------------------

FITNESS_FUNCTIONS = {
    "F1_Dist_Abs": fitness_F1_dist_abs,
    "F2_Norm_Nitidez": fitness_F2_normalizada_nitidez,
    "F3_Norm_Saturacion": fitness_F3_normalizada_saturacion
}

OPERATOR_SETS = {
    "OP1_Arith_SimpleMut": OPERATOR_SET_1,
    "OP2_Uniform_GaussMut": OPERATOR_SET_2,
    "OP3_Blend_AbarcadorMut": OPERATOR_SET_3
}

# ----------------------------------------------------------------------
# 7. Experimentación sistemática con re-render en alta
# ----------------------------------------------------------------------

def systematic_experimentation(image_orig, param_bounds, pop_size=30, generations=100, num_runs=10):
    print("\n--- INICIANDO EXPERIMENTACIÓN SISTEMÁTICA ---")

    # Landmarks una vez
    landmarks_orig = get_landmarks(image_orig)
    if landmarks_orig is None:
        raise ValueError("No se detectaron landmarks en la imagen original")

    if not os.path.exists("results"):
        os.makedirs("results")

    results = {}

    for op_name, (crossover_func, mutation_func) in OPERATOR_SETS.items():
        for fit_name, fitness_func in FITNESS_FUNCTIONS.items():

            combination_name = f"{op_name}_{fit_name}"
            print(f"\n>>>> Ejecutando Combinación: {combination_name}")

            run_metrics = []

            for seed in range(num_runs):
                random.seed(seed)
                np.random.seed(seed)
                print(f"  > Ejecución {seed + 1}/{num_runs} (Seed: {seed})")

                best_params, best_fit, history, run_time, diversity, conv_gen = genetic_algorithm_caricature(
                    image_orig,
                    landmarks_orig,
                    param_bounds,
                    pop_size,
                    generations,
                    fitness_func,
                    crossover_func,
                    mutation_func,
                    mutation_rate=0.1,
                    ga_max_side=256,
                    patience=15,
                    epsilon_improve=1e-4
                )

                # Re-render del mejor en la resolución original una sola vez
                best_image_run_full, _ = apply_effects_caricature(image_orig, landmarks_orig, best_params)
                img_filename = f"results/{combination_name}_seed{seed}_best.jpg"
                cv2.imwrite(img_filename, best_image_run_full)

                metric = {
                    "seed": seed,
                    "best_fitness": best_fit,
                    "best_params": best_params,
                    "convergence_gen": conv_gen,
                    "total_time": run_time,
                    "avg_diversity": diversity,
                    "fitness_history": history,
                    "best_image_path": img_filename
                }
                run_metrics.append(metric)

            results[combination_name] = run_metrics

    output_filename = "results/experimental_results.json"
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n--- EXPERIMENTACIÓN FINALIZADA ---")
    print(f"Resultados guardados en: {output_filename}")
    print("Las mejores imágenes por corrida están en la carpeta 'results'.")
    return results

# ----------------------------------------------------------------------
# 8. Análisis numérico
# ----------------------------------------------------------------------

def analyze_results(results, generations):
    print("\n--- ANÁLISIS CUANTITATIVO (5.1) ---")
    comparison_data = {}

    for combination, runs in results.items():
        best_fits = [r["best_fitness"] for r in runs]
        conv_gens = [r["convergence_gen"] for r in runs]
        times = [r["total_time"] for r in runs]
        history_matrix = np.array([r["fitness_history"] for r in runs])
        avg_history = np.mean(history_matrix, axis=0)

        comparison_data[combination] = {
            "Avg_Best_Fitness": float(np.mean(best_fits)),
            "Std_Best_Fitness": float(np.std(best_fits)),
            "Avg_Convergence_Gen": float(np.mean(conv_gens)),
            "Std_Convergence_Gen": float(np.std(conv_gens)),
            "Stability_Fitness_Metric": float(np.std(best_fits) / (np.mean(best_fits) + 1e-9)),
            "Avg_Run_Time": float(np.mean(times)),
            "Avg_Diversity": float(np.mean([r["avg_diversity"] for r in runs])),
            "Avg_Fitness_History": avg_history.tolist()
        }

        print(f"\n{combination}:")
        print(f"  Mejor Aptitud (Prom/Std): {comparison_data[combination]['Avg_Best_Fitness']:.4f} / {comparison_data[combination]['Std_Best_Fitness']:.4f}")
        print(f"  Convergencia (Prom/Std): {comparison_data[combination]['Avg_Convergence_Gen']:.1f} / {comparison_data[combination]['Std_Convergence_Gen']:.1f} gen")
        print(f"  Estabilidad (CV): {comparison_data[combination]['Stability_Fitness_Metric']:.4f}")

    print("\n--- ANÁLISIS CUALITATIVO (5.2) ---")
    print("Revisa 'results/' y compara exageración, diversidad y correspondencia numérica vs visual.")
    return comparison_data

# ----------------------------------------------------------------------
# 9. Ejecución Principal
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Cambia por tu archivo
    image_file = "Robert.jpg"

    image_orig = cv2.imread(image_file)
    if image_orig is None:
        print(f"ERROR: No se pudo cargar la imagen '{image_file}'. Revisa la ruta o el nombre del archivo.")
        raise SystemExit(1)

    image_orig = resize_image_keep_ratio(image_orig, max_height=400, max_width=400)

    # Límites de los Genes
    param_bounds = [
        (1.0, 1.5),  # ojos
        (0.8, 1.3),  # boca
        (0.8, 1.2),  # mentón
        (0.9, 1.2),  # cabeza
        (0.8, 1.5)   # saturación
    ]

    POP_SIZE = 30
    GENERATIONS = 100
    NUM_RUNS = 10

    experiment_results = systematic_experimentation(
        image_orig,
        param_bounds,
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        num_runs=NUM_RUNS
    )

    analysis = analyze_results(experiment_results, GENERATIONS)
    print("\nFin del script de experimentación.")
