import cv2
import numpy as np
import random
import mediapipe as mp
import time
import os
import json
from collections import deque  # Para la diversidad

# ----------------------------------------------------------------------
# 1. Configuración MediaPipe y Utilidades Comunes
# ----------------------------------------------------------------------

# Configuración MediaPipe (versión mejorada de la 'segunda version')
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
    """Calcula la distancia interpupilar normalizada (para escala facial)."""
    # Puntos laterales de los ojos: 33 y 263
    return float(np.linalg.norm(lm[33] - lm[263])) + 1e-6


def warp_image(src, src_points, dst_points, size):
    """Aplica la deformación geométrica (affine partial)."""
    src_points = np.float32(src_points)
    dst_points = np.float32(dst_points)
    matrix, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
    warped = cv2.warpAffine(
        src, matrix, (size[1], size[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )
    return warped


def apply_effects_caricature(image, landmarks_orig, params):
    """Aplica deformaciones tipo caricatura y filtros según los parámetros."""
    eyes_scale, mouth_scale, chin_scale, head_scale, sat_scale = params
    img = image.copy()
    landmarks = landmarks_orig  # Usamos los landmarks pre-detectados

    modified_landmarks = landmarks.copy()
    ipd = interpupillary(landmarks)
    sigma = 0.9 * ipd
    max_px = 0.15 * ipd

    def _apply_region(idxs, center, scale):
        for i in idxs:
            direction = landmarks[i] - center
            w = np.exp(-(np.linalg.norm(direction) ** 2) / (2 * (sigma ** 2)))
            disp = direction * (scale - 1.0) * w
            n = np.linalg.norm(disp)
            if n > max_px:
                disp = disp * (max_px / n)
            modified_landmarks[i] = (landmarks[i] + disp).astype(np.int32)

    # --- Deformación ---
    eyes_idx = [33, 133, 362, 263]
    center_eyes = np.mean(landmarks[eyes_idx], axis=0)
    _apply_region(eyes_idx, center_eyes, eyes_scale)

    mouth_idx = list(range(61, 81)) + list(range(291, 309))
    center_mouth = np.mean(landmarks[mouth_idx], axis=0)
    _apply_region(mouth_idx, center_mouth, mouth_scale)

    chin_idx = list(range(152, 178))
    center_chin = np.mean(landmarks[chin_idx], axis=0)
    _apply_region(chin_idx, center_chin, chin_scale)

    head_idx = [10, 338, 297, 332]
    center_head = np.mean(landmarks[head_idx], axis=0)
    _apply_region(head_idx, center_head, head_scale)

    warped = warp_image(img, landmarks, modified_landmarks, img.shape)

    # --- Saturación ---
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * sat_scale, 0, 255).astype(np.uint8)
    cartoon = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # --- Filtro Toon (segunda versión: mean-shift + Canny invertido) ---
    smooth = cv2.bilateralFilter(cartoon, 9, 120, 120)
    quant = cv2.pyrMeanShiftFiltering(smooth, sp=8, sr=20)
    gray = cv2.cvtColor(quant, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.bitwise_not(edges)
    edges3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    final_img = cv2.bitwise_and(quant, edges3)

    return final_img, modified_landmarks


# ----------------------------------------------------------------------
# 2. Funciones de Aptitud (Fitness) - 3 Diferentes
# ----------------------------------------------------------------------

# 2.1. F1: Basada en Distancia Absoluta (Primera versión con Distorsión en pixeles)
def fitness_F1_dist_abs(landmarks_orig, landmarks_mod, image_shape, img_mod=None):
    """
    Fitness = Exageración (E) - Distorsión (D). D no normalizada.
    """
    # Exageración (E): Mide el cambio relativo en las distancias clave
    dist_eyes_orig = np.linalg.norm(landmarks_orig[33] - landmarks_orig[263])
    dist_eyes_mod = np.linalg.norm(landmarks_mod[33] - landmarks_mod[263])
    exaggeration_eyes = dist_eyes_mod / (dist_eyes_orig + 1e-6)

    dist_mouth_orig = np.linalg.norm(landmarks_orig[61] - landmarks_orig[291])
    dist_mouth_mod = np.linalg.norm(landmarks_mod[61] - landmarks_mod[291])
    exaggeration_mouth = dist_mouth_mod / (dist_mouth_orig + 1e-6)

    E = np.clip((exaggeration_eyes + exaggeration_mouth) / 2.0, 0.0, 2.5)

    # Distorsión (D): Distancia promedio de desplazamiento de todos los landmarks (en pixeles)
    D = np.mean(np.linalg.norm(landmarks_orig - landmarks_mod, axis=1))

    alpha, beta = 0.7, 0.3
    return float(alpha * E - beta * D)


# 2.2. F2: Normalizada + Componente Visual de Nitidez (Segunda versión mejorada)
def edge_nitidez_score(img):
    """Puntaje de nitidez (Tenengrad/Sobel) para premiar bordes limpios."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag2 = gx * gx + gy * gy
    s = float(np.mean(mag2)) / (255.0 * 255.0)
    return float(np.tanh(3.0 * s))  # Compresión suave a [0,1]


def fitness_F2_normalizada_nitidez(landmarks_orig, landmarks_mod, image_shape, img_mod):
    """
    Fitness = (Exageración - Distorsión Normalizada) + Nitidez.
    """
    # Exageración (E)
    dist_eyes_orig = np.linalg.norm(landmarks_orig[33] - landmarks_orig[263])
    dist_eyes_mod = np.linalg.norm(landmarks_mod[33] - landmarks_mod[263])
    exaggeration_eyes = dist_eyes_mod / (dist_eyes_orig + 1e-6)
    dist_mouth_orig = np.linalg.norm(landmarks_orig[61] - landmarks_orig[291])
    dist_mouth_mod = np.linalg.norm(landmarks_mod[61] - landmarks_mod[291])
    exaggeration_mouth = dist_mouth_mod / (dist_mouth_orig + 1e-6)
    E = np.clip((exaggeration_eyes + exaggeration_mouth) / 2.0, 0.0, 2.5)

    # Distorsión (D): Normalizada por distancia interpupilar (ipd)
    ipd = interpupillary(landmarks_orig)
    D = np.mean(np.linalg.norm(landmarks_orig - landmarks_mod, axis=1)) / ipd

    # Fitness Geométrica
    f_lm = 0.85 * E - 0.15 * D

    # Fitness Visual (Nitidez)
    f_edge = edge_nitidez_score(img_mod)

    # Fitness Compuesta (ajuste de pesos para que f_edge no domine)
    fitness = 0.9 * f_lm + 0.1 * f_edge
    return float(fitness)


# 2.3. F3: Normalizada + Penalización Fuerte por Desviación de Saturación
def fitness_F3_normalizada_saturacion(landmarks_orig, landmarks_mod, image_shape, img_mod):
    """
    Fitness = (Exageración - Distorsión Normalizada) - Penalización de Saturación.
    Penaliza fuertemente la desviación excesiva del canal HUE y el uso de valores
    extremos de saturación.
    """
    # Exageración (E) y Distorsión (D) - Componente Geométrico F2
    dist_eyes_orig = np.linalg.norm(landmarks_orig[33] - landmarks_orig[263])
    dist_eyes_mod = np.linalg.norm(landmarks_mod[33] - landmarks_mod[263])
    exaggeration_eyes = dist_eyes_mod / (dist_eyes_orig + 1e-6)
    dist_mouth_orig = np.linalg.norm(landmarks_orig[61] - landmarks_orig[291])
    dist_mouth_mod = np.linalg.norm(landmarks_mod[61] - landmarks_mod[291])
    exaggeration_mouth = dist_mouth_mod / (dist_mouth_orig + 1e-6)
    E = np.clip((exaggeration_eyes + exaggeration_mouth) / 2.0, 0.0, 2.5)
    ipd = interpupillary(landmarks_orig)
    D = np.mean(np.linalg.norm(landmarks_orig - landmarks_mod, axis=1)) / ipd

    f_lm = 0.85 * E - 0.15 * D

    # Penalización de Saturación (P_sat)
    hsv = cv2.cvtColor(img_mod, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1].astype(np.float32) / 255.0  # Normalizar Sat a [0, 1]

    # Penaliza S muy baja (gris) o muy alta (sobre-saturada), ideal en [0.4, 0.8]
    S_mean = np.mean(S)
    P_low = max(0, 0.4 - S_mean) * 2.0  # Penaliza si es muy baja
    P_high = max(0, S_mean - 0.8) * 1.5  # Penaliza si es muy alta
    P_sat = P_low + P_high

    # Fitness
    fitness = f_lm - 0.2 * P_sat
    return float(fitness)


# ----------------------------------------------------------------------
# 3. Operadores Genéticos - 3 Diferentes
# ----------------------------------------------------------------------

# OPERADOR BASE: Cruce Aritmético + Mutación Simple (versión 1)
def arithmetic_crossover(parent1, parent2):
    """Cruce Aritmético: Child = lambda*P1 + (1-lambda)*P2"""
    lambda_ = random.random()
    child = [lambda_ * p1 + (1 - lambda_) * p2 for p1, p2 in zip(parent1, parent2)]
    return child


def simple_mutate(individual, param_bounds, mutation_rate=0.1):
    """Mutación Simple con Perturbación Limitada (versión 1)"""
    new_indiv = individual.copy()
    for i, (low, high) in enumerate(param_bounds):
        if random.random() < mutation_rate:
            # Perturbación limitada al 10% del rango total de los límites
            perturb = (high - low) * random.uniform(-0.1, 0.1)
            new_indiv[i] = np.clip(new_indiv[i] + perturb, low, high)
    return new_indiv


# 3.1. OP1: Aritmético + Simple Mutación (Usada en tus versiones)
OPERATOR_SET_1 = (arithmetic_crossover, simple_mutate)


# 3.2. OP2: Cruce Uniforme (Simple) + Mutación de Desplazamiento Gaussiano
def uniform_crossover(parent1, parent2):
    """Cruce Uniforme: cada gen viene de P1 o P2 con 50% de probabilidad."""
    child = []
    for p1, p2 in zip(parent1, parent2):
        child.append(p1 if random.random() < 0.5 else p2)
    return child


def gaussian_mutate(individual, param_bounds, mutation_rate=0.2, sigma_factor=0.05):
    """
    Mutación Gaussiana: Añade ruido gaussiano con sigma proporcional
    al rango de los límites.
    """
    new_indiv = individual.copy()
    for i, (low, high) in enumerate(param_bounds):
        if random.random() < mutation_rate:
            range_ = high - low
            sigma = range_ * sigma_factor
            perturb = random.gauss(0, sigma)
            new_indiv[i] = float(np.clip(new_indiv[i] + perturb, low, high))
    return new_indiv


OPERATOR_SET_2 = (uniform_crossover, gaussian_mutate)


# 3.3. OP3: Cruce Heurístico + Mutación Fuerte/Abarcadora
def blend_crossover(parent1, parent2, alpha=0.5):
    """
    Cruce Blended (BLX-alpha): El hijo es una mezcla de P1 y P2 que
    puede extenderse fuera de los límites de los padres (exploratorio).
    """
    child = []
    for p1, p2 in zip(parent1, parent2):
        d = abs(p1 - p2)
        min_val = min(p1, p2) - alpha * d
        max_val = max(p1, p2) + alpha * d
        # Se clip-eará al final del AG por los param_bounds
        child.append(random.uniform(min_val, max_val))
    return child


def abarcador_mutate(individual, param_bounds, mutation_rate=0.1):
    """
    Mutación Abarcadora: Con una baja probabilidad, se reinicia
    el gen completamente dentro de su rango (fuerte exploración).
    """
    new_indiv = individual.copy()
    for i, (low, high) in enumerate(param_bounds):
        if random.random() < mutation_rate:
            # Re-inicialización total del gen (fuerte)
            new_indiv[i] = random.uniform(low, high)
    return new_indiv


OPERATOR_SET_3 = (blend_crossover, abarcador_mutate)


# ----------------------------------------------------------------------
# 4. Algoritmo Genético Base (Generalizado)
# ----------------------------------------------------------------------

def initialize_population(pop_size, param_bounds):
    """Inicializa una población aleatoria dentro de los límites."""
    population = []
    for _ in range(pop_size):
        indiv = [random.uniform(low, high) for (low, high) in param_bounds]
        population.append(indiv)
    return population


def tournament_selection(population, fitnesses, k=3):
    """Selección por Torneo."""
    # En lugar de seleccionar indices, seleccionamos el objeto (individuo, fitness)
    indexed_pop_fit = list(zip(population, fitnesses))
    selected = random.sample(indexed_pop_fit, k)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]  # Retorna el individuo


def genetic_algorithm_caricature(
        image_orig,
        landmarks_orig,
        param_bounds,
        pop_size,
        generations,
        fitness_func,
        crossover_func,
        mutation_func,
        mutation_rate=0.1
):
    """
    Algoritmo Genético generalizado para experimentación.
    Retorna: mejor individuo, mejor fitness, lista de mejores fitness por generación,
             tiempo total, y diversidad (desviación estándar promedio de la población).
    """
    start_time = time.time()

    population = initialize_population(pop_size, param_bounds)
    best_individual, best_fitness = None, -np.inf
    best_fitness_history = []

    # Para el cálculo de la diversidad
    diversity_history = []

    # Generaciones
    for gen in range(generations):
        fitnesses, modified_images = [], []

        # 1. Evaluación de la población
        current_pop_params = []
        for indiv in population:
            # Aplica efectos SÓLO una vez por individuo
            img_mod, landmarks_mod = apply_effects_caricature(image_orig, landmarks_orig, indiv)

            # Penaliza si no hubo detección (aunque los landmarks se pre-detectan)
            if landmarks_mod is None:
                fitness = -1.0
            else:
                # Usa la función de fitness dinámica
                fitness = fitness_func(landmarks_orig, landmarks_mod, image_orig.shape, img_mod)

            fitnesses.append(fitness)
            modified_images.append(img_mod)
            current_pop_params.append(indiv)

        # Cálculo de la Diversidad: Desviación Estándar Promedio de los Parámetros
        diversity = np.mean(np.std(np.array(current_pop_params), axis=0))
        diversity_history.append(diversity)

        # 2. Búsqueda y registro del mejor individuo actual
        max_idx = np.argmax(fitnesses)
        current_best_individual = population[max_idx]
        current_best_fitness = fitnesses[max_idx]

        # Actualizar el mejor global
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual

        best_fitness_history.append(best_fitness)

        # 3. Nueva generación con ELITISMO (Mejor de la generación)
        new_population = [current_best_individual]

        # Llenamos el resto con cruce y mutación
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            # Cruce
            child = crossover_func(parent1, parent2)

            # Mutación
            child = mutation_func(child, param_bounds, mutation_rate)

            # Asegurar que el hijo respete los límites después del cruce/mutación
            final_child = []
            for i, val in enumerate(child):
                low, high = param_bounds[i]
                final_child.append(float(np.clip(val, low, high)))

            new_population.append(final_child)

        population = new_population

    end_time = time.time()
    total_time = end_time - start_time

    # Calcular convergencia: primera generación donde el mejor fitness se estanca (dentro de un epsilon)
    convergence_gen = generations
    epsilon = 1e-4  # Umbral de cambio para considerarse convergido
    for i in range(generations - 1, 0, -1):
        if abs(best_fitness_history[i] - best_fitness_history[i - 1]) > epsilon:
            convergence_gen = i + 1
            break

    # Retorna el mejor individuo (el mejor global), su fitness, el historial, el tiempo y la diversidad.
    return best_individual, best_fitness, best_fitness_history, total_time, np.mean(diversity_history), convergence_gen


# ----------------------------------------------------------------------
# 5. Experimentación Sistemática (Paso 6 y 7)
# ----------------------------------------------------------------------

# Diccionarios de las opciones para la experimentación
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


def systematic_experimentation(image_orig, param_bounds, pop_size=30, generations=100, num_runs=10):
    print("\n--- INICIANDO EXPERIMENTACIÓN SISTEMÁTICA ---")

    # Pre-calcular landmarks una vez
    landmarks_orig = get_landmarks(image_orig)
    if landmarks_orig is None:
        raise ValueError("No se detectaron landmarks en la imagen original")

    results = {}

    # Iterar sobre todas las combinaciones de operadores y fitness
    for op_name, (crossover_func, mutation_func) in OPERATOR_SETS.items():
        for fit_name, fitness_func in FITNESS_FUNCTIONS.items():

            combination_name = f"{op_name}_{fit_name}"
            print(f"\n>>>> Ejecutando Combinación: {combination_name}")

            run_metrics = []

            # Paso 6: Ejecutar NUM_RUNS veces con semillas diferentes
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
                    mutation_rate=0.1  # Mutación fija
                )

                # Guardar el mejor individuo de cada ejecución
                best_image_run, _ = apply_effects_caricature(image_orig, landmarks_orig, best_params)

                # Nombre del archivo para la imagen
                img_filename = f"results/{combination_name}_seed{seed}_best.jpg"
                cv2.imwrite(img_filename, best_image_run)

                # Registrar métricas
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

            # Paso 7: Almacenar resultados de la combinación
            results[combination_name] = run_metrics

    # Guardar todos los resultados en un archivo JSON
    if not os.path.exists("results"):
        os.makedirs("results")

    output_filename = "results/experimental_results.json"
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n--- EXPERIMENTACIÓN FINALIZADA ---")
    print(f"Resultados guardados en: {output_filename}")
    print("Las mejores imágenes por corrida están en la carpeta 'results'.")

    return results


def analyze_results(results, generations):
    """
    Realiza el Análisis Cuantitativo (5.1).
    """
    print("\n--- ANÁLISIS CUANTITATIVO (5.1) ---")

    comparison_data = {}

    for combination, runs in results.items():
        best_fits = [r["best_fitness"] for r in runs]
        conv_gens = [r["convergence_gen"] for r in runs]
        times = [r["total_time"] for r in runs]

        # Comparar curvas de convergencia (Promedio del Historial)
        history_matrix = np.array([r["fitness_history"] for r in runs])
        avg_history = np.mean(history_matrix, axis=0)

        comparison_data[combination] = {
            "Avg_Best_Fitness": np.mean(best_fits),
            "Std_Best_Fitness": np.std(best_fits),  # Analizar varianza
            "Avg_Convergence_Gen": np.mean(conv_gens),
            "Std_Convergence_Gen": np.std(conv_gens),
            "Stability_Fitness_Metric": np.std(best_fits) / np.mean(best_fits),  # Estabilidad (CV)
            "Avg_Run_Time": np.mean(times),
            "Avg_Diversity": np.mean([r["avg_diversity"] for r in runs]),
            "Avg_Fitness_History": avg_history.tolist()  # Para graficar la curva
        }

        print(f"\n{combination}:")
        print(
            f"  Mejor Aptitud (Prom/Std): {comparison_data[combination]['Avg_Best_Fitness']:.4f} / {comparison_data[combination]['Std_Best_Fitness']:.4f}")
        print(
            f"  Convergencia (Prom/Std): {comparison_data[combination]['Avg_Convergence_Gen']:.1f} / {comparison_data[combination]['Std_Convergence_Gen']:.1f} (Generaciones)")
        print(f"  Estabilidad (CV): {comparison_data[combination]['Stability_Fitness_Metric']:.4f}")

    # Este análisis se puede complementar con graficado (ej. matplotlib) que no es posible en este entorno.

    print("\n--- ANÁLISIS CUALITATIVO (5.2) ---")
    print("Para la evaluación visual, revisa las imágenes guardadas en 'results/'.")
    print("Busca patrones como:")
    print("1. ¿Qué función de aptitud tiende a exagerar más los rasgos?")
    print("2. ¿Qué operador genético produce resultados más diversos/extraños?")
    print("3. ¿Los individuos con mejor fitness visual se corresponden con la mejor aptitud numérica?")

    return comparison_data


def resize_image_keep_ratio(image, max_height=400, max_width=400):
    """Redimensiona la imagen manteniendo la proporción."""
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


# ----------------------------------------------------------------------
# 6. Ejecución Principal
# ----------------------------------------------------------------------
if __name__ == "__main__":

    # ⚠️ ASEGÚRATE DE TENER LA IMAGEN "Robert.jpg" O "maluma.jpg" EN LA MISMA CARPETA
    image_file = "Robert.jpg"

    image_orig = cv2.imread(image_file)

    if image_orig is None:
        print(f"ERROR: No se pudo cargar la imagen '{image_file}'. Revisa la ruta o el nombre del archivo.")
        exit()

    image_orig = resize_image_keep_ratio(image_orig, max_height=400, max_width=400)

    # Límites de los Parámetros (Genes)
    param_bounds = [
        (1.0, 1.5),  # [0] ojos (escala)
        (0.8, 1.3),  # [1] boca (escala)
        (0.8, 1.2),  # [2] mentón (escala)
        (0.9, 1.2),  # [3] cabeza (escala)
        (0.8, 1.5)  # [4] saturación (factor)
    ]

    # Parámetros para la Experimentación
    POP_SIZE = 30
    GENERATIONS = 100
    NUM_RUNS = 10  # El número de ejecuciones por combinación

    # --- Ejecutar la Experimentación Sistemática ---
    experiment_results = systematic_experimentation(
        image_orig,
        param_bounds,
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        num_runs=NUM_RUNS
    )

    # --- Realizar el Análisis Cuantitativo y Cualitativo ---
    analysis = analyze_results(experiment_results, GENERATIONS)

    print("\nFin del script de experimentación.")