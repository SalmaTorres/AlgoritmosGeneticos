# experimento_fast.py
import os, cv2, json, time, random, numpy as np
import concurrent.futures as cf
import mediapipe as mp

# ============================= PARÁMETROS RÁPIDOS =============================
IMAGE_FILE     = "Robert.jpg"     # Cambia por tu imagen
POP_SIZE       = 20               # 20 es suficiente con early stopping
GENERATIONS    = 60               # menos generaciones
NUM_RUNS       = 5                # 5 corridas por combinación
MUTATION_RATE  = 0.10
GA_MAX_SIDE    = 256              # eval en baja resolución
PATIENCE       = 10               # early stopping si no mejora
EPS_IMPROVE    = 1e-4
TOURNAMENT_K   = 3
ELITISM_K      = 1
FAST_EVAL      = True             # filtros más ligeros durante el GA
RUN_GRID       = False            # True para probar 3x3 combinaciones
RESULTS_DIR    = "results_fast"

# Opcional: reproducibilidad vs desempeño
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# =============================================================================

# -------------------------- MediaPipe FaceMesh -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def get_landmarks(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None
    lm = []
    for p in results.multi_face_landmarks[0].landmark:
        x, y = int(p.x * image.shape[1]), int(p.y * image.shape[0])
        lm.append([x, y])
    return np.array(lm, np.int32)

def interpupillary(lm):  # distancia interpupilar
    return float(np.linalg.norm(lm[33] - lm[263])) + 1e-6

def resize_keep_ratio(image, max_h=400, max_w=400):
    h, w = image.shape[:2]
    s = min(max_w / w, max_h / h)
    return cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

def resize_max_side(image, max_side=256):
    h, w = image.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return image.copy(), 1.0
    scale = max_side / float(side)
    return cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA), scale

def warp_image(src, src_points, dst_points, size):
    src_points = np.float32(src_points)
    dst_points = np.float32(dst_points)
    M, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
    return cv2.warpAffine(src, M, (size[1], size[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

# ----------------------- Caricatura (rápida y vectorizada) -------------------
def apply_effects_caricature(image, landmarks_orig, params, fast=False):
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
        w = np.exp(-np.sum(D*D, axis=1) / (2.0 * sigma * sigma)).astype(np.float32)
        disp = D * (np.float32(scale) - 1.0) * w[:, None]
        nrm = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-6
        disp *= np.minimum(1.0, (max_px / nrm)).astype(np.float32)
        modified_landmarks[idxs_arr] = (P + disp).astype(np.int32)

    eyes_idx  = [33, 133, 362, 263]
    mouth_idx = list(range(61, 81)) + list(range(291, 309))
    chin_idx  = list(range(152, 178))
    head_idx  = [10, 338, 297, 332]

    center_eyes  = np.mean(landmarks[eyes_idx],  axis=0)
    center_mouth = np.mean(landmarks[mouth_idx], axis=0)
    center_chin  = np.mean(landmarks[chin_idx],  axis=0)
    center_head  = np.mean(landmarks[head_idx],  axis=0)

    deform_region(eyes_idx,  center_eyes,  eyes_scale)
    deform_region(mouth_idx, center_mouth, mouth_scale)
    deform_region(chin_idx,  center_chin,  chin_scale)
    deform_region(head_idx,  center_head,  head_scale)

    warped = warp_image(img, landmarks, modified_landmarks, img.shape)

    # Saturación
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * np.float32(sat_scale), 0, 255).astype(np.uint8)
    tmp = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if fast:
        # Pipeline muy barato para el GA
        gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        edges = cv2.bitwise_not(edges)
        return cv2.bitwise_and(tmp, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)), modified_landmarks
    else:
        # Toon completo, igualmente liviano (sin mean-shift)
        toon = cv2.edgePreservingFilter(tmp, flags=1, sigma_s=30, sigma_r=0.35)
        toon = cv2.bilateralFilter(toon, 5, 50, 50)
        gray = cv2.cvtColor(toon, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        edges = cv2.bitwise_not(edges)
        return cv2.bitwise_and(toon, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)), modified_landmarks

# ----------------------------- Fitness ---------------------------------------
def edge_nitidez_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag2 = gx * gx + gy * gy
    s = float(np.mean(mag2)) / (255.0 * 255.0)
    return float(np.tanh(3.0 * s))

def fitness_F3_normalizada_saturacion(lm0, lm1, image_shape, img_mod):
    dist_eyes0  = np.linalg.norm(lm0[33] - lm0[263])
    dist_eyes1  = np.linalg.norm(lm1[33] - lm1[263])
    exaggeration_eyes = dist_eyes1 / (dist_eyes0 + 1e-6)
    dist_mouth0 = np.linalg.norm(lm0[61] - lm0[291])
    dist_mouth1 = np.linalg.norm(lm1[61] - lm1[291])
    exaggeration_mouth = dist_mouth1 / (dist_mouth0 + 1e-6)
    E = np.clip((exaggeration_eyes + exaggeration_mouth)/2.0, 0.0, 2.5)

    ipd = interpupillary(lm0)
    D = np.mean(np.linalg.norm(lm0 - lm1, axis=1)) / ipd
    f_lm = 0.85 * E - 0.15 * D

    hsv = cv2.cvtColor(img_mod, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1].astype(np.float32) / 255.0
    S_mean = float(np.mean(S))
    P_low  = max(0.0, 0.4 - S_mean) * 2.0
    P_high = max(0.0, S_mean - 0.8) * 1.5
    P_sat  = P_low + P_high

    # Un toque de nitidez, poco peso para no dominar
    f_edge = edge_nitidez_score(img_mod)
    return float(0.9 * f_lm - 0.2 * P_sat + 0.1 * f_edge)

# -------------------------- Operadores genéticos ------------------------------
def arithmetic_crossover(p1, p2):
    lam = random.random()
    return [lam * a + (1 - lam) * b for a, b in zip(p1, p2)]

def simple_mutate(ind, bounds, mutation_rate=0.1):
    out = ind.copy()
    for i, (lo, hi) in enumerate(bounds):
        if random.random() < mutation_rate:
            out[i] = float(np.clip(out[i] + (hi - lo) * random.uniform(-0.1, 0.1), lo, hi))
    return out

# -------------------------- GA: torneo + elitismo -----------------------------
def initialize_population(n, bounds):
    return [[random.uniform(lo, hi) for (lo, hi) in bounds] for _ in range(n)]

def tournament_selection(pop, fits, k=3):
    idxs = np.random.choice(len(pop), size=k, replace=False)
    best = idxs[0]
    best_fit = fits[best]
    for i in idxs[1:]:
        if fits[i] > best_fit:
            best, best_fit = i, fits[i]
    return pop[best]

def genetic_algorithm_fast(
    image_orig, landmarks_orig, bounds,
    pop_size=20, generations=60, mutation_rate=0.1,
    tournament_k=3, elitism_k=1, ga_max_side=256,
    patience=10, eps_improve=1e-4, n_workers=None, fast_eval=True
):
    t0 = time.time()
    work_img, _ = resize_max_side(image_orig, max_side=ga_max_side)
    scale_w = work_img.shape[1] / image_orig.shape[1]
    work_lm  = (landmarks_orig.astype(np.float32) * scale_w).astype(np.int32)

    pop = initialize_population(pop_size, bounds)
    best_ind, best_fit = None, -np.inf
    best_hist, div_hist = [], []
    no_improve = 0

    if n_workers is None:
        try:
            n_workers = min(32, (os.cpu_count() or 4) + 4)
        except:
            n_workers = 8

    def eval_ind(ind):
        img_mod, lm_mod = apply_effects_caricature(work_img, work_lm, ind, fast=fast_eval)
        if lm_mod is None:
            return -1.0
        return float(fitness_F3_normalizada_saturacion(work_lm, lm_mod, work_img.shape, img_mod))

    for gen in range(generations):
        # Eval paralela
        with cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
            fits = list(ex.map(eval_ind, pop))

        # Orden para elitismo
        order = np.argsort(fits)[::-1]
        elites_idx = order[:elitism_k]
        elites = [pop[i] for i in elites_idx]
        elite_fit = fits[elites_idx[0]]

        if elite_fit > best_fit + eps_improve:
            best_fit = elite_fit
            best_ind = elites[0]
            no_improve = 0
        else:
            no_improve += 1

        best_hist.append(best_fit)
        div_hist.append(float(np.mean(np.std(np.array(pop, dtype=np.float32), axis=0))))

        if no_improve >= patience:
            break

        # Nueva población
        new_pop = elites[:]
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fits, k=tournament_k)
            p2 = tournament_selection(pop, fits, k=tournament_k)
            child = arithmetic_crossover(p1, p2)
            child = simple_mutate(child, bounds, mutation_rate)
            for i, (lo, hi) in enumerate(bounds):
                if child[i] < lo: child[i] = lo
                if child[i] > hi: child[i] = hi
                child[i] = float(child[i])
            new_pop.append(child)
        pop = new_pop

    total_time = time.time() - t0

    # Convergencia
    conv_gen = len(best_hist)
    for i in range(conv_gen - 1, 0, -1):
        if abs(best_hist[i] - best_hist[i - 1]) > eps_improve:
            conv_gen = i + 1
            break

    return best_ind, best_fit, best_hist, total_time, float(np.mean(div_hist)), conv_gen

# ----------------------------- Experimentos ----------------------------------
def run_single_fast(image_orig, bounds, runs=5):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lm0 = get_landmarks(image_orig)
    if lm0 is None:
        raise ValueError("No se detectaron landmarks en la imagen original")

    tag = "OP1_Arith_SimpleMut_F3_Norm_Saturacion_FAST"
    print(f"\n>>>> Ejecutando: {tag}")
    all_runs = []

    for seed in range(runs):
        random.seed(seed); np.random.seed(seed)
        print(f"  > Seed {seed}")
        best_params, best_fit, hist, t, div, conv = genetic_algorithm_fast(
            image_orig, lm0, bounds,
            pop_size=POP_SIZE, generations=GENERATIONS, mutation_rate=MUTATION_RATE,
            tournament_k=TOURNAMENT_K, elitism_k=ELITISM_K, ga_max_side=GA_MAX_SIDE,
            patience=PATIENCE, eps_improve=EPS_IMPROVE, n_workers=None, fast_eval=FAST_EVAL
        )
        # Re-render final en ALTA (pipeline completo)
        best_img, _ = apply_effects_caricature(image_orig, lm0, best_params, fast=False)
        out_path = os.path.join(RESULTS_DIR, f"{tag}_seed{seed}_best.jpg")
        cv2.imwrite(out_path, best_img)

        all_runs.append({
            "seed": seed,
            "best_fitness": float(best_fit),
            "best_params": list(map(float, best_params)),
            "convergence_gen": int(conv),
            "total_time": float(t),
            "avg_diversity": float(div),
            "fitness_history": list(map(float, hist)),
            "best_image_path": out_path
        })

    with open(os.path.join(RESULTS_DIR, f"{tag}_results.json"), "w") as f:
        json.dump({tag: all_runs}, f, indent=2)
    print(f"Resultados guardados en {RESULTS_DIR}")

    return {tag: all_runs}

# ------------------------------ MAIN -----------------------------------------
if __name__ == "__main__":
    img = cv2.imread(IMAGE_FILE)
    if img is None:
        print(f"ERROR: no se pudo cargar '{IMAGE_FILE}'")
        raise SystemExit(1)

    img = resize_keep_ratio(img, 400, 400)

    # Límites de genes
    param_bounds = [
        (1.00, 1.50),  # ojos
        (0.80, 1.30),  # boca
        (0.80, 1.20),  # mentón
        (0.90, 1.20),  # cabeza
        (0.80, 1.50),  # saturación
    ]

    if RUN_GRID:
        # Si quieres volver al grid 3x3, aquí podrías iterar sobre sets,
        # pero para máxima velocidad nos quedamos con la combinación rápida.
        pass

    results = run_single_fast(img, param_bounds, runs=NUM_RUNS)
    print("\nFin.")
