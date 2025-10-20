import cv2
import numpy as np
import random
import mediapipe as mp

# ---------------------------
# MediaPipe
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# ---------------------------
# Utiles
# ---------------------------
def _odd_kernel_from_sigma(sigma):
    k = int(np.ceil(6.0 * float(sigma)))
    if k < 3: k = 3
    if k % 2 == 0: k += 1
    return k

def _clip01(x): return float(np.clip(x, 0.0, 1.0))

# ---------------------------
# Landmarks
# ---------------------------
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
    if landmarks is not None:
        for (x, y) in landmarks:
            cv2.circle(img_copy, (x, y), 1, (0, 255, 0), -1)
    return img_copy

# ---------------------------
# Máscaras (cara y piel)
# ---------------------------
def _face_mask_from_landmarks(img_shape, lms):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    if lms is None or len(lms) < 3:
        return mask
    hull = cv2.convexHull(lms.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask

def _skin_mask_from_face(image, face_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 140)  # umbrales algo bajos
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    skin = cv2.bitwise_and(face_mask, cv2.bitwise_not(edges))
    skin = cv2.medianBlur(skin, 5)  # bordes suaves
    return skin

# ---------------------------
# Warp afín global (aprox)
# ---------------------------
def warp_image(src, src_points, dst_points, size):
    src_points = np.float32(src_points)
    dst_points = np.float32(dst_points)
    matrix, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
    warped = cv2.warpAffine(src, matrix, (size[1], size[0]), flags=cv2.INTER_LINEAR)
    return warped

# ---------------------------
# Efectos
# params = [s_suavizado(0..1), lift_mejilla_px(0..4), lift_parpado_px(0..3), vol_labios(0.95..1.15)]
# ---------------------------
def apply_effects(image, params):
    s, lift_m_px, lift_p_px, vol_l = params
    img = image.copy()
    h, w = img.shape[:2]

    landmarks = get_landmarks(img)
    if landmarks is None:
        return img

    modified_landmarks = landmarks.copy().astype(np.float32)

    # --- Warp sutil: mejillas y párpados (en píxeles) ---
    cheek_idx = list(range(50, 55)) + list(range(280, 286))
    eyelid_idx = [159, 160, 161, 386, 387, 388]

    modified_landmarks[cheek_idx, 1] -= float(lift_m_px)  # subir
    modified_landmarks[eyelid_idx, 1] -= float(lift_p_px)

    # --- Volumen labios alrededor de 1.0 ---
    lips_idx = list(range(61, 81)) + list(range(291, 309))
    center_lips = np.mean(landmarks[lips_idx], axis=0).astype(np.float32)
    lip_gain = (float(vol_l) - 1.0) * 0.12  # 12% máx
    for i in lips_idx:
        direction = landmarks[i].astype(np.float32) - center_lips
        modified_landmarks[i] = landmarks[i].astype(np.float32) + direction * lip_gain

    # --- Warp global afín (rápido). Para máxima calidad usarías warp piecewise (Delaunay), pero esto sirve. ---
    warped = warp_image(img, landmarks, modified_landmarks, img.shape)

    # --- Suavizado selectivo de piel (frequency separation simple) ---
    face_mask = _face_mask_from_landmarks(img.shape, landmarks)
    skin_mask = _skin_mask_from_face(img, face_mask)

    # base suavizada
    s = _clip01(s)
    sigma = 8 + 22 * s                 # sigma 8..30
    k = _odd_kernel_from_sigma(sigma/3) # kernel acorde
    blurred = cv2.GaussianBlur(warped, (k, k), sigmaX=sigma, sigmaY=sigma)

    # detalle (altas frecuencias) de la imagen warp
    detail = cv2.subtract(warped, cv2.GaussianBlur(warped, (k, k), sigmaX=sigma/2, sigmaY=sigma/2))
    # Mezcla: en piel, usa más blurred; fuera de piel, conserva warp + detalle
    skin_f = (skin_mask > 0).astype(np.float32)[..., None]
    # mezcla:  skin -> 0.7*blur + 0.3*(warp+detalle*0.5) ; no-skin -> warp+detalle
    sharp = cv2.add(warped, (detail * 0.5).astype(np.int16), dtype=cv2.CV_8U)
    skin_area = cv2.addWeighted(blurred, 0.7, sharp, 0.3, 0)
    final_img = (skin_f * skin_area + (1 - skin_f) * sharp).astype(np.uint8)

    return final_img

# ---------------------------
# FITNESS v6 (más “amable”)
# ---------------------------
def compute_fitness(image_orig, image_mod, landmarks_orig, landmarks_mod=None,
                    alpha=0.6, beta=0.25, gamma=0.15):
    """
    fitness = α*Smooth + β*EdgeCorr + γ*ShapeStability
    - Smooth: 1 - (E_after/E_before)^(1/4) con LoG multi-escala en piel.
    - EdgeCorr: correlación de gradiente (Sobel) en rasgos con boost (expo 0.75).
    - ShapeStability: Procrustes suave (τ=0.10) con boost (expo 0.8).
    """
    eps = 1e-8

    def log_energy(gray, mask, sigmas=(1.0, 2.0, 3.0, 4.0)):
        m = (mask > 0)
        if m.sum() < 10:
            return 0.0
        E = 0.0
        for s in sigmas:
            k = _odd_kernel_from_sigma(s)
            blur = cv2.GaussianBlur(gray, (k, k), s)
            lap = cv2.Laplacian(blur, cv2.CV_32F)
            E += float((lap[m] ** 2).mean())
        return E / len(sigmas)

    def gradient_corr(img_o, img_m, face_mask):
        go = cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY)
        gm = cv2.cvtColor(img_m, cv2.COLOR_BGR2GRAY)
        def sobel_mag(g):
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            return np.sqrt(gx*gx + gy*gy)
        G1 = sobel_mag(go); G2 = sobel_mag(gm)
        edges = cv2.Canny(go, 70, 170)
        feat = cv2.bitwise_and(edges, face_mask)
        feat = cv2.dilate(feat, np.ones((3,3), np.uint8), 1)
        m = (feat > 0)
        if m.sum() < 50:
            m = (face_mask > 0)
        x = G1[m].astype(np.float32); y = G2[m].astype(np.float32)
        if x.size < 10: return 0.0
        x = (x - x.mean()) / (x.std() + eps)
        y = (y - y.mean()) / (y.std() + eps)
        corr = float(np.clip((x*y).mean(), -1.0, 1.0))
        corr01 = 0.5*(corr + 1.0)
        return float(corr01 ** 0.75)

    def procrustes_align(P, Q):
        P = P.astype(np.float32); Q = Q.astype(np.float32)
        Pm, Qm = P.mean(0, keepdims=True), Q.mean(0, keepdims=True)
        P0, Q0 = P - Pm, Q - Qm
        nP = np.sqrt((P0**2).sum()) + eps
        nQ = np.sqrt((Q0**2).sum()) + eps
        P0, Q0 = P0/nP, Q0/nQ
        U, _, VT = np.linalg.svd(Q0.T @ P0)
        R = U @ VT
        Q_al = (Q0 @ R) * nP + Pm
        return Q_al

    def shape_stability(P, Q, idx_io=(33,263), tau=0.10):
        if Q is None:
            return 1.0
        Q_al = procrustes_align(P, Q)
        d_mean = float(np.linalg.norm(P - Q_al, axis=1).mean())
        io = float(np.linalg.norm(P[idx_io[0]] - P[idx_io[1]])) + eps
        DR = float(np.clip(d_mean / io, 0.0, 1.0))
        base = float(np.exp(- (DR / tau) ** 2))
        return float(base ** 0.8)

    face_mask = _face_mask_from_landmarks(image_orig.shape, landmarks_orig)
    skin_mask = _skin_mask_from_face(image_orig, face_mask)
    gray_o = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
    gray_m = cv2.cvtColor(image_mod,  cv2.COLOR_BGR2GRAY)

    E_b = log_energy(gray_o, skin_mask)
    E_a = log_energy(gray_m, skin_mask)
    ratio = float(np.clip(E_a / (E_b + eps), 0.0, 1.0))
    smooth = 1.0 - (ratio ** 0.25)

    edgecorr = gradient_corr(image_orig, image_mod, face_mask)
    shapek = shape_stability(landmarks_orig, landmarks_mod)

    s = alpha + beta + gamma
    if s <= 0: alpha, beta, gamma, s = 0.6, 0.25, 0.15, 1.0
    alpha, beta, gamma = alpha/s, beta/s, gamma/s

    fitness = alpha*smooth + beta*edgecorr + gamma*shapek
    return float(np.clip(fitness, 0.0, 1.0))

# ---------------------------
# GA: init / selection / crossover / mutation
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

def selection_roulette(population, fitnesses, eps=1e-8):
    f = np.array(fitnesses, dtype=np.float64)
    f = f - f.min() + eps
    p = f / f.sum()
    idx = np.random.choice(len(population), p=p)
    return population[idx]

def selection_rank(population, fitnesses):
    order = np.argsort(fitnesses)
    ranks = np.empty(len(fitnesses), dtype=np.float64)
    ranks[order] = np.arange(1, len(fitnesses) + 1)
    p = ranks / ranks.sum()
    idx = np.random.choice(len(population), p=p)
    return population[idx]

def arithmetic_crossover(parent1, parent2):
    lam = random.uniform(0.25, 0.75)
    return [lam * p1 + (1 - lam) * p2 for p1, p2 in zip(parent1, parent2)]

def crossover_blx_alpha(p1, p2, alpha=0.25):
    child = []
    for a, b in zip(p1, p2):
        lo, hi = min(a, b), max(a, b)
        span = hi - lo
        L = lo - alpha * span
        U = hi + alpha * span
        child.append(random.uniform(L, U))
    return child

def crossover_sbx(p1, p2, eta=20):
    child = []
    for a, b in zip(p1, p2):
        u = random.random()
        if u <= 0.5:
            beta = (2 * u) ** (1 / (eta + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        c = 0.5 * ((1 + beta) * a + (1 - beta) * b)
        child.append(c)
    return child

def mutate(individual, param_bounds, mutation_rate=0.12, scale=0.07):
    new_indiv = individual.copy()
    for i, (low, high) in enumerate(param_bounds):
        if random.random() < mutation_rate:
            sigma = (high - low) * scale
            new_indiv[i] = float(np.clip(new_indiv[i] + random.gauss(0, sigma), low, high))
    return new_indiv

# ---------------------------
# GA principal (ELITISMO=2)
# ---------------------------
def genetic_algorithm(image_orig, param_bounds,
                      pop_size=20, generations=50,
                      selection_mode="rank", crossover_mode="blx",
                      alpha=0.6, beta=0.25, gamma=0.15,
                      mutation_rate=0.12, mutation_scale=0.07,
                      elitism=2):
    landmarks_orig = get_landmarks(image_orig)
    if landmarks_orig is None:
        raise ValueError("No se detectaron landmarks en la imagen original")

    sel_map = {
        "tournament": lambda P, F: tournament_selection(P, F, k=3),
        "roulette":   selection_roulette,
        "rank":       selection_rank
    }
    cx_map = {
        "arithmetic": arithmetic_crossover,
        "blx":        lambda a, b: crossover_blx_alpha(a, b, alpha=0.25),
        "sbx":        lambda a, b: crossover_sbx(a, b, eta=20)
    }
    select_fn = sel_map.get(selection_mode, selection_rank)
    cross_fn  = cx_map.get(crossover_mode, crossover_blx_alpha)

    population = initialize_population(pop_size, param_bounds)

    best_individual = None
    best_fitness = -np.inf
    best_image = None

    for gen in range(generations):
        fitnesses, modified_images = [], []

        for indiv in population:
            img_mod = apply_effects(image_orig, indiv)
            lms_mod = get_landmarks(img_mod)
            fit = compute_fitness(image_orig, img_mod, landmarks_orig, lms_mod,
                                  alpha=alpha, beta=beta, gamma=gamma) if lms_mod is not None else 0.0
            fitnesses.append(fit)
            modified_images.append(img_mod)

        idx = int(np.argmax(fitnesses))
        if fitnesses[idx] > best_fitness:
            best_fitness = float(fitnesses[idx])
            best_individual = population[idx][:]
            best_image = modified_images[idx].copy()

        # Nueva población con ELITISMO
        new_population = []
        if elitism > 0 and best_individual is not None:
            for _ in range(elitism):
                new_population.append(best_individual[:])

        while len(new_population) < pop_size:
            p1 = select_fn(population, fitnesses)
            p2 = select_fn(population, fitnesses)
            child = cross_fn(p1, p2)
            child = mutate(child, param_bounds, mutation_rate=mutation_rate, scale=mutation_scale)
            child = [float(np.clip(v, lo, hi)) for v, (lo, hi) in zip(child, param_bounds)]
            new_population.append(child)

        population = new_population
        print(f"Generación {gen + 1:02d} | Mejor fitness = {best_fitness:.4f} | sel={selection_mode} | cx={crossover_mode}")

    return best_individual, best_fitness, best_image

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    image_orig = cv2.imread("rostro.jpg")

    # NUEVOS límites útiles (en píxeles/escala):
    # s: 0..1  | lift mejilla: 0..4 px | lift párpado: 0..3 px | volumen labios: 0.95..1.15
    param_bounds = [(0.0, 1.0), (0.0, 4.0), (0.0, 3.0), (0.95, 1.15)]

    best_params, best_fit, best_img = genetic_algorithm(
        image_orig,
        param_bounds,
        pop_size=22,
        generations=50,
        selection_mode="rank",
        crossover_mode="blx",
        alpha=0.6, beta=0.25, gamma=0.15,
        mutation_rate=0.12,
        mutation_scale=0.07,
        elitism=2
    )

    print("\nMejores parámetros:", best_params)
    print("Mejor fitness:", best_fit)

    landmarks_orig = get_landmarks(image_orig)
    image_with_landmarks = draw_landmarks_points(image_orig, landmarks_orig)

    h = image_orig.shape[0]
    imgs_to_show = [image_orig, image_with_landmarks, best_img]
    resized_imgs = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in imgs_to_show]
    combined = np.hstack(resized_imgs)

    cv2.imshow("Original | Con Landmarks | Rejuvenecido", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
