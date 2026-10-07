import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, List, Union

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from scipy.signal import hilbert, find_peaks, peak_widths
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image


# ==============================================================================
# PART 0 — CHARGEMENT DU B-SCAN ET EXTRACTION DES A-SCANS
# ==============================================================================

BSCAN_PATH = r"C:\Users\Alix\Desktop\STG2\documents for datasets\B-scans\B-scans\data_1.png"

BSCAN_CROP_ROWS: Optional[Sequence[int]] = None
BSCAN_CROP_COLS: Optional[Sequence[int]] = None

# False : pixel [0, 255] ramené à [0, 1] (courbe de brillance, toujours positive).
# True  : gris moyen (~127.5) = amplitude nulle, pixel remis à l'échelle [-1, 1]
BSCAN_ASSUME_DIVERGING_GREY = True
BSCAN_TRANSPOSE = False

# Largeur (en lignes du B-scan) des deux paquets utilisés pour la vérification
# sur le B-scan brut (voir PART H). Mettre la même valeur pour une largeur fixe.
MIN_PACKET_ROWS = 5
MAX_PACKET_ROWS = 10


def load_bscan_grayscale(path: Union[str, Path]) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))  # "L" = niveaux de gris 8 bits


def crop_bscan(bscan_gray: np.ndarray, row_range: Optional[Sequence[int]] = None,
               col_range: Optional[Sequence[int]] = None) -> np.ndarray:
    r0, r1 = row_range if row_range is not None else (0, bscan_gray.shape[0])
    c0, c1 = col_range if col_range is not None else (0, bscan_gray.shape[1])
    return bscan_gray[r0:r1, c0:c1]


def extract_columns(bscan_gray: np.ndarray, normalize: bool = True,
                    assume_diverging_grey: bool = False) -> np.ndarray:
    bscan_float = bscan_gray.astype(np.float64)

    if assume_diverging_grey:
        signal_matrix = (bscan_float - 127.5) / 127.5   # [0, 255] -> [-1, 1]
    elif normalize:
        signal_matrix = bscan_float / 255.0
    else:
        signal_matrix = bscan_float

    # Transposée : chaque LIGNE de la sortie = une COLONNE (A-scan) de l'image.
    return signal_matrix.T


def load_bscan(path: str, transpose: bool = False, assume_diverging_grey: bool = False,
               crop_rows: Optional[Sequence[int]] = None,
               crop_cols: Optional[Sequence[int]] = None) -> np.ndarray:
    path_lower = path.lower()

    if path_lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        bscan_gray = load_bscan_grayscale(path)
        if crop_rows is not None or crop_cols is not None:
            bscan_gray = crop_bscan(bscan_gray, row_range=crop_rows, col_range=crop_cols)
        bscan = extract_columns(bscan_gray, normalize=True,
                                assume_diverging_grey=assume_diverging_grey)

    elif path_lower.endswith(".npy"):
        bscan = np.load(path)

    elif path_lower.endswith(".npz"):
        data = np.load(path)
        bscan = data["bscan"] if "bscan" in data else data[list(data.keys())[0]]

    elif path_lower.endswith(".csv"):
        bscan = np.loadtxt(path, delimiter=",")

    elif path_lower.endswith(".txt"):
        bscan = np.loadtxt(path)

    else:
        raise ValueError(
            f"Format de fichier non supporté pour '{path}'. "
            f"Utilisez une image (.png, .jpg, .bmp, .tif), ou .npy, .npz, .csv, .txt."
        )

    bscan = np.asarray(bscan, dtype=np.float32)

    if bscan.ndim != 2:
        raise ValueError(
            f"Le B-scan chargé doit être une matrice 2D (A-scans x temps), "
            f"mais a la forme {bscan.shape}."
        )

    return bscan.T if transpose else bscan


# ==============================================================================
# PART A — DENOISING
# ==============================================================================

def normalize_signals(X: np.ndarray):
    X = np.asarray(X, dtype=np.float32)

    offsets = np.median(X, axis=1, keepdims=True)
    X_centered = X - offsets

    scales = np.max(np.abs(X_centered), axis=1, keepdims=True)
    scales[scales < 1e-12] = 1.0

    return X_centered / scales, offsets, scales


def denormalize_signals(X_norm: np.ndarray, offsets: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return X_norm * scales + offsets


def cluster_signals(X_norm: np.ndarray, n_clusters: int, random_state: int = 2):
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    return km.fit_predict(X_norm), km


class DenoisingAutoencoder(nn.Module):

    def __init__(self, n_input, n_hidden=20):
        super().__init__()
        self.encoder = nn.Linear(n_input, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_input)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.decoder(self.act(self.encoder(x)))


class EMA:

    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply_to(self, model):
        model.load_state_dict(self.shadow)


def regularized_loss(y_pred, y_true, model, gamma=0.9):
    mse = torch.mean((y_pred - y_true) ** 2)

    params = list(model.parameters())
    l2 = sum(torch.sum(p ** 2) for p in params) / sum(p.numel() for p in params)

    return gamma * mse + (1 - gamma) * l2


def train_denoising_autoencoder(X_cluster, n_hidden=20, n_iterations=2000, batch_size=5,
                                lr0=0.001, gamma=0.9, ema_decay=0.99, device="cpu"):
    n_i, L = X_cluster.shape

    X_t = torch.tensor(X_cluster, dtype=torch.float32, device=device)

    model = DenoisingAutoencoder(n_input=L, n_hidden=n_hidden).to(device)
    ema = EMA(model, decay=ema_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    for _ in range(n_iterations):
        idx = np.random.choice(n_i, size=min(batch_size, n_i), replace=(n_i < batch_size))
        batch = X_t[idx]

        optimizer.zero_grad()
        loss = regularized_loss(model(batch), batch, model, gamma=gamma)
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update(model)

    ema.apply_to(model)
    model.eval()

    return model


def denoise_full_dataset(X_raw: np.ndarray, n_clusters: int, n_hidden: int = 20,
                         n_iterations: int = 2000, batch_size: int = 5,
                         device: str = "cpu", verbose: bool = True):

    X_norm, offsets, scales = normalize_signals(X_raw)
    labels, _ = cluster_signals(X_norm, n_clusters)

    X_denoised_norm = np.zeros_like(X_norm)
    models = {}

    for c in range(n_clusters):

        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue

        if verbose:
            print(f"[Cluster {c}] {len(idx)} A-scans -> entraînement de l'autoencodeur")

        X_c = X_norm[idx]
        model = train_denoising_autoencoder(X_c, n_hidden=n_hidden, n_iterations=n_iterations,
                                            batch_size=batch_size, device=device)
        models[c] = model

        with torch.no_grad():
            X_denoised_norm[idx] = model(
                torch.tensor(X_c, dtype=torch.float32, device=device)
            ).cpu().numpy()

    return denormalize_signals(X_denoised_norm, offsets, scales), labels, models


# ==============================================================================
# PART B — LOCALISATION SURFACE / BACKWALL
# ==============================================================================

def estimate_wave_bands(X_denoised: np.ndarray, threshold_ratio: float = 0.2,
                        smooth_window: int = 5) -> Tuple[Optional[Tuple[int, int]],
                                                         Optional[Tuple[int, int]],
                                                         Tuple[int, int]]:
    baseline = np.median(X_denoised)
    depth_energy = np.mean(np.abs(X_denoised - baseline), axis=0)

    kernel = np.ones(smooth_window) / smooth_window
    depth_energy_smooth = np.convolve(depth_energy, kernel, mode="same")

    L = len(depth_energy_smooth)
    above = depth_energy_smooth > threshold_ratio * depth_energy_smooth.max()

    labeled, n_runs = ndimage.label(above)

    if n_runs == 0:
        return None, None, (0, L - 1)

    runs = []
    for i in range(1, n_runs + 1):
        idx = np.where(labeled == i)[0]
        runs.append((int(idx[0]), int(idx[-1])))

    runs.sort(key=lambda r: r[0])

    surface_band = runs[0]
    backwall_band = runs[-1]

    return surface_band, backwall_band, (surface_band[1] + 1, backwall_band[0] - 1)


def most_energetic_ascan(X: np.ndarray, depth_window: Tuple[int, int],
                          top_k: int = 5) -> List[int]:
    """
    Indices des `top_k` A-scans les plus énergétiques dans la fenêtre de
    diffraction (utilisé par select_best_ascan_for_tips pour choisir les
    A-scans candidats du pipeline).
    """
    d0, d1 = depth_window
    band = X[:, d0:d1]

    energy = np.mean(np.abs(band - np.median(band)), axis=1)

    return np.argsort(energy)[::-1][:top_k].tolist()


def select_best_ascan_for_tips(X_signal: np.ndarray, axis_values: np.ndarray,
                               depth_window: Tuple[int, int],
                               sep_config: "SeparationConfig",
                               min_peak_prominence: float = 0.1,
                               min_peak_distance_samples: int = 3,
                               top_k: int = 15) -> int:
    """
    Sélectionne le A-scan qui montre le MIEUX les deux échos de diffraction.

    1. On ne garde que les `top_k` A-scans les plus énergétiques (candidats).
    2. Parmi eux, on redétecte les pics (_detect_two_peaks_in_window) et on
       classe : a) 2 pics fiables (proéminence > 0), b) 2 pics dont au moins un
       vient de la séparation de secours par bande, c) un seul pic ; puis, à
       égalité, le score de proéminence le plus élevé.
    """
    d0, d1 = depth_window
    candidate_indices = most_energetic_ascan(X_signal, depth_window, top_k=top_k)
    mask = (axis_values >= d0) & (axis_values <= d1)

    best_index = None
    best_key = None  # (n_reliable_peaks, n_peaks_total, score)

    for i in candidate_indices:

        windowed_envelope = extract_envelope(X_signal[i])[mask]

        if len(windowed_envelope) == 0:
            continue

        peaks = _detect_two_peaks_in_window(
            windowed_envelope,
            sep_config=sep_config,
            min_peak_prominence=min_peak_prominence,
            min_distance_samples=min_peak_distance_samples,
            expected_n_peaks=2,
        )

        if not peaks:
            continue

        proms = [p for _, p in peaks]

        n_reliable = sum(1 for p in proms if p > 0)
        score = float(sum(abs(p) for p in proms[:2]))
        key = (min(n_reliable, 2), min(len(proms), 2), score)

        if best_key is None or key > best_key:
            best_index = int(i)
            best_key = key

    # Dernier recours : aucun pic détectable, on renvoie le plus énergétique.
    return best_index if best_index is not None else int(candidate_indices[0])


# ==============================================================================
# PART C — DÉTECTION DES TIPS
# ==============================================================================

@dataclass
class SeparationConfig:
    """
    Regroupe les 4 paramètres qui contrôlent la séparation de deux tips proches
    (lobe potentiellement fusionné). Auparavant traînés séparément à travers
    6 fonctions différentes ; centralisés ici pour éviter d'oublier un
    paramètre en cascade quand on ajuste les seuils.

    - max_separation_factor : borne le voisinage de recherche d'un 2e pic à ce
      facteur fois la largeur à mi-hauteur du pic principal.
    - min_relative_second_peak_amplitude : amplitude minimale du second sommet
      relative au premier (écarte les résidus de bruit).
    - band_ratio : tolérance de la bande d'amplitude utilisée pour délimiter la
      "zone" du pic principal (voir _split_via_band_exit).
    - max_absolute_separation : plafond ABSOLU (échantillons) sur la distance de
      recherche d'un second tip, indépendant de la largeur du pic. None = pas
      de plafond.
    """
    max_separation_factor: float = 1.0
    min_relative_second_peak_amplitude: float = 0.3
    band_ratio: float = 0.15
    max_absolute_separation: Optional[int] = 50


def extract_envelope(signal: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(signal))


def _window(envelope: np.ndarray, axis_values: np.ndarray,
            start: float, end: float) -> Tuple[np.ndarray, np.ndarray]:
    """Restreint (axe, enveloppe) à la fenêtre [start, end]."""
    mask = (axis_values >= start) & (axis_values <= end)
    return axis_values[mask], envelope[mask]


def _half_max_width(envelope: np.ndarray, peak_idx: int) -> float:
    """
    Largeur à mi-hauteur (FWHM) du pic à `peak_idx` dans `envelope`.
    Factorisé : le même calcul (peak_widths, rel_height=0.5) était dupliqué
    dans _split_via_band_exit et _detect_two_peaks_in_window.
    """
    widths, _, _, _ = peak_widths(envelope, [peak_idx], rel_height=0.5)
    return float(widths[0])


def _is_plausible_peak(candidate_amplitude: float, reference_amplitude: float,
                       distance_samples: float, max_distance_samples: float,
                       min_relative_amplitude: float) -> bool:
    """
    Un pic candidat est plausible s'il est assez proche du pic de référence
    (<= max_distance_samples) ET assez haut par rapport à lui
    (>= min_relative_amplitude * reference_amplitude).

    Centralise un critère qui était réimplémenté indépendamment à 3 endroits
    (_detect_two_peaks_in_window, _split_via_band_exit, raw_consensus_tips),
    avec le risque que les seuils divergent si on n'en modifiait qu'un.
    """
    return (distance_samples <= max_distance_samples
            and candidate_amplitude >= min_relative_amplitude * reference_amplitude)


# Principe de la séparation d'un lobe fusionné (v4, retenue) :
#   1. A = amplitude du sommet détecté ; bande de tolérance [A*(1-band_ratio), A].
#   2. On marche vers la gauche puis la droite jusqu'au premier point où la
#      courbe SORT de la bande (passe sous le seuil bas).
#   3. Dans un voisinage borné (max_separation_factor, max_absolute_separation),
#      on cherche un second VRAI maximum local, strictement hors de la zone du
#      premier pic, d'amplitude suffisante -> second tip.
#   4. Sinon on ne fabrique rien : un seul tip est renvoyé.
# (Les approches précédentes -- dérivée seconde, bande vers le bas -- créaient
# de faux doublons sur un pic isolé et ont été abandonnées.)

def _split_via_band_exit(
    windowed_envelope: np.ndarray,
    peak_idx: int,
    sep_config: SeparationConfig,
    min_distance_samples: int = 3,
) -> Optional[List[Tuple[int, float]]]:
    """
    Cherche un second tip en dehors de la "zone" du sommet `peak_idx`.

    Renvoie [(index, proéminence_négative), ...] triée par position, ou None si
    aucun second sommet plausible n'existe (un seul tip : rien de fictif).
    """
    L = len(windowed_envelope)

    A = float(windowed_envelope[peak_idx])
    if A <= 0:
        return None

    lower = A * (1.0 - sep_config.band_ratio)

    # Sortie de bande à gauche / à droite du sommet.
    exit_left = 0
    for i in range(peak_idx - 1, -1, -1):
        if windowed_envelope[i] < lower:
            exit_left = i
            break

    exit_right = L - 1
    for i in range(peak_idx + 1, L):
        if windowed_envelope[i] < lower:
            exit_right = i
            break

    # Voisinage borné de recherche.
    w = max(2, int(np.ceil(_half_max_width(windowed_envelope, peak_idx))))
    radius = max(min_distance_samples, int(sep_config.max_separation_factor * w))
    if sep_config.max_absolute_separation is not None:
        radius = min(radius, int(sep_config.max_absolute_separation))

    lo = max(0, peak_idx - radius)
    hi = min(L, peak_idx + radius + 1)

    candidates: List[Tuple[int, float]] = []

    # Segments à gauche puis à droite de la zone du pic principal.
    for start, stop, usable in ((lo, exit_left + 1, exit_left > lo),
                                (exit_right, hi, exit_right < hi - 1)):
        if not usable:
            continue
        seg = windowed_envelope[start:stop]
        if len(seg) >= 3:
            idxs, _ = find_peaks(seg, distance=max(1, min_distance_samples))
            candidates += [(start + int(i), float(seg[i])) for i in idxs]

    plausible = [
        c for c in candidates
        if _is_plausible_peak(c[1], A, abs(c[0] - peak_idx), radius,
                              sep_config.min_relative_second_peak_amplitude)
        and abs(c[0] - peak_idx) >= min_distance_samples
    ]

    if not plausible:
        return None

    second_idx, second_amp = max(plausible, key=lambda c: c[1])
    pseudo_prom = -0.5 * min(A, second_amp)

    return sorted([(peak_idx, pseudo_prom), (second_idx, pseudo_prom)], key=lambda p: p[0])


def _detect_two_peaks_in_window(
    windowed_envelope: np.ndarray,
    sep_config: SeparationConfig,
    min_peak_prominence: float = 0.1,
    min_distance_samples: int = 3,
    expected_n_peaks: int = 2,
) -> List[Tuple[int, float]]:
    """
    Détecte jusqu'à `expected_n_peaks` pics DISTINCTS dans une enveloppe déjà
    restreinte à la fenêtre de recherche.

    Quand les deux tips sont proches, leurs enveloppes fusionnent en UNE bosse
    et find_peaks seul n'en voit qu'un : on tente alors la séparation par
    sortie de bande (_split_via_band_exit).

    Renvoie [(index_dans_la_fenetre, proéminence)] triée par position, de
    longueur <= expected_n_peaks. Proéminence négative = détection de secours
    par bande d'amplitude (confiance plus faible).
    """
    if len(windowed_envelope) == 0:
        return []

    max_env = np.max(windowed_envelope)
    if max_env <= 0:
        return []

    peak_indices, properties = find_peaks(
        windowed_envelope, prominence=min_peak_prominence * max_env,
        distance=max(1, min_distance_samples),
    )
    found_prom = properties.get("prominences", np.array([]))

    if len(peak_indices) >= expected_n_peaks:
        # Cas 1 : plusieurs pics "réels" -- on vérifie leur PLAUSIBILITÉ
        # (distance + amplitude relative au pic principal) avant de les accepter.
        order = np.argsort(found_prom)[::-1]
        idx_sorted = peak_indices[order]
        prom_sorted = found_prom[order]

        ref_idx = int(idx_sorted[0])
        ref_prom = float(prom_sorted[0])
        top_amp = float(windowed_envelope[ref_idx])

        allowed_distance = sep_config.max_separation_factor * max(
            2.0, _half_max_width(windowed_envelope, ref_idx))
        if sep_config.max_absolute_separation is not None:
            allowed_distance = min(allowed_distance, float(sep_config.max_absolute_separation))

        plausible = [
            c for c in zip(idx_sorted[1:].tolist(), prom_sorted[1:].tolist())
            if _is_plausible_peak(windowed_envelope[c[0]], top_amp,
                                  abs(c[0] - ref_idx), allowed_distance,
                                  sep_config.min_relative_second_peak_amplitude)
        ]

        if plausible:
            second_idx, second_prom = max(plausible, key=lambda c: c[1])
            pairs = sorted([(ref_idx, ref_prom), (int(second_idx), float(second_prom))],
                           key=lambda p: p[0])
            return pairs[:expected_n_peaks]

        # Aucun candidat plausible : le "2e pic" était du bruit / un autre écho.
        # On traite le cas comme un lobe fusionné (ci-dessous).

    elif len(peak_indices) == 1 and expected_n_peaks == 2:
        # Cas 2 : un seul pic "réel" -> lobe fusionné probable.
        ref_idx = int(peak_indices[0])
        ref_prom = float(found_prom[0]) if len(found_prom) else 0.0

    else:
        return sorted(zip(peak_indices.tolist(), found_prom.tolist()), key=lambda p: p[0])

    band_pair = _split_via_band_exit(
        windowed_envelope, ref_idx, sep_config=sep_config,
        min_distance_samples=min_distance_samples,
    )
    if band_pair is not None:
        return band_pair

    # Aucun second sommet plausible : les tips sont CONFONDUS, un seul pic est
    # renvoyé (pas de relâchement du seuil de proéminence : il captait du bruit
    # lointain).
    return [(ref_idx, ref_prom)]


# ==============================================================================
# PART D — RELATIONS GÉOMÉTRIQUES
# ==============================================================================

def compute_depth(t_us: float, c_mm_per_us: float, d_mm: float) -> float:
    value = (t_us * c_mm_per_us / 2.0) ** 2 - d_mm ** 2
    return np.sqrt(max(value, 0.0))


def compute_height(t1_us: float, t2_us: float, c_mm_per_us: float, d_mm: float) -> float:
    return compute_depth(t2_us, c_mm_per_us, d_mm) - compute_depth(t1_us, c_mm_per_us, d_mm)


# ==============================================================================
# PART E — CONTENEUR DE RÉSULTAT
# ==============================================================================

@dataclass
class TipDetectionResult:

    ascan_index: int
    raw_signal: np.ndarray
    denoised_signal: np.ndarray
    envelope: np.ndarray
    axis_values: np.ndarray

    surface_band: Optional[Tuple[int, int]] = None
    backwall_band: Optional[Tuple[int, int]] = None
    diffraction_window: Optional[Tuple[int, int]] = None

    top_tip: Optional[float] = None
    bottom_tip: Optional[float] = None
    height_mm: Optional[float] = None

    # True : deux vrais pics séparés (proéminence positive) ; False : séparés via
    # la détection de secours par bande (confiance plus faible) ; None : un seul
    # tip (ou aucun) trouvé.
    tips_reliable: Optional[bool] = None

    # True : un seul écho distinct alors que deux étaient attendus -> tips
    # CONFONDUS dans l'enveloppe à cette position (résolution insuffisante, pas
    # forcément absence d'un second tip). False : deux tips trouvés ; None :
    # aucun pic (pas de défaut visible).
    tips_merged: Optional[bool] = None


# ==============================================================================
# PART F — PIPELINE COMPLET
# ==============================================================================

def extract_tips(X_raw: np.ndarray, n_clusters: int,
                 sep_config: Optional[SeparationConfig] = None,
                 ascan_index: Optional[int] = None,
                 depth_window: Optional[Tuple[float, float]] = None,
                 axis_values: Optional[np.ndarray] = None,
                 min_peak_prominence: float = 0.1, n_hidden: int = 20,
                 n_iterations: int = 2000, batch_size: int = 5,
                 c_mm_per_us: Optional[float] = None, d_mm: Optional[float] = None,
                 device: str = "cpu", verbose: bool = True
                 ) -> Tuple[TipDetectionResult, np.ndarray]:
    """
    Pipeline complet : débruitage -> localisation -> détection des tips.
    `sep_config` contrôle la séparation d'un lobe fusionné (voir
    `_detect_two_peaks_in_window` / `_split_via_band_exit`) ; par défaut on
    prend les valeurs par défaut de `SeparationConfig`.
    """
    if sep_config is None:
        sep_config = SeparationConfig()

    if axis_values is None:
        axis_values = np.arange(X_raw.shape[1])

    X_denoised, _, _ = denoise_full_dataset(
        X_raw, n_clusters=n_clusters, n_hidden=n_hidden, n_iterations=n_iterations,
        batch_size=batch_size, device=device, verbose=verbose,
    )

    surface_band, backwall_band, auto_window = estimate_wave_bands(X_denoised)

    if depth_window is None:
        depth_window = auto_window

    if verbose:
        print(f"Surface wave band  : {surface_band}")
        print(f"Backwall wave band : {backwall_band}")
        print(f"Diffraction window : {depth_window}")

    if ascan_index is None:
        # A-scan qui montre le MIEUX les deux tips (voir select_best_ascan_for_tips).
        ascan_index = select_best_ascan_for_tips(
            X_denoised, axis_values, depth_window, sep_config=sep_config,
            min_peak_prominence=min_peak_prominence,
        )

        if verbose:
            print(f"Auto-selected ascan_index = {ascan_index} "
                  f"(meilleur A-scan pour voir les 2 tips)")

    raw_target = X_raw[ascan_index].copy()
    denoised_target = X_denoised[ascan_index]
    envelope = extract_envelope(denoised_target)

    windowed_axis, windowed_envelope = _window(
        envelope, axis_values, depth_window[0], depth_window[1])

    detected_peaks = _detect_two_peaks_in_window(
        windowed_envelope, sep_config=sep_config,
        min_peak_prominence=min_peak_prominence, expected_n_peaks=2,
    )

    arrival = [float(windowed_axis[idx]) for idx, _ in detected_peaks]
    reliabilities = [prom > 0 for _, prom in detected_peaks]

    result = TipDetectionResult(
        ascan_index=ascan_index, raw_signal=raw_target, denoised_signal=denoised_target,
        envelope=envelope, axis_values=axis_values, surface_band=surface_band,
        backwall_band=backwall_band, diffraction_window=depth_window,
    )

    if len(arrival) >= 1:
        result.top_tip = arrival[0]

    if len(arrival) == 1:
        result.tips_merged = True
        if verbose:
            print("[Détection] Un seul écho de diffraction distinct trouvé sur cet "
                  "A-scan (index {}) : le second tip est probablement CONFONDU avec "
                  "le premier dans l'enveloppe de Hilbert".format(result.ascan_index))

    if len(arrival) >= 2:
        result.bottom_tip = arrival[1]
        result.tips_merged = False
        result.tips_reliable = all(reliabilities[:2])

        if not result.tips_reliable and verbose:
            print("[Détection] ATTENTION : les deux tips sont très rapprochés "
                  "(échos quasi fusionnés) -- ils ont été séparés via une "
                  "détection de secours par bande d'amplitude, moins fiable "
                  "qu'un vrai double pic. Vérifiez visuellement le résultat.")

        if c_mm_per_us is not None and d_mm is not None:
            result.height_mm = compute_height(arrival[0], arrival[1], c_mm_per_us, d_mm)

    return result, X_denoised


# ==============================================================================
# PART G — VISUALISATION
# ==============================================================================

def plot_tip_detection_result(result: TipDetectionResult, x_label: str = "Depth sample index",
                              save_path: Optional[str] = None):

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(result.axis_values, result.raw_signal, "k")
    axes[0].set_title(f"Raw (noisy) A-scan {result.ascan_index} -- kept untouched")

    axes[1].plot(result.axis_values, result.denoised_signal, "r")
    axes[1].set_title("Denoised A-scan")

    if result.surface_band:
        axes[1].axvspan(*result.surface_band, color="orange", alpha=0.2, label="Surface wave")
    if result.backwall_band:
        axes[1].axvspan(*result.backwall_band, color="purple", alpha=0.2, label="Backwall wave")
    axes[1].legend()

    axes[2].plot(result.axis_values, result.envelope, "b")

    if result.diffraction_window:
        axes[2].axvspan(*result.diffraction_window, color="green", alpha=0.1, label="Search window")
    if result.top_tip is not None:
        axes[2].axvline(result.top_tip, color="g", linestyle="--", label="Top tip")
    if result.bottom_tip is not None:
        axes[2].axvline(result.bottom_tip, color="m", linestyle="--", label="Bottom tip")

    axes[2].legend()
    title = "Hilbert envelope with detected tips"
    if result.tips_merged:
        title += "\n(ATTENTION : tips probablement CONFONDUS -- un seul écho distinct trouvé)"
    elif result.tips_reliable is False:
        title += "\n(tips séparés via détection de secours -- échos très rapprochés)"
    axes[2].set_title(title)
    axes[2].set_xlabel(x_label)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved: {save_path}")

    return fig, axes


# ==============================================================================
# PART H — VÉRIFICATION CROISÉE SUR LE B-SCAN BRUT (PAQUETS DE LIGNES)
# ==============================================================================
# Contrôle du résultat du pipeline (débruitage + réseau) en revenant à la
# matrice B-scan BRUTE : dans la fenêtre de recherche, on cherche les deux
# PAQUETS de lignes (5 à 10 lignes chacun) les plus intenses du B-scan,
# indépendamment du réseau, et on compare leurs centres aux tips du pipeline.

def bscan_row_intensity(X_raw: np.ndarray, axis_values: np.ndarray,
                        depth_window: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Intensité de chaque LIGNE du B-scan (= chaque profondeur) dans la fenêtre de
    recherche `depth_window`, calculée sur le signal BRUT.

    Dans l'image du B-scan, une ligne = un échantillon de profondeur, traversant
    tous les A-scans (colonnes). Les deux tips ressortent comme deux bandes
    horizontales intenses dans la fenêtre. Pour chaque profondeur, on moyenne
    l'enveloppe de Hilbert (recentrée) sur tous les A-scans.

    Renvoie (positions des lignes dans la fenêtre, intensité de chaque ligne).
    """
    centered = X_raw - np.median(X_raw, axis=1, keepdims=True)
    envelopes = np.abs(hilbert(centered, axis=1))

    mask = (axis_values >= depth_window[0]) & (axis_values <= depth_window[1])

    return axis_values[mask], envelopes[:, mask].mean(axis=0)


def _packet_around_peak(profile: np.ndarray, peak: int,
                        min_rows: int, max_rows: int) -> Tuple[int, int]:
    """
    Paquet de lignes autour d'un pic du profil d'intensité : on étend
    progressivement (du côté le plus intense) tant que l'intensité reste
    >= 50 % de celle du pic, sans dépasser `max_rows` lignes ; si le paquet est
    plus petit que `min_rows`, on l'élargit jusqu'à `min_rows`.
    Renvoie (indice_début, indice_fin) inclus, dans le profil.
    """
    n = len(profile)
    half = 0.5 * profile[peak]
    lo = hi = peak

    def extend(lo, hi, threshold):
        can_l = lo > 0 and profile[lo - 1] >= threshold
        can_r = hi < n - 1 and profile[hi + 1] >= threshold
        if not (can_l or can_r):
            return lo, hi, False
        if can_l and (not can_r or profile[lo - 1] >= profile[hi + 1]):
            return lo - 1, hi, True
        return lo, hi + 1, True

    while hi - lo + 1 < max_rows:
        lo, hi, moved = extend(lo, hi, half)
        if not moved:
            break

    while hi - lo + 1 < min_rows:
        lo, hi, moved = extend(lo, hi, -np.inf)
        if not moved:
            break

    return lo, hi


def raw_consensus_tips(X_raw: np.ndarray, axis_values: np.ndarray,
                       depth_window: Tuple[int, int],
                       sep_config: SeparationConfig,
                       min_peak_prominence: float = 0.15,
                       min_packet_rows: int = 5, max_packet_rows: int = 10
                       ) -> Tuple[Optional[float], Optional[float], List[Tuple[float, float]]]:
    """
    Avis indépendant sur les tips, basé sur des PAQUETS de lignes du B-scan brut
    (et non sur des A-scans individuels ni sur une seule ligne).

    Dans la fenêtre de recherche, on calcule l'intensité de chaque ligne (voir
    bscan_row_intensity) et on repère :
      - le pic le plus intense ;
      - un 2e pic PLAUSIBLE (mêmes critères que le pipeline, via
        `_is_plausible_peak` et `sep_config`), pour éviter de prendre un pic de
        bruit lointain pour le second tip.
    Autour de chaque pic, on construit un paquet de `min_packet_rows` à
    `max_packet_rows` lignes (voir _packet_around_peak). Si les deux paquets se
    touchent, ils sont coupés au CREUX qui sépare les deux pics. La position du
    tip = centre de gravité (pondéré par l'intensité) du paquet.

    Cas renvoyés :
      - DEUX paquets distincts : (top_tip_raw, bottom_tip_raw, [paquet1, paquet2])
        (le plus haut = tip haut).
      - UN SEUL lobe (les deux tips sont trop proches pour être séparés dans le
        profil des lignes) : (None, None, [lobe]) -- on ne devine PAS un tip.
      - Aucun pic : (None, None, []).
    """
    positions, profile = bscan_row_intensity(X_raw, axis_values, depth_window)

    if len(profile) == 0 or profile.max() <= 0:
        return None, None, []

    peak_idx, _ = find_peaks(profile, prominence=min_peak_prominence * profile.max(),
                             distance=max(1, min_packet_rows))

    if len(peak_idx) == 0:
        return None, None, []

    # Pic principal, puis 2e pic plausible (amplitude et distance, mêmes
    # critères que dans PART C via _is_plausible_peak).
    main = int(peak_idx[np.argmax(profile[peak_idx])])
    main_amp = float(profile[main])
    max_distance = (sep_config.max_absolute_separation
                    if sep_config.max_absolute_separation is not None else np.inf)

    others = [int(p) for p in peak_idx if p != main
              and _is_plausible_peak(profile[p], main_amp, abs(p - main), max_distance,
                                     sep_config.min_relative_second_peak_amplitude)]
    strongest = sorted([main] + ([max(others, key=lambda p: profile[p])] if others else []))

    packets = [list(_packet_around_peak(profile, p, min_packet_rows, max_packet_rows))
               for p in strongest]

    # Deux paquets : on les sépare au creux entre les deux pics (et non à
    # mi-distance), pour que chaque paquet reste centré sur SON tip.
    if len(strongest) == 2:
        p0, p1 = strongest
        valley = p0 + int(np.argmin(profile[p0:p1 + 1]))
        packets[0][1] = min(packets[0][1], valley)
        packets[1][0] = max(packets[1][0], valley + 1)

    # Position de chaque tip = centre de gravité du paquet (fond retiré).
    tips = []
    for peak, (lo, hi) in zip(strongest, packets):
        seg = profile[lo:hi + 1]
        w = seg - seg.min()
        tips.append(float(np.sum(positions[lo:hi + 1] * w) / w.sum()) if w.sum() > 0
                    else float(positions[peak]))

    packet_bounds = [(float(positions[lo]), float(positions[hi])) for lo, hi in packets]

    if len(tips) == 1:   # un seul lobe : tips confondus dans le profil des lignes
        return None, None, packet_bounds

    return tips[0], tips[1], packet_bounds


def verify_and_correct(result: TipDetectionResult, X_raw: np.ndarray,
                       sep_config: SeparationConfig,
                       min_peak_prominence: float = 0.15, tolerance_samples: float = 5.0,
                       c_mm_per_us: Optional[float] = None, d_mm: Optional[float] = None,
                       verbose: bool = True,
                       min_packet_rows: int = 5, max_packet_rows: int = 10
                       ) -> TipDetectionResult:
    """
    Compare les tips du pipeline (signal DÉBRUITÉ) aux centres des deux paquets
    de lignes les plus intenses du B-scan BRUT dans la fenêtre de recherche
    (raw_consensus_tips).

    - Deux paquets distincts trouvés sur le brut :
        * concordance (écart <= tolerance_samples pour chaque tip) : résultat
          initial déclaré correct et renvoyé tel quel ;
        * sinon : l'erreur est expliquée et une copie corrigée est renvoyée
          (tips remplacés par les centres des paquets bruts, hauteur recalculée
          si calibration fournie).
    - UN SEUL lobe (ou rien) sur le brut : les deux tips sont trop proches pour
      être séparés dans le profil des lignes, donc le brut ne peut PAS juger
      chaque tip séparément. Aucune correction n'est appliquée.

    `result` n'est jamais modifié sur place.
    """

    if result.diffraction_window is None:
        if verbose:
            print("[Vérification] Impossible : pas de fenêtre de diffraction définie.")
        return result

    top_tip_raw, bottom_tip_raw, packets = raw_consensus_tips(
        X_raw, result.axis_values, result.diffraction_window, sep_config=sep_config,
        min_peak_prominence=min_peak_prominence,
        min_packet_rows=min_packet_rows, max_packet_rows=max_packet_rows,
    )

    if verbose:
        print(f"\n[Vérification] Paquets de lignes les plus intenses du B-scan brut "
              f"(début, fin) : {packets}")
        print(f"[Vérification] Centres des paquets -> top_tip={top_tip_raw}, bottom_tip={bottom_tip_raw}")

    # --- Pas deux paquets distincts : aucune correction possible. ---
    if top_tip_raw is None or bottom_tip_raw is None:
        if verbose:
            if len(packets) == 1:
                lo, hi = packets[0]
                print(f"[Vérification] Un seul lobe intense sur le B-scan brut ({lo}, {hi}) : "
                      f"les deux tips sont trop proches pour être séparés dans le profil "
                      f"des lignes. Aucune correction appliquée (le pipeline est conservé).")
                for label, tip in (("haut", result.top_tip), ("bas", result.bottom_tip)):
                    if tip is not None and not (lo - tolerance_samples <= tip <= hi + tolerance_samples):
                        print(f"[Vérification] ATTENTION : le tip {label} du pipeline ({tip}) est "
                              f"en dehors du lobe brut : vérifiez visuellement.")
            else:
                print("[Vérification] Aucun paquet exploitable sur le B-scan brut : "
                      "aucune correction appliquée (le pipeline est conservé).")
        return copy.deepcopy(result)

    was_correct = True
    corrected = copy.deepcopy(result)

    for attr, label, raw_val in (("top_tip", "haut", top_tip_raw),
                                 ("bottom_tip", "bas", bottom_tip_raw)):
        current = getattr(result, attr)

        if current is None:
            was_correct = False
            setattr(corrected, attr, raw_val)
            if verbose:
                print(f"[Vérification] ERREUR : aucun tip {label} détecté par le pipeline, "
                      f"alors que le signal brut en montre un vers {raw_val}. Correction appliquée.")

        elif abs(current - raw_val) > tolerance_samples:
            was_correct = False
            setattr(corrected, attr, raw_val)
            if verbose:
                print(f"[Vérification] ERREUR : tip {label} détecté à {current}, "
                      f"mais le paquet de lignes brutes indique {raw_val} "
                      f"(écart > {tolerance_samples}). Correction appliquée.")

    if was_correct:
        if verbose:
            print("[Vérification] OK : le pipeline initial est cohérent avec le signal brut, "
                  "aucune correction nécessaire.")
        return corrected

    # Les deux tips sont maintenant définis (le brut en avait deux distincts).
    if c_mm_per_us is not None and d_mm is not None:
        corrected.height_mm = compute_height(
            corrected.top_tip, corrected.bottom_tip, c_mm_per_us, d_mm,
        )

    if result.tips_merged and verbose:
        print("[Vérification] Les deux tips étaient marqués CONFONDUS sur "
              "l'A-scan sélectionné par le pipeline, mais les paquets de lignes "
              "les plus intenses du B-scan brut (fenêtre de recherche) ont permis "
              "de les distinguer. Le résultat corrigé n'est donc plus considéré "
              "comme confondu.")
    corrected.tips_merged = False

    if verbose:
        print(f"[Vérification] Résultat corrigé -> top_tip={corrected.top_tip}, "
              f"bottom_tip={corrected.bottom_tip}, height_mm={corrected.height_mm}")

    return corrected


# ==============================================================================
# B-SCAN SYNTHÉTIQUE DE DÉMONSTRATION (repli si le fichier est introuvable)
# ==============================================================================

def wave_packet(axis, center, amplitude, width=6, period=12):
    """Paquet d'onde ultrasonore localisé (enveloppe gaussienne x cosinus)."""
    envelope = np.exp(-((axis - center) ** 2) / (2 * width ** 2))
    return amplitude * envelope * np.cos(2 * np.pi * (axis - center) / period)


def generate_synthetic_bscan(N: int = 120, L: int = 900,
                             surface_pos: int = 120, backwall_pos: int = 700,
                             top_tip_pos: int = 380, bottom_tip_pos: int = 490) -> np.ndarray:
    """B-scan TOFD synthétique réaliste (hyperboles de diffraction + bruit)."""

    axis_demo = np.arange(L)
    x_center = (N - 1) / 2
    X_raw = np.zeros((N, L), dtype=np.float32)

    for x in range(N):

        # Surface et backwall (légère variation de position/amplitude).
        surface = surface_pos + np.random.normal(0, 1.5)
        backwall = backwall_pos + np.random.normal(0, 2.0)

        signal = wave_packet(axis_demo, surface, np.random.uniform(0.65, 0.95), width=7, period=12)
        signal += wave_packet(axis_demo, backwall, np.random.uniform(0.60, 0.90), width=8, period=13)

        # Diffraction : le temps d'arrivée apparent varie avec la position
        # de la sonde (signature hyperbolique dans le B-scan).
        lateral_distance = (x - x_center) * 3.0
        shift = 0.0025 * lateral_distance ** 2
        spatial_factor = np.exp(-(lateral_distance ** 2) / (2 * 120 ** 2))

        signal += wave_packet(axis_demo, top_tip_pos + shift, 0.35 * spatial_factor, width=5, period=10)
        signal += wave_packet(axis_demo, bottom_tip_pos + shift, 0.30 * spatial_factor, width=5, period=10)

        # Fond structurel de faible amplitude, puis bruit aléatoire.
        signal += 0.03 * np.sin(2 * np.pi * axis_demo / np.random.uniform(70, 100))
        signal += np.random.uniform(0.08, 0.18) * np.random.randn(L)

        # Petite variation d'amplitude globale.
        signal *= np.random.uniform(0.90, 1.10)

        X_raw[x, :] = signal

    print(f"Synthetic B-scan generated: {N} A-scans × {L} samples")
    print(f"(positions réelles pour comparaison : top={top_tip_pos}, bottom={bottom_tip_pos})")

    return X_raw


# ==============================================================================
# BLOC PRINCIPAL
# ==============================================================================
# 1. Charge le B-scan (repli sur un B-scan synthétique si introuvable).
# 2. Pipeline complet (débruitage -> localisation -> détection de tips).
# 3. Vérification contre les deux paquets de lignes les plus intenses du
#    B-scan brut (dans la fenêtre de recherche) + correction.

if __name__ == "__main__":

    SEP_CONFIG = SeparationConfig(
        max_separation_factor=1.0,
        min_relative_second_peak_amplitude=0.3,
        band_ratio=0.15,
        max_absolute_separation=50,
    )

    # 1. Chargement du B-scan.
    try:
        X_raw = load_bscan(
            BSCAN_PATH,
            transpose=BSCAN_TRANSPOSE,
            assume_diverging_grey=BSCAN_ASSUME_DIVERGING_GREY,
            crop_rows=BSCAN_CROP_ROWS,
            crop_cols=BSCAN_CROP_COLS,
        )
        print(f"B-scan chargé depuis '{BSCAN_PATH}' : {X_raw.shape[0]} A-scans "
              f"de longueur {X_raw.shape[1]}")

    except (FileNotFoundError, ValueError) as e:
        print(f"Impossible de charger le B-scan depuis '{BSCAN_PATH}' ({e}).")
        print("-> Génération d'un B-scan SYNTHÉTIQUE de démonstration à la place.\n")
        X_raw = generate_synthetic_bscan()

    N, L = X_raw.shape
    axis = np.arange(L)

    print(f"\nDataset : {N} A-scans de longueur {L}")

    # 2. Pipeline complet.
    result, X_denoised = extract_tips(
        X_raw, n_clusters=3, sep_config=SEP_CONFIG, ascan_index=None, depth_window=None,
        axis_values=axis, min_peak_prominence=0.1, n_hidden=20,
        n_iterations=1500, batch_size=5, c_mm_per_us=None, d_mm=None,
        device="cpu", verbose=True,
    )

    print(f"\nA-scan {result.ascan_index} (résultat brut du pipeline) :")
    print(f"  Surface band        : {result.surface_band}")
    print(f"  Backwall band       : {result.backwall_band}")
    print(f"  Diffraction window  : {result.diffraction_window}")
    print(f"  Top tip détecté     : {result.top_tip}")
    print(f"  Bottom tip détecté  : {result.bottom_tip}")
    print(f"  Tips fiables ?      : {result.tips_reliable}")
    print(f"  Tips confondus ?    : {result.tips_merged}")
    print(f"  Height              : {result.height_mm}")

    # 3. Vérification croisée sur le B-scan brut + auto-correction.
    corrected_result = verify_and_correct(
        result, X_raw, sep_config=SEP_CONFIG, min_peak_prominence=0.1,
        tolerance_samples=5.0, c_mm_per_us=None, d_mm=None, verbose=True,
        min_packet_rows=MIN_PACKET_ROWS, max_packet_rows=MAX_PACKET_ROWS,
    )

    print("\n=== Résultat FINAL (après vérification) ===")
    print(f"  Top tip    : {corrected_result.top_tip}")
    print(f"  Bottom tip : {corrected_result.bottom_tip}")
    print(f"  Height     : {corrected_result.height_mm}")

    plot_tip_detection_result(
        corrected_result, x_label="Depth sample index",
        save_path="ascan_tip_detection_verified.png",
    )
