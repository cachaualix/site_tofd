import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image


# ==============================================================================
# PART 0 — CHARGEMENT DU B-SCAN ET EXTRACTION DES A-SCANS
# ==============================================================================
# Copié de tofd_pipeline.py pour que ce script tourne seul.

BSCAN_PATH = r"C:\Users\Alix\Desktop\STG2\documents for datasets\B-scans\B-scans\data_1.png"

BSCAN_CROP_ROWS: Optional[Sequence[int]] = None
BSCAN_CROP_COLS: Optional[Sequence[int]] = None

# False : pixel [0, 255] ramené à [0, 1] (courbe de brillance, toujours positive).
# True  : gris moyen (~127.5) = amplitude nulle, pixel remis à l'échelle [-1, 1]
BSCAN_ASSUME_DIVERGING_GREY = True
BSCAN_TRANSPOSE = False


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
# VISUALISATION AVANT / APRÈS
# ==============================================================================

def plot_denoising_result(X_raw: np.ndarray, X_denoised: np.ndarray,
                          ascan_index: Optional[int] = None,
                          save_path: Optional[str] = None):
    """
    Ligne du haut : B-scan brut | B-scan débruité (même échelle de gris).
    Ligne du bas  : un A-scan, brut en gris et débruité en rouge, superposés.
    La figure enregistrée peut être mise telle quelle dans la section
    « B-scan denoising » du site.
    """
    N, L = X_raw.shape
    if ascan_index is None:
        ascan_index = N // 2

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    vmax = np.max(np.abs(X_raw))

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(X_raw.T, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax)
    ax0.set_title("Raw B-scan")
    ax0.set_xlabel("A-scan index")
    ax0.set_ylabel("Depth sample index")

    ax1 = fig.add_subplot(gs[0, 1], sharex=ax0, sharey=ax0)
    ax1.imshow(X_denoised.T, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax)
    ax1.set_title("Denoised B-scan")
    ax1.set_xlabel("A-scan index")

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(X_raw[ascan_index], color="0.6", label="Raw")
    ax2.plot(X_denoised[ascan_index], color="r", label="Denoised")
    ax2.set_title(f"A-scan {ascan_index}")
    ax2.set_xlabel("Depth sample index")
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved: {save_path}")

    return fig


# ==============================================================================
# BLOC PRINCIPAL
# ==============================================================================
# 1. Charge le B-scan (repli sur un B-scan synthétique si introuvable).
# 2. Débruite : k-means, puis un autoencodeur par groupe.
# 3. Enregistre la figure avant / après.

if __name__ == "__main__":

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
        # simulation.py doit être dans le même dossier que ce script.
        from simulation import generate_synthetic_bscan
        X_raw = generate_synthetic_bscan()

    X_denoised, labels, models = denoise_full_dataset(
        X_raw, n_clusters=3, n_hidden=20, n_iterations=1500, batch_size=5,
        device="cpu", verbose=True,
    )

    plot_denoising_result(X_raw, X_denoised, ascan_index=None,
                          save_path="denoising_before_after.png")
    plt.show()
