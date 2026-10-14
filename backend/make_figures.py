"""
make_figures.py : génère les figures du site pour chaque étape du pipeline, à partir de data_1.

À placer dans le MÊME dossier que tofd_pipeline.py (C:\\Users\\Alix\\Documents\\tofd), puis :
    python make_figures.py

Il crée un dossier `images/` avec :
    step1_clusters.png      étape 1 : cluster de chaque colonne + A-scan moyen de chaque cluster
    step2_denoising.png     étape 2 : B-scan brut / débruité + un A-scan avant et après
    step4_separation.png    étape 4 : bande autour du pic, sorties de bande, rayon de recherche
    step5_verification.png  étape 5 : profil d'intensité des lignes du B-scan brut, paquets, tips

Dépose ensuite ces 4 fichiers dans le dossier images/ du dépôt GitHub (mêmes noms).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from tofd_pipeline import (
    BSCAN_PATH, BSCAN_TRANSPOSE, BSCAN_ASSUME_DIVERGING_GREY,
    BSCAN_CROP_ROWS, BSCAN_CROP_COLS, MIN_PACKET_ROWS, MAX_PACKET_ROWS,
    SeparationConfig, load_bscan, generate_synthetic_bscan,
    normalize_signals, cluster_signals, extract_tips, verify_and_correct,
    bscan_row_intensity, raw_consensus_tips, _window, _half_max_width,
)

OUT_DIR = Path("images")
N_CLUSTERS = 3
N_ITERATIONS = 1500
SEP_CONFIG = SeparationConfig(max_separation_factor=1.0,
                              min_relative_second_peak_amplitude=0.3,
                              band_ratio=0.15, max_absolute_separation=50)
CLUSTER_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


def show_bscan(ax, X, title, vmax):
    """B-scan : une colonne = un A-scan, une ligne = une profondeur."""
    ax.imshow(X.T, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("A-scan index")
    ax.set_ylabel("Depth sample index")


# ------------------------------------------------------------------------------
# Étape 1 : clusters
# ------------------------------------------------------------------------------
def figure_clusters(X_raw, path):
    N, L = X_raw.shape
    X_norm, _, _ = normalize_signals(X_raw)
    labels, _ = cluster_signals(X_norm, N_CLUSTERS)       # même graine que le pipeline

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[4, 0.35, 3])

    ax0 = fig.add_subplot(gs[0])
    show_bscan(ax0, X_raw, "B-scan and cluster of each A-scan", np.max(np.abs(X_raw)))
    ax0.set_xlabel("")

    # Bandeau coloré : couleur du cluster de chaque colonne, aligné sous le B-scan.
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    for c in range(N_CLUSTERS):
        ax1.fill_between(np.arange(N), 0, 1, where=(labels == c), step="mid",
                         color=CLUSTER_COLORS[c], label=f"Cluster {c} ({int(np.sum(labels == c))})")
    ax1.set_yticks([])
    ax1.set_xlabel("A-scan index")
    ax1.legend(ncol=N_CLUSTERS, loc="upper center", bbox_to_anchor=(0.5, -1.2), frameon=False)

    # A-scan moyen (normalisé) de chaque cluster.
    ax2 = fig.add_subplot(gs[2])
    for c in range(N_CLUSTERS):
        if np.any(labels == c):
            ax2.plot(X_norm[labels == c].mean(axis=0), color=CLUSTER_COLORS[c], label=f"Cluster {c}")
    ax2.set_title("Average normalised A-scan of each cluster")
    ax2.set_xlabel("Depth sample index")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved", path)


# ------------------------------------------------------------------------------
# Étape 2 : débruitage
# ------------------------------------------------------------------------------
def figure_denoising(X_raw, X_denoised, ascan_index, path):
    vmax = np.max(np.abs(X_raw))
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    show_bscan(ax0, X_raw, "Raw B-scan", vmax)
    ax1 = fig.add_subplot(gs[0, 1], sharex=ax0, sharey=ax0)
    show_bscan(ax1, X_denoised, "Denoised B-scan", vmax)
    ax1.set_ylabel("")

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(X_raw[ascan_index], color="0.6", label="Raw")
    ax2.plot(X_denoised[ascan_index], color="r", label="Denoised")
    ax2.set_title(f"A-scan {ascan_index}")
    ax2.set_xlabel("Depth sample index")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved", path)


# ------------------------------------------------------------------------------
# Étape 4 : séparation d'un lobe (bande autour du pic)
# ------------------------------------------------------------------------------
def figure_separation(result, path):
    d0, d1 = result.diffraction_window
    axis_w, env_w = _window(result.envelope, result.axis_values, d0, d1)

    peak = int(np.argmax(env_w))                    # sommet du lobe principal
    A = float(env_w[peak])
    lower = A * (1.0 - SEP_CONFIG.band_ratio)

    # Sortie de bande à gauche / à droite (même logique que _split_via_band_exit).
    exit_left = next((i for i in range(peak - 1, -1, -1) if env_w[i] < lower), 0)
    exit_right = next((i for i in range(peak + 1, len(env_w)) if env_w[i] < lower), len(env_w) - 1)

    # Rayon de recherche d'un second tip.
    w = max(2, int(np.ceil(_half_max_width(env_w, peak))))
    radius = max(3, int(SEP_CONFIG.max_separation_factor * w))
    if SEP_CONFIG.max_absolute_separation is not None:
        radius = min(radius, int(SEP_CONFIG.max_absolute_separation))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(axis_w, env_w, "b", label="Hilbert envelope")
    ax.axhspan(lower, A, color="tab:blue", alpha=0.12, label=f"Band [A(1-{SEP_CONFIG.band_ratio}), A]")
    ax.axvspan(axis_w[max(0, peak - radius)], axis_w[min(len(axis_w) - 1, peak + radius)],
               color="green", alpha=0.07, label=f"Search radius ({radius} samples)")
    ax.axvline(axis_w[exit_left], color="k", linestyle=":", label="Exit of the band")
    ax.axvline(axis_w[exit_right], color="k", linestyle=":")
    ax.plot(axis_w[peak], A, "o", color="tab:green", label="Top of the lobe (A)")
    if result.top_tip is not None:
        ax.axvline(result.top_tip, color="g", linestyle="--", label="Top tip")
    if result.bottom_tip is not None:
        ax.axvline(result.bottom_tip, color="m", linestyle="--", label="Bottom tip")
    ax.set_title("Separation of two close tips")
    ax.set_xlabel("Depth sample index")
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved", path)


# ------------------------------------------------------------------------------
# Étape 5 : vérification sur le B-scan brut
# ------------------------------------------------------------------------------
def figure_verification(result, X_raw, path):
    positions, profile = bscan_row_intensity(X_raw, result.axis_values, result.diffraction_window)
    top_raw, bot_raw, packets = raw_consensus_tips(
        X_raw, result.axis_values, result.diffraction_window, sep_config=SEP_CONFIG,
        min_peak_prominence=0.1, min_packet_rows=MIN_PACKET_ROWS, max_packet_rows=MAX_PACKET_ROWS)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(positions, profile, "k", label="Row intensity (raw B-scan)")
    for k, (lo, hi) in enumerate(packets):
        ax.axvspan(lo, hi, color=["tab:orange", "tab:purple"][k % 2], alpha=0.25,
                   label=f"Packet {k + 1}")
    if top_raw is not None:
        ax.axvline(top_raw, color="g", label="Top tip, centre of packet")
    if bot_raw is not None:
        ax.axvline(bot_raw, color="m", label="Bottom tip, centre of packet")
    if result.top_tip is not None:
        ax.axvline(result.top_tip, color="g", linestyle="--", label="Top tip, pipeline")
    if result.bottom_tip is not None:
        ax.axvline(result.bottom_tip, color="m", linestyle="--", label="Bottom tip, pipeline")
    ax.set_title("Check on the raw B-scan: packets of rows")
    ax.set_xlabel("Depth sample index")
    ax.set_ylabel("Mean envelope over all A-scans")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved", path)


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    np.random.seed(0)
    torch.manual_seed(0)
    OUT_DIR.mkdir(exist_ok=True)

    try:
        X_raw = load_bscan(BSCAN_PATH, transpose=BSCAN_TRANSPOSE,
                           assume_diverging_grey=BSCAN_ASSUME_DIVERGING_GREY,
                           crop_rows=BSCAN_CROP_ROWS, crop_cols=BSCAN_CROP_COLS)
    except (FileNotFoundError, ValueError) as e:
        print(f"B-scan introuvable ({e}) : B-scan synthétique à la place.")
        X_raw = generate_synthetic_bscan()

    axis = np.arange(X_raw.shape[1])

    # Pipeline complet une seule fois ; les figures réutilisent ses résultats.
    result, X_denoised = extract_tips(
        X_raw, n_clusters=N_CLUSTERS, sep_config=SEP_CONFIG, ascan_index=None,
        depth_window=None, axis_values=axis, min_peak_prominence=0.1, n_hidden=20,
        n_iterations=N_ITERATIONS, batch_size=5, device="cpu", verbose=True)

    figure_clusters(X_raw, OUT_DIR / "step1_clusters.png")
    figure_denoising(X_raw, X_denoised, result.ascan_index, OUT_DIR / "step2_denoising.png")
    figure_separation(result, OUT_DIR / "step4_separation.png")
    figure_verification(result, X_raw, OUT_DIR / "step5_verification.png")

    print("\nTerminé : 4 figures dans le dossier", OUT_DIR.resolve())
