import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# B-SCAN SYNTHÉTIQUE DE DÉMONSTRATION
# ==============================================================================
# Extrait de tofd_pipeline.py : même code, sans le reste du pipeline.
# Utilisé par le site (onglets A-scan / B-scan de la simulation) et comme repli
# dans tofd_pipeline.py quand le fichier du B-scan est introuvable.

def wave_packet(axis, center, amplitude, width=6, period=12):
    """Paquet d'onde ultrasonore localisé (enveloppe gaussienne x cosinus)."""
    envelope = np.exp(-((axis - center) ** 2) / (2 * width ** 2))
    return amplitude * envelope * np.cos(2 * np.pi * (axis - center) / period)


def generate_synthetic_bscan(N: int = 120, L: int = 900,
                             surface_pos: int = 120, backwall_pos: int = 700,
                             top_tip_pos: int = 380, bottom_tip_pos: int = 490,
                             seed: Optional[int] = None) -> np.ndarray:
    """
    B-scan TOFD synthétique réaliste (hyperboles de diffraction + bruit).

    Ajout par rapport à tofd_pipeline.py : `seed` (None = comportement d'origine,
    bruit différent à chaque appel ; un entier = bruit reproductible).
    """
    if seed is not None:
        np.random.seed(seed)

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
# VISUALISATION
# ==============================================================================

def plot_synthetic_bscan(X_raw: np.ndarray, ascan_index: Optional[int] = None,
                         save_path: Optional[str] = None):
    """
    À gauche : le B-scan (colonne = A-scan, ligne = profondeur).
    À droite : un seul A-scan (profondeur à la verticale, amplitude à l'horizontale).
    La ligne orange du B-scan indique la position de l'A-scan affiché.
    """
    N, L = X_raw.shape
    if ascan_index is None:
        ascan_index = N // 2

    fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(11, 6),
                                     gridspec_kw={"width_ratios": [3, 1]}, sharey=True)

    vmax = np.max(np.abs(X_raw))
    ax_b.imshow(X_raw.T, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax)
    ax_b.axvline(ascan_index, color="orange", linestyle="--", linewidth=1)
    ax_b.set_title("Synthetic B-scan")
    ax_b.set_xlabel("A-scan index (probe position)")
    ax_b.set_ylabel("Depth sample index")

    ax_a.plot(X_raw[ascan_index], np.arange(L), "k", linewidth=0.8)
    ax_a.set_title(f"A-scan {ascan_index}")
    ax_a.set_xlabel("Amplitude")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved: {save_path}")

    return fig, (ax_b, ax_a)


# ==============================================================================
# BLOC PRINCIPAL
# ==============================================================================

if __name__ == "__main__":

    X_raw = generate_synthetic_bscan(seed=7)

    plot_synthetic_bscan(X_raw, ascan_index=None, save_path="simulation_bscan.png")
    plt.show()
