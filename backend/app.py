import io
import os
import tempfile

import matplotlib.pyplot as plt
import streamlit as st

from tofd_pipeline import (
    load_bscan, generate_synthetic_bscan, extract_tips, verify_and_correct,
    plot_tip_detection_result, SeparationConfig,
)

st.set_page_config(page_title="TOFD - Tip extraction", layout="wide")

# ------------------------------------------------------------------ Style ---
st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1200px;}
    h1 {letter-spacing: -0.5px; margin-bottom: 0;}
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 14px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] {opacity: 0.7;}
    [data-testid="stMetricValue"] {font-weight: 600;}
    section[data-testid="stSidebar"] [data-testid="stExpander"] {border-radius: 10px;}
    button[data-baseweb="tab"] {font-weight: 500;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- Helpers ---
def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def bscan_figure(X_raw, res):
    """B-scan vu de dessus (profondeur en vertical) avec fenêtre et tips."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(X_raw.T, aspect="auto", cmap="gray")

    if res.diffraction_window:
        ax.axhspan(*res.diffraction_window, color="green", alpha=0.12, label="Search window")
    ax.axvline(res.ascan_index, color="orange", lw=1, ls=":", label=f"A-scan {res.ascan_index}")
    if res.top_tip is not None:
        ax.axhline(res.top_tip, color="lime", ls="--", lw=1.2, label="Top tip")
    if res.bottom_tip is not None:
        ax.axhline(res.bottom_tip, color="magenta", ls="--", lw=1.2, label="Bottom tip")

    ax.set_xlabel("A-scan index (probe position)")
    ax.set_ylabel("Depth sample index")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- Sidebar ---
with st.sidebar:
    st.header("Settings")
    run_button = st.button("Run analysis", type="primary")

    with st.expander("Data", expanded=True):
        uploaded = st.file_uploader("B-scan image", type=["png", "jpg", "jpeg", "bmp"])
        bscan_path = st.text_input("...or local path (optional)", value="")
        use_synthetic = st.checkbox(
            "Use a simulated B-scan (demo)", value=False,
            help="Ignores the path and generates a synthetic B-scan with known tips "
                 "(top = 380, bottom = 490).",
        )

    with st.expander("Denoising"):
        n_clusters = st.slider(
            "Number of clusters", 1, 10, 3,
            help="Number of groups of similar A-scans (k-means). One autoencoder is "
                 "trained per group, on mini-batches of 5 A-scans.",
        )
        n_iterations = st.slider(
            "Training iterations", 200, 3000, 1500, step=100,
            help="Training steps per autoencoder. Run time grows with this value and "
                 "the number of clusters; gains are small beyond ~2000.",
        )

    with st.expander("Peak detection", expanded=True):
        min_peak_prominence = st.slider(
            "Minimum prominence", 0.01, 0.5, 0.10, step=0.01,
            help="How much a peak must stand out from its surroundings, as a fraction "
                 "of the envelope maximum. Lower = more peaks (some may be noise).",
        )

    with st.expander("Separation of close tips"):
        max_separation_factor = st.slider(
            "Separation factor", 0.5, 5.0, 1.0, step=0.1,
            help="Search distance for the second tip, in multiples of the main peak's "
                 "half-height width.",
        )
        min_relative_amp = st.slider(
            "Min. relative amplitude of 2nd tip", 0.05, 1.0, 0.3, step=0.05,
            help="The second tip must reach at least this fraction of the main tip's amplitude.",
        )
        band_ratio = st.slider(
            "Band ratio", 0.01, 0.5, 0.15, step=0.01,
            help="Tolerance zone below the main peak's top. The second tip must lie "
                 "outside it (i.e. the minimum dip between the two peaks).",
        )
        max_absolute_separation = st.slider(
            "Max absolute separation (samples)", 5, 200, 50, step=5,
            help="Maximum distance between the two tips for the pair to be considered realistic.",
        )

    with st.expander("Verification on raw B-scan"):
        tolerance_samples = st.slider(
            "Tolerance (samples)", 1.0, 20.0, 5.0, step=1.0,
            help="Max gap between a pipeline tip and the packet center found on the raw "
                 "B-scan. Above it, the tip is replaced.",
        )
        min_packet_rows = st.slider("Min. rows per packet", 2, 20, 5)
        max_packet_rows = st.slider("Max. rows per packet", 5, 40, 10)

    with st.expander("Calibration (optional)"):
        use_calib = st.checkbox("Compute defect height", value=False)
        c_mm_per_us = st.number_input("Sound velocity (mm/µs)", value=5.9) if use_calib else None
        d_mm = st.number_input("Half probe spacing (mm)", value=10.0) if use_calib else None


# ------------------------------------------------------------------- Run ---
if run_button:
    sep_config = SeparationConfig(
        max_separation_factor=max_separation_factor,
        min_relative_second_peak_amplitude=min_relative_amp,
        band_ratio=band_ratio,
        max_absolute_separation=int(max_absolute_separation),
    )

    source_note = None
    with st.spinner("Loading B-scan..."):
        if use_synthetic:
            X_raw = generate_synthetic_bscan()
            source_note = ("info", "Simulated B-scan (top tip = 380, bottom tip = 490).")
        else:
            try:
                path = bscan_path
                if uploaded is not None:
                    suffix = os.path.splitext(uploaded.name)[1] or ".png"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded.getvalue())
                    path = tmp.name
                if not path:
                    raise FileNotFoundError("no file provided")
                X_raw = load_bscan(path, assume_diverging_grey=True)
            except (FileNotFoundError, ValueError) as e:
                X_raw = generate_synthetic_bscan()
                source_note = ("warning",
                               f"Could not load the B-scan ({e}). "
                               f"A simulated B-scan is used instead.")

    with st.spinner("Denoising and detecting tips..."):
        result, _ = extract_tips(
            X_raw, n_clusters=n_clusters, sep_config=sep_config,
            min_peak_prominence=min_peak_prominence, n_iterations=n_iterations,
            c_mm_per_us=c_mm_per_us, d_mm=d_mm, verbose=False,
        )

    corrected = verify_and_correct(
        result, X_raw, sep_config=sep_config, tolerance_samples=tolerance_samples,
        c_mm_per_us=c_mm_per_us, d_mm=d_mm, verbose=False,
        min_packet_rows=min_packet_rows, max_packet_rows=max_packet_rows,
    )

    # Résultats gardés en mémoire : ils restent affichés quand on bouge un slider.
    st.session_state["run"] = {
        "corrected": corrected,
        "X_raw": X_raw,
        "source_note": source_note,
        "params": {
            "denoising": {"clusters": n_clusters, "iterations": n_iterations},
            "detection": {"min_prominence": min_peak_prominence},
            "separation": {
                "separation_factor": max_separation_factor,
                "min_relative_amplitude": min_relative_amp,
                "band_ratio": band_ratio,
                "max_absolute_separation": int(max_absolute_separation),
            },
            "verification": {
                "tolerance_samples": tolerance_samples,
                "packet_rows": [min_packet_rows, max_packet_rows],
            },
            "calibration": {"sound_velocity": c_mm_per_us, "half_spacing": d_mm},
        },
    }


# ------------------------------------------------------------------ Page ---
st.title("Tip extraction (TOFD)")
st.caption("Tune the parameters in the sidebar, then run the analysis.")

if "run" not in st.session_state:
    st.info("No result yet. Set the parameters in the sidebar and click **Run analysis**.")
else:
    run = st.session_state["run"]
    corrected = run["corrected"]
    X_raw = run["X_raw"]

    if run["source_note"]:
        level, text = run["source_note"]
        getattr(st, level)(text)

    n_ascans, n_samples = X_raw.shape
    st.caption(f"B-scan: {n_ascans} A-scans x {n_samples} samples")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("A-scan chosen", corrected.ascan_index)
    c2.metric("Top tip", f"{corrected.top_tip:.1f}" if corrected.top_tip is not None else "—")
    c3.metric("Bottom tip", f"{corrected.bottom_tip:.1f}" if corrected.bottom_tip is not None else "—")
    c4.metric("Height (mm)", f"{corrected.height_mm:.2f}" if corrected.height_mm is not None else "—")

    st.write("")
    if corrected.tips_merged:
        st.warning("The two tips seem merged: only one distinct echo was found.")
    elif corrected.tips_reliable is False:
        st.info("Tips were separated by the fallback detection (very close echoes). "
                "A visual check is recommended.")
    else:
        st.success("Two reliable tips detected.")

    tab_plot, tab_bscan, tab_params = st.tabs(["Detection", "B-scan overview", "Parameters used"])

    with tab_plot:
        fig, _ = plot_tip_detection_result(corrected)
        st.pyplot(fig)
        st.download_button("Download figure (PNG)", fig_to_png(fig),
                           file_name="tip_detection.png", mime="image/png")
        plt.close(fig)

    with tab_bscan:
        fig_b = bscan_figure(X_raw, corrected)
        st.pyplot(fig_b)
        plt.close(fig_b)

    with tab_params:
        st.json(run["params"])
