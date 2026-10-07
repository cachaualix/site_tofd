import io
import streamlit as st
import matplotlib.pyplot as plt
from tofd_modules.pipeline import (
    load_bscan, 
    generate_synthetic_bscan, 
    extract_tips, 
    verify_and_correct,
    plot_tip_detection_result, 
    SeparationConfig,
)
from tofd_modules.utils import fig_to_png

# --- Config ---
st.set_page_config(
    page_title="TOFD Analysis Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1400px;}
    h1 {letter-spacing: -0.5px; margin-bottom: 0.5rem; color: #6ee7f9;}
    h2 {color: #6ee7f9; margin-top: 1.5rem;}
    .stMetric {
        background: rgba(110, 231, 249, 0.08);
        border: 1px solid rgba(110, 231, 249, 0.2);
        border-radius: 12px;
        padding: 16px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6ee7f9, #7c9dff);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"] {font-weight: 500;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("## 🎯 TOFD")
with col2:
    st.markdown("**Analysis Platform** — Ultrasonic B-scan analysis, denoising & defect detection")

st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    run_button = st.button("▶️ Run Analysis", type="primary", use_container_width=True)

    with st.expander("📂 Data", expanded=True):
        use_synthetic = st.checkbox(
            "Use synthetic B-scan (demo)",
            value=True,
            help="Generate a demo B-scan with known defects"
        )
        if not use_synthetic:
            bscan_path = st.text_input(
                "B-scan path or upload",
                value="data_1.png",
            )

    with st.expander("🧹 Denoising"):
        n_clusters = st.slider("Clusters", 1, 10, 3)
        n_iterations = st.slider("Iterations", 500, 3000, 1500, step=100)

    with st.expander("🎯 Peak Detection", expanded=True):
        min_peak_prominence = st.slider("Min. prominence", 0.01, 0.5, 0.1, step=0.01)

    with st.expander("🔍 Tip Separation"):
        max_separation_factor = st.slider("Separation factor", 0.5, 5.0, 1.0, step=0.1)
        min_relative_amp = st.slider("Min. relative amplitude", 0.05, 1.0, 0.3, step=0.05)
        band_ratio = st.slider("Band ratio", 0.01, 0.5, 0.15, step=0.01)
        max_absolute_separation = st.slider("Max separation (samples)", 5, 200, 50, step=5)

    with st.expander("✅ Verification"):
        tolerance_samples = st.slider("Tolerance (samples)", 1.0, 20.0, 5.0, step=1.0)
        min_packet_rows = st.slider("Min. packet rows", 2, 20, 5)
        max_packet_rows = st.slider("Max. packet rows", 5, 40, 10)

    with st.expander("📏 Calibration (optional)"):
        use_calib = st.checkbox("Compute height")
        c_mm_per_us = st.number_input("Sound velocity (mm/µs)", value=5.9) if use_calib else None
        d_mm = st.number_input("Half probe spacing (mm)", value=10.0) if use_calib else None

# --- Main ---
if run_button:
    sep_config = SeparationConfig(
        max_separation_factor=max_separation_factor,
        min_relative_second_peak_amplitude=min_relative_amp,
        band_ratio=band_ratio,
        max_absolute_separation=int(max_absolute_separation),
    )

    with st.spinner("Loading B-scan..."):
        if use_synthetic:
            X_raw = generate_synthetic_bscan()
            st.session_state["source_note"] = "✅ Synthetic B-scan loaded"
        else:
            try:
                X_raw = load_bscan(bscan_path, assume_diverging_grey=True)
                st.session_state["source_note"] = f"✅ Loaded: {bscan_path}"
            except Exception as e:
                X_raw = generate_synthetic_bscan()
                st.session_state["source_note"] = f"⚠️ Error: {str(e)}. Using synthetic B-scan."

    with st.spinner("Analyzing..."):
        result, _ = extract_tips(
            X_raw, 
            n_clusters=n_clusters,
            sep_config=sep_config,
            min_peak_prominence=min_peak_prominence,
            n_iterations=n_iterations,
            c_mm_per_us=c_mm_per_us,
            d_mm=d_mm,
            verbose=False,
        )

        corrected = verify_and_correct(
            result,
            X_raw,
            sep_config=sep_config,
            tolerance_samples=tolerance_samples,
            c_mm_per_us=c_mm_per_us,
            d_mm=d_mm,
            verbose=False,
            min_packet_rows=min_packet_rows,
            max_packet_rows=max_packet_rows,
        )

    st.session_state["result"] = corrected
    st.session_state["X_raw"] = X_raw

# --- Display Results ---
if "result" in st.session_state and "X_raw" in st.session_state:
    result = st.session_state["result"]
    X_raw = st.session_state["X_raw"]

    st.success(st.session_state.get("source_note", "Ready"))

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("A-scan index", result.ascan_index)
    with col2:
        st.metric("Top tip", f"{result.top_tip:.1f}" if result.top_tip else "—")
    with col3:
        st.metric("Bottom tip", f"{result.bottom_tip:.1f}" if result.bottom_tip else "—")
    with col4:
        st.metric("Height (mm)", f"{result.height_mm:.2f}" if result.height_mm else "—")

    st.divider()

    # Status
    if result.tips_merged:
        st.warning("⚠️ Tips appear merged (low resolution)")
    elif result.tips_reliable is False:
        st.info("ℹ️ Tips separated by fallback detection (close echoes)")
    else:
        st.success("✅ Two reliable tips detected")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Detection Plot", "🖼️ B-scan Overview", "⚙️ Parameters"])

    with tab1:
        fig, _ = plot_tip_detection_result(result)
        st.pyplot(fig)
        st.download_button(
            "📥 Download Plot",
            fig_to_png(fig),
            "tip_detection.png",
            "image/png"
        )
        plt.close(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(X_raw.T, aspect="auto", cmap="gray")
        if result.diffraction_window:
            ax.axhspan(*result.diffraction_window, color="green", alpha=0.1)
        ax.axvline(result.ascan_index, color="orange", lw=1.5, ls=":", label="Selected A-scan")
        if result.top_tip:
            ax.axhline(result.top_tip, color="lime", ls="--", lw=1.5, label="Top tip")
        if result.bottom_tip:
            ax.axhline(result.bottom_tip, color="magenta", ls="--", lw=1.5, label="Bottom tip")
        ax.legend()
        ax.set_xlabel("A-scan index")
        ax.set_ylabel("Depth sample")
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.json({
            "denoising": {"clusters": n_clusters, "iterations": n_iterations},
            "detection": {"min_prominence": min_peak_prominence},
            "separation": {
                "factor": max_separation_factor,
                "min_amplitude": min_relative_amp,
                "band_ratio": band_ratio,
                "max_separation": int(max_absolute_separation),
            },
        })

else:
    st.info("👈 Set parameters in the sidebar and click **Run Analysis**")
