import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="TOFD Analysis", layout="wide")

st.title("TOFD Analysis Platform")
st.markdown("Analyse TOFD : conversion B-scan → A-scan, débruitage, détection et visualisation.")

# --- PAGE SELECTION ---
choice = st.radio(
    "Que veux-tu faire ?",
    ["Conversion B-scan → A-scan", "Débruitage", "Détection défauts", "Animation A-scan"]
)

if choice == "Conversion B-scan → A-scan":
    st.header("Conversion B-scan → A-scan")

    uploaded_file = st.file_uploader("Télécharger un B-scan", type=["png", "jpg", "jpeg", "bmp", "npy"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".npy"):
            bscan = np.load(uploaded_file)
        else:
            image = Image.open(uploaded_file).convert("L")
            bscan = np.array(image)

        st.image(bscan, caption="B-scan chargé", use_column_width=True)
        st.write(f"Dimensions : {bscan.shape}")

        if st.button("Convertir en A-scans"):
            if bscan.ndim == 2:
                ascans = bscan.astype(np.float32)
                if ascans.max() > 1:
                    ascans = ascans / 255.0
                ascans = ascans.T

                st.success("Conversion terminée.")
                st.write(f"Nombre d'A-scans : {ascans.shape[0]}")
                st.write(f"Longueur d'un A-scan : {ascans.shape[1]}")
                np.save("ascans.npy", ascans)

                with open("ascans.npy", "rb") as f:
                    st.download_button("Télécharger les A-scans", f, file_name="ascans.npy")

elif choice == "Débruitage":
    st.header("Débruitage")
    st.info("Cette partie sera reliée à ton code de débruitage TOFD.")

    uploaded_file = st.file_uploader("Télécharger un fichier TOFD", type=["png", "jpg", "jpeg", "npy"])

    if uploaded_file is not None:
        st.write("Fichier chargé :", uploaded_file.name)
        if st.button("Lancer le débruitage"):
            st.success("Débruitage lancé. À relier à ta fonction Python.")

elif choice == "Détection défauts":
    st.header("Détection des défauts")
    st.info("Cette partie sera reliée à ton pipeline de détection de tips.")

    uploaded_file = st.file_uploader("Télécharger un B-scan", type=["png", "jpg", "jpeg", "npy"])

    if uploaded_file is not None:
        st.write("Fichier chargé :", uploaded_file.name)
        if st.button("Lancer la détection"):
            st.success("Détection lancé. À relier à ton script TOFD.")

elif choice == "Animation A-scan":
    st.header("Animation des A-scans")
    st.info("Cette partie va permettre d'animer les profils A-scan.")

    uploaded_file = st.file_uploader("Télécharger un B-scan", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        bscan = np.array(image)
        st.image(bscan, caption="B-scan", use_column_width=True)

        if st.button("Démarrer l'animation"):
            st.success("Animation activée. À relier à ton script d'animation.")
