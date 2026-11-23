# app.py
import streamlit as st
import pandas as pd
import os
import joblib
import plotly.express as px

st.set_page_config(page_title="Klasifikasi Spesies Penguin", page_icon="🐧", layout="wide")

@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'penguin_pipeline.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

artifact = load_model()

# Header Utama
st.title("Prediksi Spesies Penguin Antarktika")
st.markdown("Aplikasi *Machine Learning* interaktif untuk mengklasifikasikan spesies penguin (*Adelie*, *Chinstrap*, *Gentoo*) berdasarkan ciri fisiknya.")

if artifact is None:
    st.error("Model belum ditemukan! Harap jalankan skrip `!python latih.py` terlebih dahulu.")
else:
    pipeline = artifact['pipeline']
    metrics = artifact['metrics']

    # Sidebar untuk Input Parameter
    st.sidebar.header("Panel Kontrol Karakteristik")
    island = st.sidebar.selectbox("Pulau Pengamatan (Island)", ["Biscoe", "Dream", "Torgersen"])
    sex = st.sidebar.sidebar.selectbox("Jenis Kelamin (Sex)", ["male", "female"]) if "sex" in st.sidebar.__dir__() else st.sidebar.selectbox("Jenis Kelamin (Sex)", ["male", "female"])

    bill_length_mm = st.sidebar.slider("Panjang Paruh / Bill Length (mm)", 30.0, 60.0, 40.0)
    bill_depth_mm = st.sidebar.slider("Kedalaman Paruh / Bill Depth (mm)", 13.0, 22.0, 17.0)
    flipper_length_mm = st.sidebar.slider("Panjang Sirip / Flipper Length (mm)", 170.0, 240.0, 200.0)
    body_mass_g = st.sidebar.slider("Massa Tubuh / Body Mass (gram)", 2700.0, 6300.0, 4000.0)

    # Menyiapkan data input
    input_data = pd.DataFrame([{
        'island': island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': sex,
        'year': 2008
    }])

    # Layout Utama Menjadi 2 Kolom
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Ringkasan Parameter Input")
        st.dataframe(input_data.drop(columns=['year']), use_container_width=True)

        predict_btn = st.button("🔍 Jalankan Prediksi Spesies", type="primary", use_container_width=True)

    with col2:
        st.subheader("Hasil & Analisis Probabilitas")
        if predict_btn:
            prediction = pipeline.predict(input_data)[0]
            probabilities = pipeline.predict_proba(input_data)[0]

            st.success(f"### Spesies Terprediksi: **{prediction}** 🐧")

            # Membuat Bar Chart interaktif dengan Plotly yang memiliki animasi transisi
            prob_df = pd.DataFrame({
                'Spesies': pipeline.classes_,
                'Probabilitas': probabilities * 100
            })

            fig = px.bar(
                prob_df,
                x='Spesies',
                y='Probabilitas',
                text_auto='.1f',
                color='Spesies',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title="Tingkat Keyakinan Model (%)"
            )
            fig.update_layout(yaxis_title="Probabilitas (%)", showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Silakan atur parameter di sidebar lalu klik tombol **Jalankan Prediksi Spesies**.")

    st.markdown("---")
    st.info(f"**Informasi Model**: Menggunakan algoritma **{metrics['best_model']}** dengan akurasi pengujian data test sebesar **{metrics['test_accuracy']*100:.2f}%**.")
