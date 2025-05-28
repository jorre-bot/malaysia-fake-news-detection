import streamlit as st
from predict_function import predict_news
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Pengesan Berita Palsu Malaysia",
    page_icon="🔍",
    layout="centered"
)

# Debug information
st.sidebar.write("Debug Information:")
st.sidebar.write(f"Current directory: {os.getcwd()}")
st.sidebar.write(f"Files in directory: {os.listdir()}")

try:
    # Check if model file exists
    model_path = 'best_fake_news_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
except Exception as e:
    st.error(f"Error checking model file: {str(e)}")

# Add custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .prediction-text {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .real {
        background-color: #d4edda;
        color: #155724;
    }
    .fake {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Pengesan Berita Palsu 🔍")

# Text input
news_text = st.text_area(
    "Masukkan Teks Berita:",
    height=200,
    placeholder="Taipkan atau tampal teks berita di sini..."
)

# Create two columns for buttons
col1, col2 = st.columns(2)

# Add buttons
if col1.button("Analisis Berita", use_container_width=True):
    if news_text.strip():
        with st.spinner('Sedang menganalisis...'):
            try:
                # Make prediction
                result = predict_news(news_text)
                
                # Display results
                st.markdown("### Keputusan Analisis:")
                
                # Create result container
                result_class = "real" if result['prediction'] == "Real" else "fake"
                st.markdown(f"""
                <div class="prediction-text {result_class}">
                    Ramalan: {result['prediction']}<br>
                    Tahap Keyakinan: {result['confidence']*100:.2f}%
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error analyzing news: {str(e)}")
                st.error("Stack trace:", exc_info=True)
    else:
        st.warning("Sila masukkan teks berita untuk dianalisis.")

if col2.button("Kosongkan", use_container_width=True):
    # This will trigger a rerun with empty text
    st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("© 2024 Pengesan Berita Palsu Malaysia. Semua hak cipta terpelihara.") 