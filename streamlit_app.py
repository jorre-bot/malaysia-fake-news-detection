import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from predict_function import predict_news
import pickle
import os
from datetime import datetime
import pandas as pd

# Initialize session state
if 'login_status' not in st.session_state:
    st.session_state['login_status'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(columns=['Date', 'Text', 'Prediction', 'Confidence'])

# Set page config
st.set_page_config(
    page_title="Pengesan Berita Palsu Malaysia",
    page_icon="🔍",
    layout="centered"
)

# Load configuration file
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Create authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
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

# Authentication
if not st.session_state['login_status']:
    # Create tabs for login and registration
    tab1, tab2 = st.tabs(["Log Masuk", "Daftar"])
    
    with tab1:
        name, authentication_status, username = authenticator.login("Log Masuk", "main")
        if authentication_status:
            st.session_state['login_status'] = True
            st.session_state['username'] = username
            st.rerun()
        elif authentication_status == False:
            st.error('Username/password is incorrect')
        elif authentication_status == None:
            st.warning('Please enter your username and password')

    with tab2:
        try:
            if authenticator.register_user('Register user', preauthorization=False):
                st.success('User registered successfully')
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(e)

else:
    # Show logout button in sidebar
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.title(f'Selamat Datang, {st.session_state["username"]}!')

    # Main app
    st.title("Pengesan Berita Palsu 🔍")

    # Tabs for Analysis and History
    tab1, tab2 = st.tabs(["Analisis", "Sejarah"])

    with tab1:
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
                        
                        # Add to history
                        new_record = pd.DataFrame({
                            'Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                            'Text': [news_text],
                            'Prediction': [result['prediction']],
                            'Confidence': [f"{result['confidence']*100:.2f}%"]
                        })
                        st.session_state['history'] = pd.concat([new_record, st.session_state['history']]).reset_index(drop=True)
                        
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
            else:
                st.warning("Sila masukkan teks berita untuk dianalisis.")

        if col2.button("Kosongkan", use_container_width=True):
            # This will trigger a rerun with empty text
            st.experimental_rerun()

    with tab2:
        if not st.session_state['history'].empty:
            st.dataframe(st.session_state['history'], use_container_width=True)
        else:
            st.info("Tiada sejarah analisis setakat ini.")

    # Footer
    st.markdown("---")
    st.markdown("© 2024 Pengesan Berita Palsu Malaysia. Semua hak cipta terpelihara.") 