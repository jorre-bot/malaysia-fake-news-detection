# Pengesan Berita Palsu Malaysia 🔍

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://malaysia-fake-news-detection.streamlit.app)

A machine learning-powered fake news detection system for Malaysian news articles, built as part of a Final Year Project.

## Features

- 🔒 Secure user authentication system
- 📊 Real-time news article analysis
- 🤖 Machine learning-based prediction
- 📱 Simple and intuitive user interface
- 🇲🇾 Support for Malay language text
- 📜 User history tracking

## Live Demo

Try the application here: [Pengesan Berita Palsu Malaysia](https://malaysia-fake-news-detection.streamlit.app)

## Technology Stack

- Python 3.9+
- Streamlit for web interface
- Scikit-learn for machine learning
- NLTK for text processing
- Streamlit-Authenticator for user management

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/jorre-bot/malaysia-fake-news-detection.git
cd malaysia-fake-news-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

## Project Structure

```
├── streamlit_app.py        # Main Streamlit application
├── predict_function.py     # Prediction model functions
├── config.yaml            # Authentication configuration
├── requirements.txt       # Project dependencies
└── best_fake_news_model.pkl  # Trained ML model
```

## How It Works

1. Users register/login to access the system
2. Enter news text for analysis
3. The system processes the text using NLP techniques
4. Machine learning model predicts if the news is real or fake
5. Results are displayed with confidence scores
6. Prediction history is saved for each user

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit
- Powered by scikit-learn
- Part of Final Year Project at [Your University] 