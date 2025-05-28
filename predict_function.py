import pickle
import os

def preprocess_text(text):
    import re
    # Convert to string
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove numbers and special characters but keep letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def predict_news(text):
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'best_fake_news_model.pkl')
    
    # Load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Make prediction
    prediction = model.predict([processed_text])[0]
    probability = model.predict_proba([processed_text])[0]
    
    return {
        'prediction': 'Real' if prediction == 1 else 'Fake',
        'confidence': float(max(probability)),
        'processed_text': processed_text
    }
