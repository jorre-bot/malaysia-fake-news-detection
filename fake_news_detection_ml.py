import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
import pickle
import re
from nltk.tokenize import word_tokenize
import nltk
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
print("Downloading required NLTK data...")
nltk.download('punkt')

# Function to clean and preprocess text
def preprocess_text(text):
    if pd.isna(text):
        return ""
    
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

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('combined_real_fake_news.csv')
print(f"Dataset shape: {df.shape}")

# Combine title and content for better feature representation
print("Preprocessing text data...")
df['text'] = df['title'] + " " + df['content']
df['text'] = df['text'].apply(preprocess_text)

# Split features and target
X = df['text']
y = df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(random_state=42, n_jobs=-1),
    'Linear SVC': LinearSVC(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
print("\nTraining and evaluating models...")
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('classifier', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'pipeline': pipeline,
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(results[name]['classification_report'])

# Find the best model
best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
best_accuracy = results[best_model_name]['accuracy']
print(f"\nBest performing model: {best_model_name}")
print(f"Best accuracy: {best_accuracy:.4f}")

# Save the best model
print(f"\nSaving the best model ({best_model_name})...")
best_pipeline = results[best_model_name]['pipeline']
with open('best_fake_news_model.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)

# Create confusion matrix plot for the best model
plt.figure(figsize=(8, 6))
cm = results[best_model_name]['confusion_matrix']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()

# Function to predict on new text
def predict_news(text, model):
    # Preprocess the text
    processed_text = preprocess_text(text)
    # Make prediction
    prediction = model.predict([processed_text])[0]
    probability = model.predict_proba([processed_text])[0]
    return prediction, probability

# Example usage
print("\nExample prediction:")
example_text = df['text'].iloc[0]
prediction, probability = predict_news(example_text, best_pipeline)
print("Text:", example_text[:200] + "...")
print("Prediction:", "Real" if prediction == 1 else "Fake")
print("Confidence:", max(probability))

# Save example function for later use
with open('predict_function.py', 'w') as f:
    f.write("""import pickle

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
    # Load the model
    with open('best_fake_news_model.pkl', 'rb') as f:
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
""") 