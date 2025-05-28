from predict_function import predict_news
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

# Example news texts (you can replace these with your own)
news_texts = [
    "KUALA LUMPUR: Perdana Menteri mengumumkan pakej bantuan baharu untuk rakyat",
    "BREAKING: Alien mendarat di Putrajaya semalam malam!!!"
]

# Function to format the output nicely
def analyze_news(text):
    result = predict_news(text)
    print("\nAnalyzing news text:")
    print("-" * 50)
    print("Text:", text[:100] + "..." if len(text) > 100 else text)
    print("Prediction:", result['prediction'])
    print("Confidence: {:.2%}".format(result['confidence']))
    print("-" * 50)

# Analyze each news text
for text in news_texts:
    analyze_news(text)

# Interactive mode
print("\nWould you like to test your own news text? (Enter 'quit' to exit)")
while True:
    user_text = input("\nEnter news text to analyze: ")
    if user_text.lower() == 'quit':
        break
    analyze_news(user_text)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(random_state=42, n_jobs=-1),
    'Linear SVC': LinearSVC(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

pipeline = Pipeline([
    ('tfidf', tfidf),
    ('classifier', model)
]) 