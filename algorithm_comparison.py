import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

class AlgorithmComparison:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42),
            'Linear SVC': LinearSVC(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        self.results = {}
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
    def load_data(self, data_path):
        """Load and preprocess the dataset"""
        try:
            # Load the dataset
            print(f"Loading data from {data_path}...")
            df = pd.read_csv(data_path)
            
            # Print the column names to help with debugging
            print("Available columns:", df.columns.tolist())
            
            # Get text and label columns
            text_col = 'text' if 'text' in df.columns else 'content'  # Try alternative column name
            label_col = 'label' if 'label' in df.columns else 'class'  # Try alternative column name
            
            if text_col not in df.columns:
                raise ValueError(f"Could not find text column. Available columns: {df.columns.tolist()}")
            if label_col not in df.columns:
                raise ValueError(f"Could not find label column. Available columns: {df.columns.tolist()}")
            
            # Get features and target
            texts = df[text_col].fillna('')  # Handle any missing values
            labels = df[label_col]
            
            # Convert text to TF-IDF features
            print("Converting text to TF-IDF features...")
            X = self.vectorizer.fit_transform(texts)
            y = labels
            
            print(f"Dataset loaded successfully. Shape: {X.shape}")
            return X, y
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
        
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """Train and evaluate a single model"""
        # Train the model
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        print(f"Making predictions with {model_name}...")
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
        }
        
        # Perform cross-validation
        print(f"Performing cross-validation for {model_name}...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"{model_name} evaluation completed.")
        return metrics, conf_matrix
        
    def compare_algorithms(self, X, y):
        """Compare all algorithms and store results"""
        # Split the data
        print("\nSplitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            metrics, conf_matrix = self.evaluate_model(model, X_train, X_test, y_train, y_test, name)
            self.results[name] = {
                'metrics': metrics,
                'confusion_matrix': conf_matrix.tolist()
            }
            print(f"{name} Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            
    def plot_metrics_comparison(self):
        """Create comparison visualizations"""
        print("\nGenerating visualizations...")
        # Create directory for plots if it doesn't exist
        os.makedirs('comparison_results', exist_ok=True)
        
        # Prepare data for plotting
        metrics_df = pd.DataFrame({
            model_name: model_results['metrics'] 
            for model_name, model_results in self.results.items()
        }).T
        
        # Plot metrics comparison
        plt.figure(figsize=(12, 6))
        metrics_df[['accuracy', 'precision', 'recall', 'f1']].plot(kind='bar')
        plt.title('Model Performance Comparison')
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('comparison_results/metrics_comparison.png')
        plt.close()
        
        # Plot cross-validation results
        plt.figure(figsize=(10, 6))
        cv_means = [results['metrics']['cv_mean'] for results in self.results.values()]
        cv_stds = [results['metrics']['cv_std'] for results in self.results.values()]
        plt.errorbar(range(len(self.models)), cv_means, yerr=cv_stds, fmt='o')
        plt.xticks(range(len(self.models)), list(self.models.keys()), rotation=45)
        plt.title('Cross-validation Results')
        plt.xlabel('Models')
        plt.ylabel('CV Score')
        plt.tight_layout()
        plt.savefig('comparison_results/cv_results.png')
        plt.close()
        
        # Plot confusion matrices
        for name, results in self.results.items():
            plt.figure(figsize=(8, 6))
            sns.heatmap(results['confusion_matrix'], 
                       annot=True, 
                       fmt='d',
                       cmap='Blues',
                       xticklabels=['Fake', 'Real'],
                       yticklabels=['Fake', 'Real'])
            plt.title(f'Confusion Matrix - {name}')
            plt.tight_layout()
            plt.savefig(f'comparison_results/confusion_matrix_{name.lower().replace(" ", "_")}.png')
            plt.close()
        print("Visualizations saved in 'comparison_results' directory.")
            
    def save_results(self):
        """Save results to a JSON file"""
        print("\nSaving results...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_with_timestamp = {
            'timestamp': timestamp,
            'results': self.results
        }
        
        with open(f'comparison_results/results_{timestamp}.json', 'w') as f:
            json.dump(results_with_timestamp, f, indent=4)
            
        # Create a markdown report
        report = f"""# Algorithm Comparison Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Metrics

| Model | Accuracy | Precision | Recall | F1 Score | CV Mean ± Std |
|-------|----------|-----------|---------|-----------|---------------|
"""
        
        for name, results in self.results.items():
            metrics = results['metrics']
            report += f"| {name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | "
            report += f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | "
            report += f"{metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f} |\n"
            
        report += """
## Visualization Files
- metrics_comparison.png: Bar plot comparing all metrics across models
- cv_results.png: Cross-validation results with error bars
- confusion_matrix_{model_name}.png: Confusion matrices for each model

## Conclusions
- Best performing model based on accuracy: """
        
        # Find best model based on accuracy
        best_model = max(self.results.items(), 
                        key=lambda x: x[1]['metrics']['accuracy'])
        report += f"{best_model[0]} ({best_model[1]['metrics']['accuracy']:.4f})\n"
        
        with open(f'comparison_results/report_{timestamp}.md', 'w') as f:
            f.write(report)
        print(f"Results saved in 'comparison_results' directory with timestamp {timestamp}")

def main():
    # Initialize comparison
    comparison = AlgorithmComparison()
    
    try:
        # Load data
        print("Starting algorithm comparison...")
        X, y = comparison.load_data('combined_real_fake_news.csv')  # Updated file name
        
        # Run comparison
        comparison.compare_algorithms(X, y)
        
        # Generate visualizations
        comparison.plot_metrics_comparison()
        
        # Save results
        comparison.save_results()
        
        print("\nAlgorithm comparison completed successfully!")
        
    except Exception as e:
        print(f"\nError during algorithm comparison: {str(e)}")
        raise

if __name__ == "__main__":
    main() 