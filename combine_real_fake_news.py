import pandas as pd
import numpy as np

def clean_text(text):
    if pd.isna(text):
        return ""
    # Convert to string if it's not already
    text = str(text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Read the cleaned real news dataset
print("Reading the cleaned real news dataset...")
real_news_df = pd.read_csv('cleaned_news_articles.csv')
print(f"Number of real news articles: {len(real_news_df)}")

# Read the fake news dataset
print("\nReading the fake news dataset...")
fake_news_df = pd.read_csv('fake_news_1964.csv')
print(f"Number of fake news articles: {len(fake_news_df)}")

# Clean and prepare fake news dataset
print("\nCleaning fake news dataset...")

# Convert column names to lowercase
fake_news_df.columns = fake_news_df.columns.str.lower()

# Clean text in fake news dataset
text_columns = ['title', 'content', 'date']
for column in text_columns:
    if column in fake_news_df.columns:
        fake_news_df[column] = fake_news_df[column].apply(clean_text)

# Add label for fake news (0)
fake_news_df['label'] = 0

# Keep only necessary columns from fake news dataset
keep_columns = ['title', 'content', 'date', 'label']
fake_news_df = fake_news_df[keep_columns]

# Remove any rows with empty titles or content
fake_news_df = fake_news_df.dropna(subset=['title', 'content'])
print(f"Number of fake news articles after cleaning: {len(fake_news_df)}")

# Combine real and fake news
print("\nCombining datasets...")
combined_df = pd.concat([real_news_df, fake_news_df], ignore_index=True)

# Shuffle the dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the combined dataset
output_file = 'combined_real_fake_news.csv'
combined_df.to_csv(output_file, index=False)

# Display statistics
print("\nDataset Statistics:")
print("-" * 50)
print(f"Total number of articles: {len(combined_df)}")
print(f"Number of real news (label 1): {len(combined_df[combined_df['label'] == 1])}")
print(f"Number of fake news (label 0): {len(combined_df[combined_df['label'] == 0])}")
print("\nColumns in the dataset:")
for col in combined_df.columns:
    non_null = combined_df[col].count()
    print(f"{col}: {non_null} non-null values")

# Display sample of the data
print("\nSample of combined dataset (first 5 rows):")
print(combined_df[['title', 'label']].head()) 