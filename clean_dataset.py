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

# Read the combined dataset
print("Reading the combined dataset...")
df = pd.read_csv('combined_news_articles.csv')
print(f"Initial number of articles: {len(df)}")

# Rename columns to lowercase for consistency
df.columns = df.columns.str.lower()

# Remove rows where title contains "not found" or is empty/NaN
print("\nRemoving articles with missing or 'not found' titles...")
df = df[~df['title'].str.contains('not found', case=False, na=True)]
df = df.dropna(subset=['title'])
print(f"Number of articles after removing invalid titles: {len(df)}")

# Clean the text columns
print("\nCleaning text data...")
text_columns = ['title', 'story', 'date']  # Changed 'content' to 'story'
for column in text_columns:
    if column in df.columns:
        df[column] = df[column].apply(clean_text)

# Add label column (1 for real news)
df['label'] = 1

# Remove any remaining rows with empty content
df = df.dropna(subset=['story'])  # Changed 'content' to 'story'
print(f"Final number of articles after cleaning: {len(df)}")

# Remove specified columns
columns_to_remove = ['category', 'source', 'type', 'url']  # Added 'url', changed case
for col in columns_to_remove:
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f"Removed column: {col}")

# Rename 'story' to 'content' for consistency
df = df.rename(columns={'story': 'content'})

# Reorder columns to put label at the end
columns = [col for col in df.columns if col != 'label'] + ['label']
df = df[columns]

# Save the cleaned dataset
output_file = 'cleaned_news_articles.csv'
df.to_csv(output_file, index=False)
print(f"\nCleaned dataset saved to: {output_file}")

# Display some statistics
print("\nDataset Statistics:")
print("-" * 50)
print(f"Total number of articles: {len(df)}")
print("\nColumns in the dataset:")
for col in df.columns:
    non_null = df[col].count()
    print(f"{col}: {non_null} non-null values") 