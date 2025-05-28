import pandas as pd
import os

# Define the file paths
files = {
    'awani/astroawani_articles.csv': 'Astro Awani',
    'awani/astroawani_articles_part2.csv': 'Astro Awani',
    'berita harian/beritaharian_articles.csv': 'Berita Harian',
    'berita harian/beritaharian_articles_part2.csv': 'Berita Harian',
    'buletin utama/buletin_utama.csv': 'Buletin Utama'
}

# List to store all dataframes
dfs = []

# Read each CSV file
for file_path, source in files.items():
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Add a source column if it doesn't exist
        if 'source' not in df.columns:
            df['source'] = source
            
        # Add the dataframe to our list
        dfs.append(df)
        print(f"Successfully read {file_path}")
        print(f"Number of articles: {len(df)}")
        
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")

# Combine all dataframes
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Remove any duplicate articles if they exist
    combined_df = combined_df.drop_duplicates()
    
    # Save the combined dataset
    output_file = 'combined_news_articles.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully combined all articles!")
    print(f"Total number of articles: {len(combined_df)}")
    print(f"Output saved to: {output_file}")
else:
    print("No data was read from the CSV files.") 