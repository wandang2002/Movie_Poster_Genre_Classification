import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from DataPreprocessing.check_url import is_url_active
# ---------------------------
# Load & Preprocess Data
# ---------------------------
csv_path = "original_dataset\MovieGenre.csv"
df = pd.read_csv(csv_path, encoding="latin1")

# Keep only rows with valid Poster URLs and Genres
df = df.dropna(subset=["Poster", "Genre"])

valid_data = []
all_genres = set()

for _, row in df.iterrows():
    poster_url = row['Poster']
    genre_str = row['Genre']
    
    if is_url_active(poster_url):
        # Clean genres: Split and standardize formatting
        genres = [g.strip().title() for g in genre_str.split('|')]
        
        valid_data.append({
            'URL': poster_url, 
            'Genre': genres
        })
        all_genres.update(genres)
    else:
        print(f"Skipped inactive URL: {poster_url}")

# ---------------------------
# Binarize Genres
# ---------------------------
mlb = MultiLabelBinarizer()
binarized = mlb.fit_transform([item['Genre'] for item in valid_data])

# Create DataFrame with binary columns
binarized_df = pd.DataFrame(binarized, columns=mlb.classes_)

# ---------------------------
# Create Final DataFrame
# ---------------------------
final_df = pd.DataFrame({
    'Id' : range(1, len(valid_data) + 1),  # Add ID column
    'URL': [item['Id'] for item in valid_data],  # Poster URLs
    'Genre': [item['Genre'] for item in valid_data]  # Original genre lists
}).join(binarized_df)

# Save to CSV
final_df.to_csv("poster_genres_binarized.csv", index=False)