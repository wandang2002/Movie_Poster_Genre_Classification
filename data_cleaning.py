import pandas as pd
import requests
import time
from sklearn.preprocessing import MultiLabelBinarizer

# ---------------------------
# 1. Load & Preprocess Data
# ---------------------------
csv_path = "MovieGenre.csv"
df = pd.read_csv(csv_path, encoding="latin1")

# Keep only rows with valid Poster URLs and Genres
df = df.dropna(subset=["Poster", "Genre"])

# ---------------------------
# 2. URL Validation Function
# ---------------------------
def is_url_active(url, max_retries=3):
    """Check if a URL is accessible (status code 200)"""
    for _ in range(max_retries):
        try:
            response = requests.head(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException as e:
            print(f"URL check failed for {url}: {e}")
            time.sleep(1)
    return False

# ---------------------------
# 3. Process Valid Entries
# ---------------------------
valid_data = []
all_genres = set()

for _, row in df.iterrows():
    poster_url = row['Poster']
    genre_str = row['Genre']
    
    if is_url_active(poster_url):
        # Clean genres: Split and standardize formatting
        genres = [g.strip().title() for g in genre_str.split('|')]
        
        valid_data.append({
            'Id': poster_url, 
            'Genre': genres
        })
        all_genres.update(genres)
    else:
        print(f"Skipped inactive URL: {poster_url}")

# ---------------------------
# 4. Binarize Genres
# ---------------------------
mlb = MultiLabelBinarizer()
binarized = mlb.fit_transform([item['Genre'] for item in valid_data])

# Create DataFrame with binary columns
binarized_df = pd.DataFrame(binarized, columns=mlb.classes_)

# ---------------------------
# 5. Create Final DataFrame
# ---------------------------
final_df = pd.DataFrame({
    'Id': [item['Id'] for item in valid_data],  # Poster URLs
    'Genre': [item['Genre'] for item in valid_data]  # Original genre lists
}).join(binarized_df)

# Save to CSV
final_df.to_csv("poster_genres_binarized.csv", index=False)