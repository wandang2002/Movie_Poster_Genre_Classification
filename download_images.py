import os
import requests
import pandas as pd
from PIL import Image
from io import BytesIO

# Create Images directory if it doesn't exist
os.makedirs('Images', exist_ok=True)

# Load the CSV file
df = pd.read_csv('poster_genres_binarized_1.csv') 

def download_image(url, id):
    """Download and save image with error handling"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Open image and convert to RGB (to handle PNG transparency issues)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Save image with ID as filename
        img.save(f'Images/{id}.jpg')
        return True
        
    except Exception as e:
        print(f"Failed to download {url} (ID: {id}): {str(e)}")
        return False

# Process each row in the dataframe
for index, row in df.iterrows():
    # Get ID and URL from the row
    image_id = str(row['Id'])
    image_url = row['URL']
    
    # Skip if URL is missing or invalid
    if pd.isna(image_url) or not isinstance(image_url, str):
        continue
    
    # Check if image already exists
    if os.path.exists(f'Images/{image_id}.jpg'):
        print(f"Image {image_id} already exists, skipping...")
        continue
    
    # Download the image
    print(f"Downloading {image_id}...")
    success = download_image(image_url, image_id)
    
    if success:
        print(f"Successfully downloaded {image_id}")
    else:
        print(f"Failed to download {image_id}")

print("Download process completed!")