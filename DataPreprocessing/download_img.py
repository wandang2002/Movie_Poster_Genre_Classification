import requests
from PIL import Image
from io import BytesIO

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