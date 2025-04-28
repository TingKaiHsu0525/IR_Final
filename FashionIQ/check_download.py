import os
import requests
from tqdm import tqdm

# Define your file path
file_path = 'fashion-iq-metadata/image_url/asin2url.toptee.txt'
# Define your image folder
image_folder = 'images'

# Create the image folder if it doesn't exist
os.makedirs(image_folder, exist_ok=True)

# Read the asin-url pairs
asin_url_dict = {}
with open(file_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            asin, url = parts
            asin_url_dict[asin] = url

# Define a fake browser headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Find missing images
missing_asins = []
for asin in asin_url_dict.keys():
    image_path = os.path.join(image_folder, f'{asin}.jpg')
    if not os.path.exists(image_path):
        missing_asins.append(asin)

print(f'Total missing images: {len(missing_asins)}')

# Function to download a single image
def download_image(asin, url):
    try:
        response = requests.get(url, timeout=10, headers=headers)
        if response.status_code == 200:
            with open(os.path.join(image_folder, f'{asin}.jpg'), 'wb') as f:
                f.write(response.content)
            return True
        else:
            return False
    except:
        return False

# Download missing images
# for asin in tqdm(missing_asins, desc="Downloading missing images"):
#     url = asin_url_dict[asin]
#     success = download_image(asin, url)
#     if not success:
#         print(f'Failed to download {asin}')

# print("Missing images download complete!")
