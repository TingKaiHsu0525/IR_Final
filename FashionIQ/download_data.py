import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Define your file path
file_path = 'fashion-iq-metadata/image_url/asin2url.toptee.txt'
# Define your output folder
output_folder = 'images'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the URL list file
with open(file_path, 'r') as f:
    lines = f.readlines()

# Parse the asin and url pairs
asin_url_list = []
for line in lines:
    parts = line.strip().split('\t')
    if len(parts) == 2:
        asin_url_list.append((parts[0], parts[1]))

# Define a function to download a single image
def download_image(asin, url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img_path = os.path.join(output_folder, f'{asin}.jpg')
            with open(img_path, 'wb') as img_file:
                img_file.write(response.content)
            return True
        else:
            return False
    except:
        return False

# Set the number of threads (you can adjust based on your machine)
num_threads = 16

# Use ThreadPoolExecutor for multithreaded downloading
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Submit all download tasks
    futures = [executor.submit(download_image, asin, url) for asin, url in asin_url_list]
    
    # Use tqdm to display the progress bar
    for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
        pass

print("All downloads finished!")
