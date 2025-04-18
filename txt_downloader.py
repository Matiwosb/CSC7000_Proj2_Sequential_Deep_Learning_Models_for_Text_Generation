import requests
import os

api_url = "https://api.github.com/repos/jghawaly/CSC7809_FoundationModels/contents/Project2/data/raw"
headers = {"Accept": "application/vnd.github.v3+json"}

response = requests.get(api_url, headers=headers)
response.raise_for_status()

files = response.json()

if not os.path.exists("raw_data"):
    os.makedirs("raw_data")

for file in files:
    if file['name'].endswith('.txt'):
        raw_url = file['download_url']
        print(f"Downloading: {file['name']}")
        file_response = requests.get(raw_url)
        file_response.raise_for_status()
        with open(os.path.join("raw_data", file['name']), 'wb') as f:
            f.write(file_response.content)

print("All .txt files downloaded to the 'raw_data' directory.")
