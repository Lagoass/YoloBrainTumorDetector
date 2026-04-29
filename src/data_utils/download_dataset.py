import os
import urllib.request
import zipfile

# URLs obtidas diretamente da API do Figshare para o artigo 1512427
FILES = {
    "brainTumorDataPublic_1-766.zip": "https://ndownloader.figshare.com/files/3381290",
    "brainTumorDataPublic_767-1532.zip": "https://ndownloader.figshare.com/files/3381296",
    "brainTumorDataPublic_1533-2298.zip": "https://ndownloader.figshare.com/files/3381293",
    "brainTumorDataPublic_2299-3064.zip": "https://ndownloader.figshare.com/files/3381302",
    "cvind.mat": "https://ndownloader.figshare.com/files/7005344",
    "README.txt": "https://ndownloader.figshare.com/files/51340418"
}

def download_and_extract(raw_dir="data/raw"):
    os.makedirs(raw_dir, exist_ok=True)
    
    for filename, url in FILES.items():
        filepath = os.path.join(raw_dir, filename)
        
        # Download
        if not os.path.exists(filepath):
            print(f"Baixando {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Download de {filename} concluído!")
        else:
            print(f"{filename} já existe. Pulando download.")
            
        # Extração se for ZIP
        if filename.endswith(".zip"):
            print(f"Extraindo {filename}...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
            print(f"Extração de {filename} concluída!")

if __name__ == "__main__":
    download_and_extract()
