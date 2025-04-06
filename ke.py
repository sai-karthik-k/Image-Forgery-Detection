import gdown  # add 'gdown' to requirements.txt

# Download model from Google Drive (shared as public link)
url = 'https://drive.google.com/file/d/1frIR6crojKV86I_UtjIZmFinCCb2y6s_/view?usp=sharing'
output = 'model.keras'
gdown.download(url, output, quiet=False)
