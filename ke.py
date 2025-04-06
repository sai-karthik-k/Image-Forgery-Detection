import gdown  # Add 'gdown' to requirements.txt

# Correct Google Drive link for gdown
url = 'https://drive.google.com/uc?id=1frIR6crojKV86I_UtjIZmFinCCb2y6s_'
output = 'model.keras'

gdown.download(url, output, quiet=False)
