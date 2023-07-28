import wget, os, wfdb

import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

base_url = "https://archive.physionet.org/physiobank/database/mitdb/"  # replace this with your actual base url
dir_path = 'original_dataset/BIH/'  # replace this with your actual directory path


def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def download_mit_bih_dataset(base_url, dir_path):
    page = requests.get(base_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    links = soup.find_all('a')

    for link in links:
        file_url = link.get('href')
        if file_url and (file_url.endswith('.dat') or file_url.endswith('.atr') or file_url.endswith('.hea')):  # adjust the condition based on the files you need
            full_url = base_url + file_url if 'http' not in file_url else file_url
            file_name = os.path.join(dir_path, os.path.basename(file_url))
            download_file(full_url, file_name)

def BIH_preprocess():
    
    record = wfdb.rdrecord('original_dataset/BIH/100', sampto=None)
    annotation = wfdb.rdann('original_dataset/BIH/100', 'atr', sampto=None)

    print(record.p_signal)
    print(annotation.sample)
    print(annotation.symbol)

    wfdb.plot_wfdb(record=record, annotation=annotation,
               title='Record 100 from MIT-BIH Arrhythmia Database',
               time_units='seconds', figsize=(10, 4))
    plt.savefig('emg_100.png')
    



if __name__ == '__main__':
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # download_mit_bih_dataset(base_url, dir_path)
    BIH_preprocess()