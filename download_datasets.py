import requests
import shutil
from tqdm.auto import tqdm
import os
# Download the datasets used from the following link by invoking this Python file.
dataset_url = "https://www.dropbox.com/scl/fi/241a0ehp23lqodozgrxft/meta_data.zip?rlkey=mk7u0tp43sw707a4o1h7a6lj5&dl=1"


def download_file(url,dest_dir="./"):
    file_path = f"{dest_dir}/temp.zip"
    with requests.get(url, stream=True) as r:
        #get length
        total_length = int(r.headers.get("Content-Length"))

        #progress bar via tqdm
        with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as file:
            #save the file
            with open(file_path, 'wb') as output:
                shutil.copyfileobj(file, output)

    #unzip the file
    shutil.unpack_archive(file_path, dest_dir)
    #remove the zip file since not needed
    if os.path.exists(file_path):
        os.remove(file_path)

    print(f"If download not successful, use {url} to download directly from dropbox...")

def run():
    download_file(dataset_url)

if __name__ == "__main__":
    run()
