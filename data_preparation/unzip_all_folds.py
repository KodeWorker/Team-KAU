import os
import glob
import zipfile

if __name__ == "__main__":

    data_dir = r"D:\Datasets\Brain Tumor Segmentation Challenge"
    extract_dir = os.path.join(data_dir, "data")
    
    filepaths = glob.glob(data_dir + "\*.zip")
    
    for filepath in filepaths:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)