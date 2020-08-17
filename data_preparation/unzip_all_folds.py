import os
import glob
import zipfile
from tqdm import tqdm
from argparse import ArgumentParser

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", help="source *.zip files.",
                        required=True, type=str)
    parser.add_argument("-t", "--target", help="target folder for *.nii.gz.", 
                        required=True, type=str)
    return parser
    
if __name__ == "__main__":
    args = build_argparser().parse_args()
    
    data_dir = args.source
    extract_dir = args.target
    
    filepaths = glob.glob(data_dir + "/*.zip")
    
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    
    for filepath in tqdm(filepaths):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)