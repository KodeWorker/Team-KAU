import os 
import glob
import numpy as np
import nibabel as nib
from tqdm import trange
from PIL import Image
from preprocessing import normalization, stack_channels, pad_and_resize, resample
from argparse import ArgumentParser

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", help="source directory for *.nii.gz files.",
                        required=True, type=str)
    parser.add_argument("-t", "--target", help="target directory for pasrsed images and masks.", 
                        required=True, type=str)
    parser.add_argument("--image_size", help="image size", 
                        required=True, default=512, type=int)
    return parser

class DataPreparation(object):
    
    def __init__(self, data_dir, output_dir, image_size, preprocess=None, export_ext="npy"):
        
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.preprocess = preprocess
        self.export_ext = export_ext
        self.image_size = image_size
        
    def run(self, is_test=False):
    
        for fold in ["train", "validation"]:
            
            print("Data Preparation: {}".format(fold))
            
            if fold == "train":
                images = glob.glob(os.path.join(self.data_dir, fold) + "/image/*.nii.gz")
                labels = glob.glob(os.path.join(self.data_dir, fold) + "/label/*.nii.gz")
            elif fold == "validation":
                images = glob.glob(os.path.join(self.data_dir, fold) + "/*/image/*.nii.gz")
                labels = glob.glob(os.path.join(self.data_dir, fold) + "/*/label/*.nii.gz")
            
            output_dir = os.path.join(self.output_dir, fold)
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                    
            for n_files in trange(0, len(images)):
            
                image_path = images[n_files]
                label_path = labels[n_files]
                
                pname = os.path.basename(image_path).split(".")[0]
                
                epi_image = nib.load(image_path)
                epi_label = nib.load(label_path)
                
                epi_image_data = epi_image.get_fdata()
                epi_label_data = epi_label.get_fdata()
                
                pixdims = (epi_image.header["pixdim"], epi_label.header["pixdim"])
                
                if self.preprocess:
                    if type(self.preprocess) == type(list()):
                        for preprocess_ in self.preprocess:
                            if preprocess_ == pad_and_resize:
                                epi_image_data, epi_label_data = preprocess_(epi_image_data, epi_label_data, image_size=self.image_size)
                            elif preprocess_ == resample:
                                epi_image_data, epi_label_data = preprocess_(epi_image_data, epi_label_data, pixdims = pixdims)
                            else:
                                epi_image_data, epi_label_data = preprocess_(epi_image_data, epi_label_data)
                    else:
                        epi_image_data, epi_label_data = self.preprocess(epi_image_data, epi_label_data)
                
                imgs_folder = os.path.join(output_dir, "imgs")
                masks_folder = os.path.join(output_dir, "masks")
                if not os.path.exists(imgs_folder):
                    os.makedirs(imgs_folder)
                if not os.path.exists(masks_folder):
                    os.makedirs(masks_folder)
                
                for n_slices in range(epi_image.shape[-1]):
                    
                    image_filename = "{}_{}.{}".format(pname, n_slices+1, self.export_ext)
                    mask_filename = "{}_{}.{}".format(pname, n_slices+1, self.export_ext)
                    
                    if self.export_ext == "npy":
                        np.save(os.path.join(imgs_folder, image_filename), epi_image_data[...,n_slices])
                        np.save(os.path.join(masks_folder, mask_filename), epi_label_data[...,n_slices])
                    elif self.export_ext == "png":
                        pil_image = Image.fromarray(epi_image_data[...,n_slices].transpose(1, 0, 2))
                        pil_image.save(os.path.join(imgs_folder, image_filename))
                        pil_mask = Image.fromarray(epi_label_data[...,n_slices].transpose(1, 0))
                        pil_mask.save(os.path.join(masks_folder, mask_filename))

                    if is_test: break
                if is_test: break
            if is_test: break

if __name__ == "__main__":
    args = build_argparser().parse_args()

    data_dir = args.source
    output_dir = args.target
    image_size = args.image_size
    
    DataPreparation(data_dir, output_dir, image_size, export_ext="png", preprocess=[resample, stack_channels, normalization, pad_and_resize]).run()
    
    