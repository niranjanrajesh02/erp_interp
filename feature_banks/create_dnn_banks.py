# This script extracts dnn features from the THINGS images and saves them as .npz files.

import os
import re
import torch
import argparse
import numpy as np
from tqdm import tqdm
import plenoptic as po
from sklearn.decomposition import PCA
from utilz.preprocessing_utilz import load_stim_metadata
from skimage import io, color, transform, filters
from utilz.dnns import load_model, make_things_dataloader, extract_model_activations


parser = argparse.ArgumentParser(description='Extract DNN features from THINGS images')
parser.add_argument('--stim-metadata-path', type=str, required=True, help='Path to the stim metadata file')
parser.add_argument('--stim-dir', type=str,  help='Directory containing the THINGS images')
parser.add_argument('--out-dir', type=str, required=True, help='Directory to save the extracted feature banks')
parser.add_argument('--max-feature-dim', type=int, default=100, help='Maximum dimensionality for each feature bank after PCA')
parser.add_argument('--device-id', type=int, default=0, help='GPU device ID to use for mid-level feature extraction')


def load_img(img_path: str, stim_dir: str, crop_size: int =224, grayscale: bool =True) -> np.ndarray:
    # object name is the first part of the img path before the number
    object_name = re.split(r'_\d+', img_path)[0]
    if object_name.endswith('_'):
        object_name = object_name[:-1]

    stim_path = os.path.join(stim_dir, object_name, img_path)
    img = io.imread(stim_path)
    if img.ndim == 2:
        img_gray = img.astype(np.float32)
        if img_gray.max() > 1:
            img_gray = img_gray / 255.0
        return img_gray
    
    img = img[..., :3]
    if img.max() > 1:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)

    if grayscale:
        img = color.rgb2gray(img)

    img = transform.resize(img, (crop_size, crop_size), anti_aliasing=True)
    return img.astype(np.float32)



def main():
    args = parser.parse_args()


    # load img_from_metadata
    img_metadata = load_stim_metadata(args.stim_metadata_path)
    stim_ids = img_metadata["stim_ids"]
    stim_files = img_metadata["img_files"]
    device = torch.device(f'cuda:{args.device_id}') if torch.cuda.is_available() else torch.device('cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    dnns_to_extract = ['vgg16', 'clip', 'dino']
    regions_to_extract = ['early', 'mid', 'late']

    for dnn in dnns_to_extract:
        print(f"Loading model {dnn}...")
        model, preprocess = load_model(dnn)

        for region in regions_to_extract:
            print(f"Extracting {region}-level features from {dnn}...")
            dl = make_things_dataloader(args.stim_dir, stim_files, preprocess, batch_size=64, shuffle=False, num_workers=4)
            activations = extract_model_activations(model, dnn, region, dl, device)
            extracted_layers = list(activations.keys())
            print(f"Extracted activations from layers: {extracted_layers}")
            
            region_acts = []

            for ln in extracted_layers:
                acts_arr = activations[ln]
                region_acts.append(acts_arr)
            region_acts = np.concatenate(region_acts, axis=1)
            print(f"Concatenated activations shape for {dnn} {region}: {region_acts.shape}")

            # PCA dimensionality reduction
            pca = PCA(n_components=args.max_feature_dim)
            region_acts_reduced = pca.fit_transform(region_acts)
            print(f"Reduced features shape for {dnn} {region}: {region_acts_reduced.shape}")

            # save features
            out_path = os.path.join(args.out_dir, f"{dnn}_{region}_features.npz")
            np.savez_compressed(out_path, features=region_acts_reduced, layer_names=extracted_layers)
            print(f"Saved {dnn} {region}-level features to {out_path}")

            

            

    


if __name__ == "__main__": main()

