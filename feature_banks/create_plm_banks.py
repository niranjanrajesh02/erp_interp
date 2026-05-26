# This script extracts pixel, low-level, and mid-level features from the THINGS images and saves them as .npz files.

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

parser = argparse.ArgumentParser(description='Extract P,L,M features from THINGS images')
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


def get_pixel_features(img_gray: np.ndarray) -> np.ndarray:
    img_flat = img_gray.ravel()
    return img_flat.astype(np.float32)


def compute_gradients(img_gray: np.ndarray, sigma: float =1.0) -> tuple[np.ndarray, np.ndarray]:
    smoothed = filters.gaussian(img_gray, sigma=sigma)
    gx = filters.sobel_h(smoothed)
    gy = filters.sobel_v(smoothed)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.mod(np.arctan2(gy, gx), np.pi)
    return magnitude, orientation


def orientation_histogram(magnitude: np.ndarray, orientation: np.ndarray, n_bins: int =8) -> np.ndarray:
    bins = np.linspace(0, np.pi, n_bins + 1)
    hist, _ = np.histogram(orientation.ravel(), bins=bins, weights=magnitude.ravel())
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def get_low_level_features(
    img_gray: np.ndarray,
    sigma: float =1.0,
    n_bins: int =8,
    grid_size: tuple[int, int] = (4, 4),
    edge_percentile: int =75,
) -> np.ndarray:
    '''
    Extract low-level features from a grayscale image, including:
    - Global gradient statistics (mean, std, max, edge density)
    - Global orientation histogram
    - Local orientation histograms in a grid
    
    Parameters:
    - img_gray: 2D array of shape (H, W), grayscale image
    - sigma: float, standard deviation for Gaussian smoothing before gradient computation
    - n_bins: int, number of bins for orientation histograms
    - grid_size: tuple of ints (gh, gw), number of grid cells in height and width for local histograms
    - edge_percentile: int, percentile to threshold edges for edge density feature

    Returns:
    - features: 1D array of shape (num_features,), concatenated low-level features
    
    '''
    magnitude, orientation = compute_gradients(img_gray, sigma=sigma)

    feats = []

    thr = np.percentile(magnitude, edge_percentile)
    edge_mask = magnitude >= thr

    feats.extend([
        float(magnitude.mean()),
        float(magnitude.std()),
        float(magnitude.max()),
        float(edge_mask.mean()),
    ])

    global_hist = orientation_histogram(magnitude, orientation, n_bins=n_bins)
    feats.extend(global_hist.tolist())

    gh, gw = grid_size
    h, w = img_gray.shape
    cell_h = h // gh
    cell_w = w // gw

    for i in range(gh):
        for j in range(gw):
            r0, r1 = i * cell_h, (i + 1) * cell_h if i < gh - 1 else h
            c0, c1 = j * cell_w, (j + 1) * cell_w if j < gw - 1 else w

            mag_cell = magnitude[r0:r1, c0:c1]
            ori_cell = orientation[r0:r1, c0:c1]
            hist_cell = orientation_histogram(mag_cell, ori_cell, n_bins=n_bins)
            feats.extend(hist_cell.tolist())

    return np.array(feats, dtype=np.float32)



def get_midlevel_features(img: np.ndarray, device: torch.device) -> np.ndarray:
    '''
    Extract mid-level features using the Portilla-Simoncelli texture model implemented in the plenoptic library.

    Parameters:
    - img: 4D array of shape (N, C, H, W), input image batch (N=1 for single image)
    - device: torch.device, device to run the model on
    Returns:
    - features: 2D array of shape (N, feature_dim), extracted mid-level features for each image in the batch
    
    '''

    po.tools.set_seed(42)
    assert img.ndim == 4, "Input must be a batched images with shape (N, C, H, W)"
    img = img.to(device).to(torch.float32)
    model = po.simul.PortillaSimoncelli(
        image_shape = (224, 224),
        n_scales = 4,
        n_orientations = 4,
        spatial_corr_width=10,        
    ).to(device)
    features = model(img)
    #return one of the channels
    out = features[:, 0, :].cpu().numpy().astype(np.float32)
    return out





def main():
    args = parser.parse_args()


    # load img_from_metadata
    img_metadata = load_stim_metadata(args.stim_metadata_path)
    stim_ids = img_metadata["stim_ids"]
    stim_files = img_metadata["img_files"]


    pixel_features = []
    lowlevel_features = []
    midlevel_features = []

    device = torch.device(f'cuda:{args.device_id}') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    for stim_i, stim_file in enumerate(tqdm(stim_files)):
        img = load_img(stim_file, stim_dir=args.stim_dir, grayscale=True)
        pixel_features.append(get_pixel_features(img))

        lowlevel_features.append(get_low_level_features(img))

        img_3dims = np.stack([img, img, img], axis=-1)
        img_batched = np.expand_dims(img_3dims, axis=0).transpose(0, 3, 1, 2)  # (N, C, H, W)
        midlevel_features.append(get_midlevel_features(torch.tensor(img_batched, dtype=torch.float32), device=device))

    print("Extracted features for all stims, now applying PCA if needed...")



    pixel_features = np.array(pixel_features, dtype=np.float32)
    if pixel_features.shape[1] > args.max_feature_dim:
        print(f"Pixel features have {pixel_features.shape[1]} dimensions, applying PCA to reduce to {args.max_feature_dim}...")
        pca = PCA(n_components=args.max_feature_dim)
        pixel_features = pca.fit_transform(pixel_features)
    print("Pixel features shape:", pixel_features.shape)


    lowlevel_features = np.array(lowlevel_features, dtype=np.float32)
    if lowlevel_features.shape[1] > args.max_feature_dim:
        print(f"Low-level features have {lowlevel_features.shape[1]} dimensions, applying PCA to reduce to {args.max_feature_dim}...")
        pca = PCA(n_components=args.max_feature_dim)
        lowlevel_features = pca.fit_transform(lowlevel_features)
    print("Low-level features shape:", lowlevel_features.shape)

    midlevel_features = np.array(midlevel_features, dtype=np.float32)[:,0,:]
    # midlevel_features = np.array(midlevel_features, dtype=np.float32)
    if midlevel_features.shape[1] > args.max_feature_dim:
        print(f"Mid-level features have {midlevel_features.shape[1]} dimensions, applying PCA to reduce to {args.max_feature_dim}...")
        pca = PCA(n_components=args.max_feature_dim)
        midlevel_features = pca.fit_transform(midlevel_features)
    print("Mid-level features shape:", midlevel_features.shape)


    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, "pixel_features.npz"), features=pixel_features)
    np.savez(os.path.join(args.out_dir, "lowlevel_features.npz"), features=lowlevel_features)
    np.savez(os.path.join(args.out_dir, "midlevel_features.npz"), features=midlevel_features)
    print(f"Saved feature banks to {args.out_dir}!")


if __name__ == "__main__": main()

