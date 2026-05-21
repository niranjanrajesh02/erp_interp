import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilz.preprocessing_utilz import load_eeg_preprocessed, average_across_reps, extract_erp_features, save_erp_features, get_stim_img_and_category, save_stim_metadata



parser = argparse.ArgumentParser(description='Preprocess ERP component data')

# things_metadata_dir = './THINGS_metadata'
# preprocessed_dir = './processed'

parser.add_argument('--things-metadata-dir', type=str, help='Path to the THINGS metadata directory')
parser.add_argument('--preprocessed-dir', type=str, help='Path to the preprocessed data directory')
parser.add_argument('--output-dir', type=str, default='./erp_processed', help='Path to save the ERP features and metadata')
parser.add_argument('--config', type=int, default=2, help='Configuration number for preprocessing (e.g., 1, 2, etc.)')

ERP_component_windows = {
    "C1": {"spatial": [ 'Pz', 'POz', 'Oz'],  # posterior midline
           "temporal": (40, 100)},  
    "P1": {"spatial": ['PO7', 'PO3', 'O1', 'PO4', 'PO8', 'O2'],  # lateral occipital
           "temporal": (60, 130)},  
    "N1": {"spatial": ['PO7', 'PO3', 'POz', 'PO4', 'PO8'],  # parietal-occipital
           "temporal": (100, 200)},
    "N170": {"spatial": ['TP7', 'P7', 'PO7', 'PO8', 'P8', 'TP8'],  # inferior occipito-temporal
             "temporal": (110, 200)},
    "EPN": {"spatial":  ["PO7", "PO3", "PO4", "PO8"],               # lateral occipito-parietal
            "temporal": (200, 300)},
    "P2": {"spatial": ['CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'], # centro-parietal
           "temporal": (150, 250)},
    "N2": {"spatial": ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6'], # fronto-central
           "temporal": (200, 350)},
    "P3b": {"spatial": ['CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'], # centro-parietal
           "temporal": (300, 600)},
    "LPC": {"spatial": ['Pz', 'POz', 'P3', 'P4'], # parietal
           "temporal": (400, 800)},
    "N400": {"spatial": ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6'], # fronto-central
             "temporal": (300, 500)},
}


def main():
    args = parser.parse_args()

    concepts_df = pd.read_csv(os.path.join(args.things_metadata_dir, 'concepts-metadata_things.tsv'), sep='\t') # for category avging when needed
    image_metadata = np.load(os.path.join(args.things_metadata_dir, 'image_metadata.npy'), allow_pickle=True).item() # keys: train_img_concepts, train_img_concepts_THINGS, train_img_files (same for test)
    print(f"Found {len(image_metadata['train_img_files'])} train images and {len(image_metadata['test_img_files'])} test images in metadata.")



    subject = 1
    config = args.config

    for subject in range(1, 6):
        for split in ['train', 'test']:
            print(f"\nProcessing subject {subject}, split {split}...")

            eeg_data = load_eeg_preprocessed(args.preprocessed_dir, subject, config, split)
            stim_erps, stim_ids = average_across_reps(eeg_data)
            stim_img_files, stim_categories = get_stim_img_and_category(stim_ids, split, image_metadata, concepts_df)
            save_stim_metadata(stim_ids, stim_img_files, stim_categories, subject, split, output_dir=f'{args.output_dir}/config{config}')

            erp_features = extract_erp_features(stim_erps, eeg_data['times'], eeg_data['ch_names'], component_windows=ERP_component_windows)
            save_erp_features(erp_features, stim_ids, subject, split, output_dir=f'{args.output_dir}/config{config}')




if __name__ == "__main__":
    main()




