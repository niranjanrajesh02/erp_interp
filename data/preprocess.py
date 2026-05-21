import os
import mne
import argparse
import numpy as np
import pandas as pd
import h5py, json
from pathlib import Path
from utilz.preprocessing_utilz import *
from utilz.vis_utilz import plot_all_channel_averages
mne.set_log_level('WARNING')

# argument parser
parser = argparse.ArgumentParser(description='Preprocess EEG data')
parser.add_argument('--data-dir', type=str, required=True, help='Path to the raw EEG data directory')
parser.add_argument('--output-dir', type=str, required=True, help='Path to save the preprocessed data')
parser.add_argument('--subset-trials', type=str, default='all', help='Subset of trials to preprocess (e.g., "all", "block_start", "block_end")')
parser.add_argument('--hp-freq', type=float, default=0.1, required=True, help='High-pass filter cutoff frequency')
parser.add_argument('--lp-freq', type=float, default=40, required=False, help='Low-pass filter cutoff frequency')
parser.add_argument('--target-sfreq', type=float, default=250, help='Sampling frequency')
parser.add_argument('--epoch-tmin', type=float, default=-0.2, help='Epoch start time (in seconds)')
parser.add_argument('--epoch-tmax', type=float, default=0.8, help='Epoch end time (in seconds)')
parser.add_argument('--artifact-threshold', type=float, default=200e-6, help='Threshold for artifact rejection (in microvolts)')
parser.add_argument('--ica-n-components', type=int, default=63, help='Number of ICA components to compute')
parser.add_argument('--ica-random-state', type=int, default=42, help='Random state for ICA reproducibility')
# dataset website: https://osf.io/crxs4/wiki?wiki=dz9mj
# original preprocessing code: https://github.com/gifale95/eeg_encoding/tree/main/02_eeg_preprocessing

montage = mne.channels.make_standard_montage('easycap-M1')
def preprocess(args, subject_id=None, plot_dir=None):
    #make subdirs for raw, ica and epochs
    raw_plot_dir = os.path.join(plot_dir, "raw")
    os.makedirs(raw_plot_dir, exist_ok=True)
    ica_plot_dir = os.path.join(plot_dir, "ica")
    os.makedirs(ica_plot_dir, exist_ok=True)
    epochs_plot_dir = os.path.join(plot_dir, "epochs")
    os.makedirs(epochs_plot_dir, exist_ok=True)

    # 1. load raw data and concatenate sessions 
    split_data = load_raw_eeg(subject_id=subject_id, data_dir=args.data_dir, plot_dir=raw_plot_dir)

    split_epochs = {}

    for split in ['train', 'test']:
        print(f"Preprocessing {split} data...")
        raw = split_data[split]['raw_eeg']
        events = split_data[split]['events']
        # TODO: add option to only preprocess a subset of trials (e.g., block start/end) based on args.subset_trials

        #2. filter raw data
        filter_raw(raw=raw, hp_freq=args.hp_freq, lp_freq=args.lp_freq)
    
        #3. downsample raw data
        raw, events = downsample_raw(raw=raw, events=events, target_sfreq=args.target_sfreq)

        # 4. set montage
        raw.set_montage(montage)

        # 5. ica for artifact correction
        run_ica(raw, n_components=args.ica_n_components, random_state=args.ica_random_state, save_plot_dir=ica_plot_dir, subject_id=subject_id, split=split)

        #6. re-reference to average
        rereference_raw(raw)

        #7. epoching + artifact rejection + baseline correction
        epochs = epoch_raw(raw=raw, events=events, tmin=args.epoch_tmin, 
                           tmax=args.epoch_tmax, 
                           baseline=(args.epoch_tmin, 0.0),
                           reject_thresh=args.artifact_threshold)  
        
        # 7.5 plot & save epoched data across each channel
        plot_all_channel_averages(epochs, subject_id=subject_id, split=split, save_plot_dir=epochs_plot_dir)
        
        split_epochs[split] = epochs

    return split_epochs


def save_preprocessed(
    epochs: mne.Epochs,
    subject_id: int,
    split: str,
    output_dir: str,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fpath = os.path.join(output_dir, f'sub-{subject_id:02d}_{split}_epochs.h5')

    data     = epochs.get_data(picks='eeg').astype(np.float32)  # (n_ep, n_ch, n_t)
    stim_ids = epochs.events[:, 2].astype(np.int32)             # (n_ep,)
    times    = epochs.times.astype(np.float64)                  # (n_t,)
    ch_names = np.array(epochs.ch_names, dtype='S')             # bytes for h5py

    with h5py.File(fpath, 'w') as f:
        # --- EEG data ---
        f.create_dataset('eeg',      data=data,     compression='gzip', compression_opts=4)
        f.create_dataset('stim_ids', data=stim_ids)
        f.create_dataset('times',    data=times)
        f.create_dataset('ch_names', data=ch_names)

        # --- Metadata ---
        f.attrs['subject_id'] = subject_id
        f.attrs['split']      = split
    

    print(f"  Saved {data.shape} → {fpath}")



def main():

    args = parser.parse_args()


    # create output data directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    new_config_df = pd.DataFrame([vars(args)])
    config_idx = 0
    # list all current configs in output dir and check if current config already exists
    config_dirs = [d for d in os.listdir(args.output_dir)
        if os.path.isdir(os.path.join(args.output_dir, d)) and d.startswith('config_')]

    config_idx = 0
    for config_dir in config_dirs:
        config_path = os.path.join(args.output_dir, config_dir, "preprocessing_config.csv")
        if os.path.exists(config_path):
            existing_config_df = pd.read_csv(config_path)
            if existing_config_df.equals(new_config_df):
                print(f"Config already exists in {config_dir}. Skipping.")
                return
            idx = int(config_dir.split('_')[-1])
            config_idx = max(config_idx, idx)  # highest existing index 
    config_idx += 1
    print(f"First time running with this configuration. Using config index {config_idx} for new preprocessing run.")
    current_config_dir = os.path.join(args.output_dir, f"config_{config_idx}")

    os.makedirs(current_config_dir, exist_ok=True)
    new_config_df.to_csv(os.path.join(current_config_dir, "preprocessing_config.csv"), index=False)


    save_plot_dir = os.path.join(current_config_dir, "preprocessing_plots")
    os.makedirs(save_plot_dir, exist_ok=True)

    # start preprocessing for each subject and save preprocessed data
    for subject_id in range(1,6):  # loop through subjects 1-5
        print(f"\n=== Preprocessing Subject {subject_id:02d} ===")
        split_epochs = preprocess(args, subject_id=subject_id, plot_dir=save_plot_dir)
        for split, epochs in split_epochs.items():
            save_preprocessed(epochs=epochs, subject_id=subject_id, split=split, output_dir=current_config_dir)

    return

if __name__ == "__main__":   
    main()