import re
import os
import mne
import h5py
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from utilz.vis_utilz import plot_raw_channel_overview
from pathlib import Path
import pandas as pd
from typing import Tuple


def _concatenate_events_with_offsets(
    event_list: list[np.ndarray],
    length_list: list[int],
) -> np.ndarray:
    """
    Concatenate per-session event arrays, adding cumulative sample offsets
    to column 0 (sample index) so indices are correct in the concatenated raw.

    Parameters
    ----------
    event_list  : list of (n_events, 3) arrays — one per session
    length_list : list of ints — n_times for each session raw, same order

    Returns
    -------
    np.ndarray of shape (total_events, 3) with corrected sample indices
    """
    adjusted = []
    offset = 0
    for events, length in zip(event_list, length_list):
        ev = events.copy()
        ev[:, 0] += offset   # shift sample indices by cumulative length
        adjusted.append(ev)
        offset += length
    return np.concatenate(adjusted, axis=0)


def load_raw_eeg(subject_id: int, data_dir: str, num_sessions: int = 4, plot_dir: str = None) -> Dict[str, Dict[str, List[mne.io.RawArray, np.ndarray]]]:
    '''
    Load raw EEG data and events for a given subject across multiple sessions.
    
    Parameters
    ----------
    subject_id : int
        The ID of the subject for whom to load data.
    data_dir : str
        The directory containing the subject's data.
    num_sessions : int, optional
        The number of sessions to load (default is 4).
    plot_dir : str, optional
        If provided, directory to save raw signal overview plots for each session.

    Returns
    -------
    split_data : dict
        A dictionary with keys 'train' and 'test', each containing:
        - 'raw_eeg': an mne.io.RawArray object with concatenated EEG data
        - 'events': a numpy array of shape (n_events, 3) with event information
    '''
    split_session_data = {
        'train': {'raw_eeg': [], 'lengths': []},
        'test':  {'raw_eeg': [], 'lengths': []},
    }
    split_data = {
        'train': {'raw_eeg': None, 'events': []},
        'test':  {'raw_eeg': None, 'events': []},
    }
    channel_names = None

    channel_names = None
    

    for session in range(1, num_sessions + 1):
        fpath = os.path.join(data_dir, f'sub-{subject_id:02}', f'sub-{subject_id:02}', f'ses-{session:02}')
        train_data = np.load(os.path.join(fpath, 'raw_eeg_training.npy'), allow_pickle=True).item()
        test_data = np.load(os.path.join(fpath, 'raw_eeg_test.npy'), allow_pickle=True).item()

        if channel_names is None:
            channel_names = train_data['ch_names']
            channel_types = train_data['ch_types']
            sfreq = train_data['sfreq']
        
        train_eeg = train_data['raw_eeg_data']
        test_eeg = test_data['raw_eeg_data']

        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)
        train_raw = mne.io.RawArray(train_eeg, info)
        test_raw = mne.io.RawArray(test_eeg, info)


        if plot_dir:
            plot_raw_channel_overview(
                train_raw,
                subject_id=subject_id,
                split=f'train_ses-{session:02d}',
                save_plot_dir=plot_dir,
                duration_sec=10.0,
            )
            plot_raw_channel_overview(
                test_raw,
                subject_id=subject_id,
                split=f'test_ses-{session:02d}',
                save_plot_dir=plot_dir,
                duration_sec=10.0,
            )

        del train_eeg, test_eeg, train_data, test_data
        
        train_events = mne.find_events(train_raw, stim_channel='stim')
        test_events = mne.find_events(test_raw, stim_channel='stim')
        # event structure: [sample_idx, 0, event_code]

        # print(f"{train_events.shape} training events")

        # reject target trials
        idx_target_train = np.where(train_events[:, 2] == 99999)[0]
        idx_target_test = np.where(test_events[:, 2] == 99999)[0]
        train_events = np.delete(train_events, idx_target_train, axis=0)
        test_events = np.delete(test_events, idx_target_test, axis=0)

        split_session_data['train']['raw_eeg'].append(train_raw)
        split_session_data['train']['lengths'].append(train_raw.n_times) 
        split_session_data['test']['raw_eeg'].append(test_raw)
        split_session_data['test']['lengths'].append(test_raw.n_times)
        split_data['train']['events'].append(train_events)
        split_data['test']['events'].append(test_events)



    for split in ['train', 'test']:
        split_data[split]['raw_eeg'] =  mne.concatenate_raws(split_session_data[split]['raw_eeg'])
        split_data[split]['events'] = _concatenate_events_with_offsets(
            split_data[split]['events'],
            split_session_data[split]['lengths'],
        )

   
    print(f"Subject {subject_id}: Loaded raw EEG data and events.")
    return split_data

      
def filter_raw(raw: mne.io.Raw, hp_freq: float, lp_freq: float) -> mne.io.Raw:
    '''
    Apply a bandpass filter to the raw EEG data.
    
    Parameters
    ----------
    raw     : mne.io.Raw — the raw EEG data to filter
    hp_freq : float — high-pass cutoff frequency in Hz (e.g., 0.1)
    lp_freq : float — low-pass cutoff frequency in Hz (e.g., 40)

    Returns
    -------
    mne.io.Raw — the filtered raw EEG data (in-place modification)
    
    '''
    print(f"Applying bandpass filter: {hp_freq} {f' - {lp_freq}' if lp_freq else ''} Hz")
    raw.filter(
        l_freq=hp_freq,
        h_freq=lp_freq,
        method='fir',
        fir_design='firwin',
        fir_window='hamming',
        phase='zero',
        picks='eeg',
    )
    print(f"Filtering complete. sfreq={raw.info['sfreq']:.0f} Hz, shape={raw.get_data(picks='eeg').shape}")
    return raw



def downsample_raw(
    raw: mne.io.Raw,
    events: np.ndarray,
    target_sfreq: float = 250.0,
) -> tuple[mne.io.Raw, np.ndarray]:
    """
    Downsample continuous raw EEG and rescale event sample indices.

    Parameters
    ----------
    raw          : mne.io.Raw  — filtered continuous raw
    events       : np.ndarray  — (n_events, 3) at original sfreq
    target_sfreq : float       — target sampling rate in Hz (default 250)

    Returns
    -------
    raw    : mne.io.Raw   — downsampled in-place
    events : np.ndarray   — (n_events, 3) with col 0 rescaled to target_sfreq
    """
    original_sfreq = raw.info['sfreq']

    if original_sfreq == target_sfreq:
        print(f"  [downsample] Already at {target_sfreq:.0f} Hz, skipping")
        return raw, events

    scale = target_sfreq / original_sfreq
    n_before = raw.n_times

    raw.resample(target_sfreq, verbose=False)

    # Rescale event sample indices (col 0) to match new sfreq
    # cols 1 and 2 (prev event, event ID) are unchanged
    events = events.copy()
    events[:, 0] = np.round(events[:, 0] * scale).astype(int)

    print(f"  [downsample] {original_sfreq:.0f} -> {target_sfreq:.0f} Hz  "
          f"({n_before} -> {raw.n_times} samples)")
    return raw, events


def run_ica(
    raw: mne.io.Raw,
    n_components: int = 30,
    random_state: int = 42,
    fit_hp_freq: float = 1.0,
    save_plot_dir: str = None,
    subject_id: int = None,
    split: str = None,
) -> mne.io.Raw:
    """
    Fit ICA on a 1 Hz high-pass filtered copy of raw, then apply the
    resulting component exclusions to the original (0.1 Hz filtered) raw.

    This two-pass approach is best practice: ICA decomposes more cleanly
    on 1 Hz filtered data (less slow drift), but we preserve the slow
    components (P3, LPC) in the analysis data by applying to 0.1 Hz raw.

    Parameters
    ----------
    raw           : mne.io.Raw — continuous raw EEG, already filtered at 0.1 Hz
    n_components  : int        — number of ICA components to compute (default 30)
                                 rule of thumb: <= n_channels - n_bad_channels
    random_state  : int        — random seed for reproducibility
    fit_hp_freq   : float      — high-pass cutoff for ICA fitting copy (default 1.0 Hz)
    save_plot_dir : str        — if not None, directory to save ICA diagnostic plots
    subject_id    : int        — subject ID for plot titles and filenames (optional)
    split         : str        — 'train' or 'test' for plot titles and filenames

    Returns
    -------
    mne.io.Raw — original raw with artifact components removed in-place

    Notes
    -----
    - EOG and ECG components are identified automatically via channel
      correlation. Verify manually on first subject (see plot lines below).
    - ICA is fit on EEG channels only — stim channel is excluded.
    - If no EOG/ECG channel is present, mne falls back to finding the
      worst-correlated frontal EEG channel as a proxy — this is less
      reliable; check the scores plots.
    """

    # --- Step 1: Make a 1 Hz high-pass copy just for fitting ICA ---
    # Do NOT use this copy for analysis — only for decomposition
    print(f"  [ICA] Creating 1 Hz high-pass copy for fitting...")
    raw_for_ica = raw.copy().filter(
        l_freq=fit_hp_freq,
        h_freq=None,
        picks='eeg',
        method='iir',
        iir_params=dict(order=4, ftype='butter'),
        verbose=False,
    )

    # --- Step 2: Fit ICA on the 1 Hz copy ---
    print(f"  [ICA] Fitting ICA with {n_components} components...")
    ica = ICA(
        n_components=n_components,
        method='fastica',
        random_state=random_state,
        max_iter='auto',
    )
    ica.fit(raw_for_ica, picks='eeg', verbose=False)
    del raw_for_ica  # free memory — no longer needed

    # --- Step 3: Auto-detect ocular (EOG) components ---
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2'], verbose=False)
    print(f"  [ICA] EOG components detected: {eog_indices}")

    
    # --- Step 5: Mark components for exclusion ---
    ica.exclude = list(set(eog_indices))
    print(f"  [ICA] Total components excluded: {len(ica.exclude)} -> {ica.exclude}")

    # --- Step 6: Verification plots (uncomment for first subject) ---
    if save_plot_dir is not None:
        os.makedirs(save_plot_dir, exist_ok=True)
        ica.plot_components(show=False)                        # topographies of all components
        plt.savefig(os.path.join(save_plot_dir, f'ica_components_{subject_id}_{split}.png'), bbox_inches='tight', dpi=300)
        ica.plot_scores(eog_scores, title='EOG', show=False)     # correlation scores per component
        plt.savefig(os.path.join(save_plot_dir, f'ica_eog_scores_{subject_id}_{split}.png'), bbox_inches='tight', dpi=300)
        ica.plot_overlay(raw, exclude=ica.exclude, show=False)   # before/after on raw signal
        plt.savefig(os.path.join(save_plot_dir, f'ica_overlay_{subject_id}_{split}.png'), bbox_inches='tight', dpi=300)


    # --- Step 7: Apply to the original 0.1 Hz filtered raw ---
    print(f"  [ICA] Applying to 0.1 Hz filtered raw...")
    ica.apply(raw, verbose=False)
    print(f"  [ICA] Done.")

    return raw

def rereference_raw(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Re-reference EEG to the average of all channels.

    Parameters
    ----------
    raw : mne.io.Raw — ICA-cleaned continuous raw

    Returns
    -------
    mne.io.Raw — re-referenced in-place
    """
    print(f"  [reref] Applying average reference...")
    raw.set_eeg_reference('average', projection=False, verbose=False)
    print(f"  [reref] Done.")
    return raw


def epoch_raw(
    raw: mne.io.Raw,
    events: np.ndarray,
    tmin: float = -0.2,
    tmax: float = 0.8,
    baseline: tuple = (-0.2, 0.0),
    reject_thresh: float = 200e-6,
) -> mne.Epochs:
    """
    Epoch, baseline-correct, and reject bad epochs.

    Parameters
    ----------
    raw           : mne.io.Raw  — re-referenced, ICA-cleaned continuous raw
    events        : np.ndarray  — (n_events, 3) at current sfreq
    tmin          : float       — epoch start in seconds (default -0.2)
    tmax          : float       — epoch end in seconds (default 0.8)
    baseline      : tuple       — baseline window in seconds (default (-0.2, 0.0))
    reject_thresh : float       — peak-to-peak rejection threshold in Volts
                                  (default 200e-6 = 200 µV)

    Returns
    -------
    mne.Epochs — epoched, baseline-corrected, bad trials dropped
    """
    print(f"  [epoch] Epoching {tmin} to {tmax} s, baseline {baseline}...")

    epochs = mne.Epochs(
        raw,
        events,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        reject=None,       # apply separately so we can log dropped trials
        picks='eeg',
        verbose=False,
    )

    n_before = len(epochs)

    # Track which stimulus IDs lose trials before rejecting
    pre_rejection_ids = epochs.events[:, 2].copy()

    epochs.drop_bad(reject=dict(eeg=reject_thresh), verbose=False)

    n_after  = len(epochs)
    n_dropped = n_before - n_after
    print(f"  [epoch] {n_before} epochs -> {n_after} kept, "
          f"{n_dropped} dropped ({100 * n_dropped / n_before:.1f}%)")

    # Log which stimulus IDs had trials dropped
    kept_ids    = set(epochs.events[:, 2])
    dropped_ids = set(pre_rejection_ids) - kept_ids
    if dropped_ids:
        print(f"  [epoch] {len(dropped_ids)} stimulus IDs lost all trials: "
              f"{sorted(dropped_ids)[:10]}{'...' if len(dropped_ids) > 10 else ''}")

    return epochs


def get_stim_img_and_category(stim_ids: np.ndarray, split: str, image_metadata: dict, concepts_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Given an array of stimulus IDs, return the corresponding image file paths and category labels.
    
    Args:
        stim_ids (np.ndarray): Array of stimulus IDs.
        split (str): The split (e.g., 'train' or 'test').
        image_metadata (dict): Dictionary containing image metadata.
        concepts_df (pd.DataFrame): DataFrame containing concept metadata.

    Returns:
        img_files (List[str]): List of image file paths corresponding to the stimulus IDs.
        categories (List[str]): List of category labels corresponding to the stimulus IDs.
    """
    
    img_files = []
    categories = []

    all_img_paths = image_metadata[f'{split}_img_files']
    all_img_names = image_metadata[f'{split}_img_concepts']

    img_files = [all_img_paths[stim_id-1] for stim_id in stim_ids]
    
    # map stim_ids to their corresponding concepts and then to their categories
    for stim_id in stim_ids:
        concept = all_img_names[stim_id-1].split('_',1)[1] # split after number and _
        # replace _ with space
        concept = concept.replace('_', ' ')
        #remove any numbers from the concept
        concept = re.findall(r'[^\d]+', concept)[0].strip()
        if concept == "flip flop": concept = "flip-flop" # special case for flip flop which is not found in concepts_df
        
        if concept not in concepts_df['Word'].values:
            print(f"Warning: Concept '{concept}' not found in concepts_df. Assigning category as 'Unknown'.")
            return
        

        category = concepts_df.loc[concepts_df['Word'] == concept, 'Top-down Category (WordNet)'].values
        categories.append(category[0])
    
    return img_files, categories
    
def load_eeg_preprocessed(preprocessed_dir: str, subject: int, config: int, split: str) -> Dict[str, np.ndarray]:
    eeg_data_path = os.path.join(preprocessed_dir, f'config_{config}', f'sub-{subject:02d}_{split}_epochs.h5')
    with h5py.File(eeg_data_path, 'r') as f:
        eeg      = f['eeg'][:]                              # loads fully into RAM
        stim_ids = f['stim_ids'][:]
        times    = f['times'][:]
        ch_names = [c.decode() for c in f['ch_names'][:]]  # bytes -> str
        meta     = dict(f.attrs)
    
    print(f"Loaded sub-{subject:02d} {split}:")
    print(f"  EEG      : {eeg.shape}  (epochs, channels, timepoints)")
    print(f"  Stim IDs : {stim_ids.shape}  unique={len(np.unique(stim_ids))}")
    print(f"  Times    : {times[0]*1000:.0f} to {times[-1]*1000:.0f} ms  "
          f"@ {1/(times[1]-times[0]):.0f} Hz")
    print(f"  Channels : {len(ch_names)}")
    
    return {'eeg': eeg, 'stim_ids': stim_ids, 'times': times, 'ch_names': ch_names, 'meta': meta}


def average_across_reps(eeg_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    eeg = eeg_data['eeg']
    stim_ids = eeg_data['stim_ids']
    
    unique_ids = np.unique(stim_ids)  

    erps = np.stack([
        eeg[stim_ids == sid].mean(axis=0)
        for sid in unique_ids
    ]).astype(np.float32)

    # Sanity check
    reps_per_stim = np.array([np.sum(stim_ids == sid) for sid in unique_ids])
    print(f"  Averaged: {len(eeg)} epochs → {len(unique_ids)} ERPs")
    print(f"  Reps per stimulus: min={reps_per_stim.min()}  "
          f"max={reps_per_stim.max()}  "
          f"mean={reps_per_stim.mean():.2f}")
    
    return erps, unique_ids

def extract_erp_features(
    erps: np.ndarray,        # (n_stimuli, n_ch, n_t)
    times: np.ndarray,       # (n_t,) in seconds
    ch_names: list,
    component_windows: dict,
) -> dict:
    """
    For each ERP component, extracts:
      scalar  : (n_stimuli,)          — mean amplitude over ROI channels × time window
      vector  : (n_stimuli, n_ch*n_t) — flattened spatiotemporal pattern
      ch_used : list[str]             — which channels were actually found
      t_used  : (n_t_roi,)            — timepoints used (ms)
    """
    times_ms = times * 1000
    results  = {}

    for comp, spec in component_windows.items():
        # --- channel mask ---
        valid_chs = [ch for ch in spec["spatial"] if ch in ch_names]
        missing   = [ch for ch in spec["spatial"] if ch not in ch_names]
        if missing:
            print(f"  [{comp}] channels not in montage (skipped): {missing}")
        if not valid_chs:
            print(f"  [{comp}] WARNING: no valid channels — skipping component")
            continue
        ch_idx = np.array([ch_names.index(ch) for ch in valid_chs])

        # --- temporal mask ---
        t0, t1 = spec["temporal"]
        t_mask = (times_ms >= t0) & (times_ms <= t1)
        if t_mask.sum() == 0:
            print(f"  [{comp}] WARNING: no timepoints in [{t0}, {t1}] ms — skipping")
            continue

        # roi_data : (n_stimuli, n_roi_ch, n_roi_t)
        roi_data = erps[:, ch_idx, :][:, :, t_mask]

        results[comp] = {
            "scalar":  roi_data.mean(axis=(1, 2)).astype(np.float32),   # (n_stimuli,)
            "vector":  roi_data.reshape(erps.shape[0], -1).astype(np.float32),
            "ch_used": valid_chs,
            "t_used":  times_ms[t_mask],
        }

        print(
            f"  {comp:6s} | "
            f"channels: {len(valid_chs):2d} {valid_chs}  |  "
            f"window: {t0}–{t1} ms ({t_mask.sum()} pts)  |  "
            f"scalar: {results[comp]['scalar'].shape}  "
            f"vector: {results[comp]['vector'].shape}"
        )

    return results


def save_erp_features(
    features:   dict,
    unique_ids: np.ndarray,
    subject_id: int,
    split:      str,
    output_dir: str,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fpath = os.path.join(output_dir, f"sub-{subject_id:02d}_{split}_erp_features.h5")

    with h5py.File(fpath, "w") as f:
        f.create_dataset("unique_ids", data=unique_ids.astype(np.int32))
        f.attrs["subject_id"] = subject_id
        f.attrs["split"]      = split

        # per-component groups
        comp_grp = f.create_group("components")
        for comp, data in features.items():
            if comp == "__joint__":
                continue
            grp = comp_grp.create_group(comp)
            grp.create_dataset("scalar", data=data["scalar"], compression="gzip")
            grp.create_dataset("vector", data=data["vector"], compression="gzip")
            grp.create_dataset("t_used", data=data["t_used"])
            grp.attrs["ch_used"] = data["ch_used"]


    print(f"\n  Saved → {fpath}")
    return fpath


def load_erp_features(fpath: str) -> dict:
    """Load saved ERP features back into the same dict structure."""
    out = {"components": {}, "joint": {}}
    with h5py.File(fpath, "r") as f:
        out["unique_ids"] = f["unique_ids"][:]
        out["subject_id"] = f.attrs["subject_id"]
        out["split"]      = f.attrs["split"]

        for comp in f["components"]:
            grp = f["components"][comp]
            out["components"][comp] = {
                "scalar":  grp["scalar"][:],
                "vector":  grp["vector"][:],
                "t_used":  grp["t_used"][:],
                "ch_used": list(grp.attrs["ch_used"]),
            }

       
    return out


def save_stim_metadata(
    stim_ids: np.ndarray,
    img_files: List[str],
    categories: List[str],
    subject_id: int,
    split: str,
    output_dir: str,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fpath = os.path.join(output_dir, f"sub-{subject_id:02d}_{split}_stim_metadata.h5")

    with h5py.File(fpath, "w") as f:
        f.create_dataset("stim_ids", data=stim_ids.astype(np.int32))
        f.create_dataset("img_files", data=np.array(img_files, dtype='S'), compression="gzip")
        f.create_dataset("categories", data=np.array(categories, dtype='S'), compression="gzip")
        f.attrs["subject_id"] = subject_id
        f.attrs["split"]      = split

    print(f"\n  Saved stimulus metadata → {fpath}")
    return fpath


def load_stim_metadata(fpath: str) -> dict:
    """Load saved stimulus metadata back into a dict."""
    out = {}
    with h5py.File(fpath, "r") as f:
        out["stim_ids"] = f["stim_ids"][:]
        out["img_files"] = [s.decode() for s in f["img_files"][:]]
        out["categories"] = [s.decode() for s in f["categories"][:]]
        out["subject_id"] = f.attrs["subject_id"]
        out["split"]      = f.attrs["split"]   
    
    print(f"\n  Loaded stimulus metadata from {fpath} for subject {out['subject_id']} split {out['split']}")

    return out