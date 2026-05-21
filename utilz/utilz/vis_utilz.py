import os
import mne
import numpy as np
import matplotlib.pyplot as plt


def plot_raw_channel_overview(
    raw: mne.io.Raw,
    subject_id: int,
    split: str,
    save_plot_dir: str = None,
    duration_sec: float = 10.0,   
    n_cols: int = 8,
):
    """
    Plot a short segment of the continuous raw signal for every EEG channel.
    Shows the first `duration_sec` seconds of the recording to give a quick
    visual check of signal quality before any preprocessing.

    Look for:
    - Dead channels (flat line)
    - Noisy channels (amplitude >> neighbours)
    - 50 Hz hum (dense oscillation visible at this zoom)
    - Obvious saturation (square clipping at ±max range)
    - Gross drift (signal wandering far from zero)
    """
    sfreq      = raw.info['sfreq']
    n_samples  = int(duration_sec * sfreq)
    data_uv    = raw.get_data(picks='eeg')[:, :n_samples] * 1e6   # V -> µV
    times_sec  = raw.times[:n_samples]

    ch_names   = [ch for ch in raw.ch_names if ch in raw.copy().pick('eeg').ch_names]
    n_ch       = data_uv.shape[0]
    n_rows     = int(np.ceil(n_ch / n_cols))

    fig, axes  = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2))
    axes       = axes.flatten()

    for i, ax in enumerate(axes[:n_ch]):
        ax.plot(times_sec, data_uv[i], linewidth=0.6, color='steelblue')
        ax.axhline(0, color='k', linewidth=0.3)
        ax.set_title(ch_names[i], fontsize=7, pad=2)
        ax.set_xlim(times_sec[0], times_sec[-1])
        ax.tick_params(labelsize=5)

        # Flag visually: red title for suspiciously flat or noisy channels
        ch_std = data_uv[i].std()
        if ch_std < 0.5 or ch_std > 100:
            ax.set_title(ch_names[i], fontsize=7, pad=2, color='red')

    for ax in axes[n_ch:]:
        ax.set_visible(False)

    fig.supxlabel('Time (s)', fontsize=9)
    fig.supylabel('Amplitude (µV)', fontsize=9)
    fig.suptitle(
        f"Sub-{subject_id:02d} {split.upper()} — Raw signal (first {duration_sec:.0f} s)",
        fontsize=11,
    )
    plt.tight_layout()

    if save_plot_dir:
        os.makedirs(save_plot_dir, exist_ok=True)
        fpath = os.path.join(save_plot_dir, f'sub-{subject_id:02d}_{split}_raw_overview.png')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"  Saved raw overview plot → {fpath}")
        plt.close(fig)
    else:
        plt.show()

def plot_all_channel_averages(epochs: mne.Epochs, subject_id: int, split: str, save_plot_dir: str = None):
    """
    Plot grand average ERP waveform for every EEG channel in a grid layout.
    Useful for quickly scanning all channels for artifacts, polarity issues,
    or unexpected noise before deeper analysis.
    """
    evoked = epochs.average()
    times_ms = evoked.times * 1000
    ch_names = evoked.ch_names
    data_uv = evoked.data * 1e6    # convert V -> µV

    n_ch = len(ch_names)
    n_cols = 8
    n_rows = int(np.ceil(n_ch / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2))
    axes = axes.flatten()

    for i, (ax, ch_name) in enumerate(zip(axes, ch_names)):
        ax.plot(times_ms, data_uv[i], linewidth=0.8, color='steelblue')
        ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
        ax.axhline(0, color='k', linewidth=0.3)
        ax.set_xlim(-200, 800)
        ax.set_title(ch_name, fontsize=7, pad=2)
        ax.tick_params(labelsize=5)
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Hide any unused axes (if n_ch not divisible by n_cols)
    for ax in axes[n_ch:]:
        ax.set_visible(False)

    fig.supxlabel('Time (ms)', fontsize=9)
    fig.supylabel('Amplitude (µV)', fontsize=9)
    fig.suptitle(
        f"Sub-{subject_id:02d} {split.upper()} — Grand Average, All Channels",
        fontsize=11,
    )
    plt.tight_layout()
    
    if save_plot_dir is not None:
        plt.savefig(os.path.join(save_plot_dir, f"channel_averages_{subject_id:02d}_{split}.png"), dpi=300, bbox_inches='tight')