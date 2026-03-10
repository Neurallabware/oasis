"""Visualization utilities for OASIS deconvolution results.

Plotting functions for fluorescence traces, denoised calcium signals,
and inferred spike trains.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_deconvolution(y, c, s, bl=None, framerate=None, title=None,
                       figsize=(14, 6), show=True, save_path=None):
    """Plot fluorescence trace with deconvolution results.

    Shows raw data, denoised calcium trace, and inferred spike train
    in a 3-panel layout.

    Args:
        y: np.ndarray
            Raw fluorescence trace.
        c: np.ndarray
            Denoised calcium trace (from constrained_foopsi).
        s: np.ndarray
            Inferred spike train.
        bl: float, optional
            Baseline value. If given, shown as a horizontal line.
        framerate: float, optional
            Frame rate in Hz. If given, x-axis is in seconds.
        title: str, optional
            Figure title.
        figsize: tuple
            Figure size.
        show: bool
            Whether to call plt.show().
        save_path: str, optional
            Path to save the figure.

    Returns:
        fig: matplotlib.figure.Figure
        axes: array of matplotlib.axes.Axes
    """
    T = len(y)
    if framerate is not None:
        t = np.arange(T) / framerate
        xlabel = 'Time (s)'
    else:
        t = np.arange(T)
        xlabel = 'Frame'

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 1, height_ratios=[2, 2, 1], hspace=0.3)

    # Panel 1: Raw fluorescence with denoised overlay
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, y, color='gray', alpha=0.7, linewidth=0.8, label='Raw fluorescence')
    ax1.plot(t, c + (bl if bl is not None else 0), color='#1f77b4', linewidth=1.2,
             label='Denoised (OASIS)')
    if bl is not None:
        ax1.axhline(y=bl, color='#d62728', linestyle='--', alpha=0.5, linewidth=0.8,
                     label=f'Baseline = {bl:.2f}')
    ax1.set_ylabel('Fluorescence')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlim(t[0], t[-1])

    # Panel 2: Denoised calcium trace only
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t, c, color='#1f77b4', linewidth=1.0)
    ax2.fill_between(t, 0, c, alpha=0.15, color='#1f77b4')
    ax2.set_ylabel('Calcium (a.u.)')
    ax2.set_xlim(t[0], t[-1])

    # Panel 3: Spike train
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    spike_idx = np.where(s > 0)[0]
    if len(spike_idx) > 0:
        spike_times = t[spike_idx]
        spike_heights = s[spike_idx]
        ax3.vlines(spike_times, 0, spike_heights, color='#2ca02c', linewidth=1.0)
    ax3.set_ylabel('Spikes')
    ax3.set_xlabel(xlabel)
    ax3.set_xlim(t[0], t[-1])
    ax3.set_ylim(bottom=0)

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')

    axes = [ax1, ax2, ax3]

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    return fig, axes


def plot_comparison(y, results, labels=None, framerate=None, title=None,
                    figsize=(14, 8), show=True, save_path=None):
    """Compare multiple deconvolution results side by side.

    Args:
        y: np.ndarray
            Raw fluorescence trace.
        results: list of tuples
            Each tuple is (c, s) or (c, s, bl) from different parameter settings.
        labels: list of str, optional
            Labels for each result.
        framerate: float, optional
            Frame rate in Hz.
        title: str, optional
            Figure title.
        figsize: tuple
            Figure size.
        show: bool
            Whether to call plt.show().
        save_path: str, optional
            Path to save the figure.

    Returns:
        fig: matplotlib.figure.Figure
        axes: array of matplotlib.axes.Axes
    """
    n = len(results)
    if labels is None:
        labels = [f'Result {i+1}' for i in range(n)]

    T = len(y)
    if framerate is not None:
        t = np.arange(T) / framerate
        xlabel = 'Time (s)'
    else:
        t = np.arange(T)
        xlabel = 'Frame'

    fig, axes = plt.subplots(n + 1, 1, figsize=figsize, sharex=True)

    # Top panel: raw data
    axes[0].plot(t, y, color='gray', linewidth=0.8)
    axes[0].set_ylabel('Raw')
    axes[0].set_title(title or 'Deconvolution Comparison')

    colors = plt.cm.tab10(np.linspace(0, 1, n))

    for i, (res, label) in enumerate(zip(results, labels)):
        c = res[0]
        s = res[1]
        bl = res[2] if len(res) > 2 else 0

        ax = axes[i + 1]
        ax.plot(t, c + bl, color=colors[i], linewidth=1.0, label=label)

        # Overlay spikes as markers
        spike_idx = np.where(s > 0)[0]
        if len(spike_idx) > 0:
            ax.vlines(t[spike_idx], 0, s[spike_idx] * 0.3 + np.max(c) * 0.8,
                      color=colors[i], alpha=0.4, linewidth=0.5)
        ax.set_ylabel(label)

    axes[-1].set_xlabel(xlabel)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    return fig, axes


def plot_trace_overlay(y, c, bl=None, framerate=None, ax=None, show=True):
    """Simple overlay plot of raw fluorescence and denoised calcium.

    Args:
        y: np.ndarray
            Raw fluorescence.
        c: np.ndarray
            Denoised calcium.
        bl: float, optional
            Baseline.
        framerate: float, optional
            Frame rate in Hz.
        ax: matplotlib.axes.Axes, optional
            Axes to plot on. Created if not given.
        show: bool
            Whether to call plt.show().

    Returns:
        ax: matplotlib.axes.Axes
    """
    T = len(y)
    if framerate is not None:
        t = np.arange(T) / framerate
        xlabel = 'Time (s)'
    else:
        t = np.arange(T)
        xlabel = 'Frame'

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))

    ax.plot(t, y, color='gray', alpha=0.6, linewidth=0.7, label='Raw')
    ax.plot(t, c + (bl if bl is not None else 0), color='#1f77b4', linewidth=1.0,
            label='Denoised')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Fluorescence')
    ax.legend(fontsize=8)

    if show:
        plt.show()

    return ax
