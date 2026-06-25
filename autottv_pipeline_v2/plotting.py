"""
Plotting Functions for AutoTTV Pipeline v2.0

Generates all required visualizations:
1. MCMC chain plots (multi-panel)
2. Corner plots (parameter covariance)
3. Phase-folded light curve with model
4. O-C diagrams (linear and quadratic)
5. Lomb-Scargle periodogram
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

try:
    import corner
    CORNER_AVAILABLE = True
except ImportError:
    CORNER_AVAILABLE = False
    logging.warning("corner package not installed")

from . import config

logger = logging.getLogger(__name__)

# Plot style settings
plt.style.use('default')
COLORS = {
    'primary': '#1a365d',
    'secondary': '#ed8936',
    'accent': '#48bb78',
    'error': '#e53e3e',
    'gray': '#718096'
}


def setup_plot_style():
    """Set up consistent plot styling."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })


def plot_mcmc_chains(sampler, param_names: List[str], output_path: Path,
                     title: str = "MCMC Chain Evolution",
                     burnin_chain: Optional[np.ndarray] = None,
                     burnin_log_prob: Optional[np.ndarray] = None) -> Path:
    """
    Create multi-panel plot showing MCMC chain evolution for each parameter,
    plus log_probability for convergence diagnostics.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        The MCMC sampler after running
    param_names : list of str
        Names of fitted parameters
    output_path : Path
        Path to save the plot
    title : str
        Plot title
    burnin_chain : np.ndarray, optional
        Burn-in chains, shape (n_burnin_steps, n_walkers, n_params)
    burnin_log_prob : np.ndarray, optional
        Burn-in log probabilities, shape (n_burnin_steps, n_walkers)

    Returns
    -------
    Path
        Path to saved plot
    """
    setup_plot_style()

    # Get production chains
    prod_chains = sampler.get_chain()  # (n_steps, n_walkers, n_params)
    prod_log_prob = sampler.get_log_prob()  # (n_steps, n_walkers)
    n_prod_steps, n_walkers, n_params = prod_chains.shape

    # Combine burn-in and production if burn-in provided
    if burnin_chain is not None and burnin_log_prob is not None:
        n_burnin_steps = burnin_chain.shape[0]
        chains = np.concatenate([burnin_chain, prod_chains], axis=0)
        log_prob = np.concatenate([burnin_log_prob, prod_log_prob], axis=0)
        burnin_end = n_burnin_steps
    else:
        chains = prod_chains
        log_prob = prod_log_prob
        burnin_end = None

    n_steps = chains.shape[0]

    # Create figure with n_params + 1 panels (extra for log_probability)
    n_panels = n_params + 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.5 * n_panels), sharex=True)

    # Plot parameter chains (vectorized: all walkers in one call)
    for i, (ax, name) in enumerate(zip(axes[:-1], param_names)):
        ax.plot(chains[:, :, i], alpha=0.3, linewidth=0.5, color=COLORS['primary'])

        ax.set_ylabel(name)

        # Show median line
        median = np.median(chains[:, :, i], axis=1)
        ax.plot(median, color=COLORS['error'], linewidth=1.5, alpha=0.8, label='Median' if i == 0 else None)

        # Mark burn-in end
        if burnin_end is not None:
            ax.axvline(burnin_end, color=COLORS['secondary'], linestyle='--', linewidth=2,
                      label='Burn-in end' if i == 0 else None)

    # Plot log_probability in the last panel (vectorized)
    ax_logp = axes[-1]
    ax_logp.plot(log_prob, alpha=0.3, linewidth=0.5, color=COLORS['accent'])

    # Show median log_prob
    median_logp = np.median(log_prob, axis=1)
    ax_logp.plot(median_logp, color=COLORS['error'], linewidth=1.5, alpha=0.8)

    # Mark burn-in end on log_prob panel
    if burnin_end is not None:
        ax_logp.axvline(burnin_end, color=COLORS['secondary'], linestyle='--', linewidth=2)

    ax_logp.set_ylabel('log(probability)')
    ax_logp.set_xlabel('Step')

    # Add legend to first panel
    axes[0].legend(loc='upper right')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved chain plot to {output_path}")
    return output_path


def plot_corner(samples: np.ndarray, param_names: List[str], output_path: Path,
                title: str = "Parameter Covariance") -> Optional[Path]:
    """
    Create corner plot showing parameter covariances.

    Parameters
    ----------
    samples : np.ndarray
        MCMC samples, shape (n_samples, n_params)
    param_names : list of str
        Names of parameters
    output_path : Path
        Path to save the plot
    title : str
        Plot title

    Returns
    -------
    Path or None
        Path to saved plot, or None if corner not available
    """
    if not CORNER_AVAILABLE:
        logger.warning("corner package not available, skipping corner plot")
        return None

    setup_plot_style()

    fig = corner.corner(
        samples[::10],
        labels=param_names,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 10},
        color=COLORS['primary']
    )

    fig.suptitle(title, fontsize=14, y=1.02)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved corner plot to {output_path}")
    return output_path


def plot_phase_folded_lightcurve(phase: np.ndarray, flux: np.ndarray,
                                 flux_err: np.ndarray,
                                 bin_phase: np.ndarray, bin_flux: np.ndarray,
                                 bin_err: np.ndarray,
                                 model_phase: np.ndarray, model_flux: np.ndarray,
                                 output_path: Path,
                                 title: str = "Phase-Folded Light Curve") -> Path:
    """
    Create phase-folded light curve plot with binned data and model.

    Parameters
    ----------
    phase, flux, flux_err : np.ndarray
        Unbinned phase-folded data
    bin_phase, bin_flux, bin_err : np.ndarray
        Binned phase-folded data
    model_phase, model_flux : np.ndarray
        Model for plotting
    output_path : Path
        Path to save plot
    title : str
        Plot title

    Returns
    -------
    Path
        Path to saved plot
    """
    setup_plot_style()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                             sharex=True, gridspec_kw={'hspace': 0.05, 'height_ratios': [3, 1]})

    ax_main, ax_resid = axes

    # Main plot
    # Unbinned data (faint)
    ax_main.scatter(phase, flux, s=1, alpha=0.1, color=COLORS['gray'],
                    label='Unbinned data')

    # Binned data
    ax_main.errorbar(bin_phase, bin_flux, yerr=bin_err, fmt='o',
                     color=COLORS['primary'], markersize=4, capsize=2,
                     label='Binned data')

    # Model
    sort_idx = np.argsort(model_phase)
    ax_main.plot(model_phase[sort_idx], model_flux[sort_idx],
                 color=COLORS['secondary'], linewidth=2, label='Best-fit model')

    ax_main.set_ylabel('Normalized Flux')
    ax_main.legend(loc='lower right')
    ax_main.set_title(title)

    # Residuals plot
    # Interpolate model to bin positions for residuals
    model_at_bins = np.interp(bin_phase, model_phase[sort_idx], model_flux[sort_idx])
    residuals = (bin_flux - model_at_bins) * 1e6  # Convert to ppm

    ax_resid.errorbar(bin_phase, residuals, yerr=bin_err * 1e6, fmt='o',
                      color=COLORS['primary'], markersize=4, capsize=2)
    ax_resid.axhline(y=0, color=COLORS['gray'], linestyle='--', alpha=0.5)

    ax_resid.set_xlabel('Phase')
    ax_resid.set_ylabel('Residuals (ppm)')

    # Set x limits
    ax_resid.set_xlim(-0.1, 0.1)  # Focus on transit

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved phase-folded plot to {output_path}")
    return output_path


def plot_oc_diagram(epochs: np.ndarray, oc_minutes: np.ndarray,
                    oc_err_minutes: np.ndarray, output_path: Path,
                    model: str = "linear",
                    title: str = "O-C Diagram",
                    model_epochs: Optional[np.ndarray] = None,
                    model_oc: Optional[np.ndarray] = None) -> Path:
    """
    Create O-C (Observed minus Calculated) diagram.

    Parameters
    ----------
    epochs : np.ndarray
        Transit epoch numbers
    oc_minutes : np.ndarray
        O-C residuals in minutes
    oc_err_minutes : np.ndarray
        O-C errors in minutes
    output_path : Path
        Path to save plot
    model : str
        Model name for labeling
    title : str
        Plot title
    model_epochs, model_oc : np.ndarray, optional
        Model curve for overlay (e.g., quadratic trend)

    Returns
    -------
    Path
        Path to saved plot
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Data points
    ax.errorbar(epochs, oc_minutes, yerr=oc_err_minutes, fmt='o',
                color=COLORS['primary'], markersize=5, capsize=3,
                label=f'O-C ({model} ephemeris)')

    # Zero line
    ax.axhline(y=0, color=COLORS['gray'], linestyle='--', alpha=0.5)

    # Model curve if provided
    if model_epochs is not None and model_oc is not None:
        sort_idx = np.argsort(model_epochs)
        ax.plot(model_epochs[sort_idx], model_oc[sort_idx],
                color=COLORS['secondary'], linewidth=2, label='Model')

    # Statistics text
    rms = np.sqrt(np.mean(oc_minutes**2))
    ax.text(0.02, 0.98, f'RMS: {rms:.2f} min\nN: {len(epochs)}',
            transform=ax.transAxes, verticalalignment='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('O-C (minutes)')
    ax.set_title(title)
    ax.legend(loc='upper right')

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved O-C diagram to {output_path}")
    return output_path


def plot_oc_comparison(epochs: np.ndarray,
                       oc_linear_min: np.ndarray,
                       oc_quad_min: np.ndarray,
                       oc_err_min: np.ndarray,
                       output_path: Path,
                       title: str = "Linear vs Quadratic Ephemeris") -> Path:
    """
    Create comparison plot of O-C for linear and quadratic ephemerides.

    Parameters
    ----------
    epochs : np.ndarray
        Transit epochs
    oc_linear_min : np.ndarray
        O-C for linear ephemeris (minutes)
    oc_quad_min : np.ndarray
        O-C for quadratic ephemeris (minutes)
    oc_err_min : np.ndarray
        O-C errors (minutes)
    output_path : Path
        Path to save plot
    title : str
        Plot title

    Returns
    -------
    Path
        Path to saved plot
    """
    setup_plot_style()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={'hspace': 0.15})

    # Linear O-C
    axes[0].errorbar(epochs, oc_linear_min, yerr=oc_err_min, fmt='o',
                     color=COLORS['primary'], markersize=4, capsize=2)
    axes[0].axhline(y=0, color=COLORS['gray'], linestyle='--', alpha=0.5)
    axes[0].set_ylabel('O-C (minutes)')
    axes[0].set_title('Linear Ephemeris')
    rms_linear = np.sqrt(np.mean(oc_linear_min**2))
    axes[0].text(0.02, 0.98, f'RMS: {rms_linear:.2f} min',
                 transform=axes[0].transAxes, verticalalignment='top',
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Quadratic O-C
    axes[1].errorbar(epochs, oc_quad_min, yerr=oc_err_min, fmt='o',
                     color=COLORS['secondary'], markersize=4, capsize=2)
    axes[1].axhline(y=0, color=COLORS['gray'], linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('O-C (minutes)')
    axes[1].set_title('Quadratic Ephemeris')
    rms_quad = np.sqrt(np.mean(oc_quad_min**2))
    axes[1].text(0.02, 0.98, f'RMS: {rms_quad:.2f} min',
                 transform=axes[1].transAxes, verticalalignment='top',
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved O-C comparison to {output_path}")
    return output_path


def plot_periodogram(frequencies: np.ndarray, power: np.ndarray,
                     output_path: Path,
                     fap_01: Optional[float] = None,
                     fap_05: Optional[float] = None,
                     peak_freq: Optional[float] = None,
                     title: str = "Lomb-Scargle Periodogram") -> Path:
    """
    Create Lomb-Scargle periodogram plot with frequency on x-axis.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency array (1/day)
    power : np.ndarray
        Lomb-Scargle power
    output_path : Path
        Path to save plot
    fap_01, fap_05 : float, optional
        FAP levels to show
    peak_freq : float, optional
        Peak frequency to highlight
    title : str
        Plot title

    Returns
    -------
    Path
        Path to saved plot
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Main periodogram
    ax.plot(frequencies, power, color=COLORS['primary'], linewidth=0.8)

    # FAP levels
    if fap_01 is not None:
        ax.axhline(y=fap_01, color=COLORS['error'], linestyle='--',
                   label='1% FAP', alpha=0.7)
    if fap_05 is not None:
        ax.axhline(y=fap_05, color=COLORS['secondary'], linestyle=':',
                   label='5% FAP', alpha=0.7)

    # Peak marker
    if peak_freq is not None and len(frequencies) > 0:
        peak_idx = np.argmin(np.abs(frequencies - peak_freq))
        if peak_idx < len(power):
            ax.axvline(x=peak_freq, color=COLORS['accent'], linestyle='-.',
                       alpha=0.5)
            peak_period = 1.0 / peak_freq if peak_freq > 0 else np.inf
            ax.scatter([peak_freq], [power[peak_idx]], s=100, color=COLORS['accent'],
                       marker='*', zorder=5, label=f'Peak: {peak_period:.1f} days')

    ax.set_xlabel('Frequency (1/day)')
    ax.set_ylabel('Lomb-Scargle Power')
    ax.set_title(title)
    ax.legend(loc='upper right')

    # Add secondary x-axis for period
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    # Set period ticks
    freq_ticks = ax.get_xticks()
    freq_ticks = freq_ticks[freq_ticks > 0]
    period_ticks = 1.0 / freq_ticks
    ax2.set_xticks(freq_ticks)
    ax2.set_xticklabels([f'{p:.1f}' for p in period_ticks])
    ax2.set_xlabel('Period (days)')

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved periodogram to {output_path}")
    return output_path


def create_summary_figure(tic_id: int, step1_results: Dict,
                         transit_results: List[Dict],
                         ephemeris_results: Dict,
                         periodogram_results: Dict,
                         output_path: Path) -> Path:
    """
    Create a summary figure with all key visualizations.

    Parameters
    ----------
    tic_id : int
        TIC ID for title
    step1_results : dict
        Results from phase-folded fitting
    transit_results : list
        Individual transit results
    ephemeris_results : dict
        Ephemeris analysis results
    periodogram_results : dict
        Periodogram results
    output_path : Path
        Path to save plot

    Returns
    -------
    Path
        Path to saved plot
    """
    setup_plot_style()

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

    # 1. Phase-folded light curve (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    # Placeholder - would need actual data
    ax1.set_title('Phase-Folded Light Curve')
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Flux')
    ax1.text(0.5, 0.5, 'Phase-folded data', ha='center', va='center',
             transform=ax1.transAxes, fontsize=12, color=COLORS['gray'])

    # 2. O-C diagram (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'oc_linear' in ephemeris_results:
        oc_data = ephemeris_results['oc_linear']
        epochs = np.array(oc_data['epochs'])
        oc_min = np.array(oc_data['oc_minutes'])
        oc_err = np.array(oc_data['t_mid_err_minutes'])
        ax2.errorbar(epochs, oc_min, yerr=oc_err, fmt='o',
                     color=COLORS['primary'], markersize=3, capsize=2)
        ax2.axhline(y=0, color=COLORS['gray'], linestyle='--', alpha=0.5)
    ax2.set_title('O-C Diagram')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('O-C (minutes)')

    # 3. Periodogram (middle, spanning both columns)
    ax3 = fig.add_subplot(gs[1, :])
    if 'periodogram_data' in periodogram_results:
        pg_data = periodogram_results['periodogram_data']
        freqs = np.array(pg_data['frequencies'])
        power = np.array(pg_data['power'])
        if len(freqs) > 0:
            ax3.plot(freqs, power, color=COLORS['primary'], linewidth=0.8)
            fap_levels = periodogram_results.get('fap_levels', {})
            if '1%' in fap_levels:
                ax3.axhline(y=fap_levels['1%'], color=COLORS['error'],
                           linestyle='--', label='1% FAP')
    ax3.set_title('Lomb-Scargle Periodogram')
    ax3.set_xlabel('Frequency (1/day)')
    ax3.set_ylabel('Power')
    ax3.legend()

    # 4. Parameter summary (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')

    params = step1_results.get('parameters', {})
    param_text = "Fitted Parameters:\n"
    param_text += f"Period: {params.get('period', {}).get('value', 'N/A'):.8f} days\n"
    param_text += f"Rp/Rs: {params.get('rp_rs', {}).get('value', 'N/A'):.4f}\n"
    param_text += f"a/Rs: {params.get('a_rs', {}).get('value', 'N/A'):.2f}\n"
    param_text += f"b: {params.get('b', {}).get('value', 'N/A'):.3f}\n"

    eph = ephemeris_results.get('linear', {})
    param_text += f"\nEphemeris:\n"
    param_text += f"T0: {eph.get('t0', 'N/A')}\n"
    param_text += f"P: {eph.get('period', 'N/A'):.8f} days\n"

    ax4.text(0.1, 0.9, param_text, transform=ax4.transAxes,
             verticalalignment='top', fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    # 5. Statistics summary (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    n_transits = len([t for t in transit_results if t.get('success', False)])
    model_sel = ephemeris_results.get('model_selection', {})

    stats_text = "Analysis Summary:\n"
    stats_text += f"Transits fitted: {n_transits}\n"
    stats_text += f"Preferred model: {model_sel.get('preferred_model', 'N/A')}\n"
    stats_text += f"Delta BIC: {model_sel.get('delta_bic', 'N/A'):.2f}\n"

    linear_stats = ephemeris_results.get('linear', {})
    stats_text += f"O-C RMS: {linear_stats.get('oc_rms_min', 'N/A'):.2f} min\n"

    pg_stats = periodogram_results.get('peak_fap', 1.0)
    stats_text += f"\nTTV Detection:\n"
    stats_text += f"Peak FAP: {pg_stats:.2e}\n"
    stats_text += f"Significant (1%): {'Yes' if periodogram_results.get('is_significant_1pct', False) else 'No'}\n"

    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes,
             verticalalignment='top', fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    fig.suptitle(f'TIC {tic_id} - Transit Timing Analysis Summary', fontsize=16)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved summary figure to {output_path}")
    return output_path


if __name__ == "__main__":
    # Test plotting functions with synthetic data
    logging.basicConfig(level=logging.INFO)

    print("Testing plotting functions...")

    # Create test output directory
    test_dir = Path("/tmp/autottv_test_plots")
    test_dir.mkdir(exist_ok=True)

    # Test periodogram plot
    freqs = np.linspace(0.01, 1.0, 1000)
    power = np.random.exponential(0.5, len(freqs))
    power[500] = 10  # Add peak

    plot_periodogram(freqs, power, test_dir / "periodogram.png",
                    fap_01=3.0, fap_05=2.0, peak_freq=freqs[500])

    # Test O-C plot
    epochs = np.arange(0, 50, 2)
    oc = np.random.normal(0, 2, len(epochs))
    oc_err = np.ones_like(oc) * 0.5

    plot_oc_diagram(epochs, oc, oc_err, test_dir / "oc.png")

    print(f"Test plots saved to {test_dir}")
