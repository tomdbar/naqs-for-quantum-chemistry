import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec

from collections.abc import Iterable
# from src.optimizer.variational import LogKey
from src.optimizer.utils import LogKey


import pandas as pd

try:
    import seaborn as sns
    sns.set()
except:
    # seaborn not found
    pass

def load_vmc_log(log_path):
    ITERS = "Iteration"
    df_log = pd.read_pickle(log_path)
    log = {}
    for key in LogKey:
        log[key] = df_log[[ITERS, key]].dropna().values
    return log

def plot_training(vmc_log, molecule, window=None, print_summary=True):
    if type(vmc_log) is not dict:
        try:
            vmc_log = vmc_log.log
        except:
            raise Exception("plot_training : expects a log dictionary.")

    chem_acc = 1.6e-3
    cmap = plt.get_cmap("tab10")
    cols = {"energy": cmap(0), "local energy": cmap(1), "HF": "r", "chem acc": "g", "FCI": "b"}

    iters_energy, energy = np.array(vmc_log[LogKey.E]).T
    iters_local_energy, local_energy = np.array(vmc_log[LogKey.E_LOC]).T
    iters_local_energy_var, local_energy_var = np.array(vmc_log[LogKey.E_LOC_VAR]).T

    if window is not None:
        def _convole(data, window):
            avg_mask = np.ones(window) / window
            if isinstance(data, Iterable):
                return [np.convolve(d, avg_mask, 'valid') for d in data]
            else:
                return np.convolve(data, avg_mask, 'valid')
        iters_local_energy, local_energy, iters_local_energy_var, local_energy_var = \
            _convole([iters_local_energy, local_energy, iters_local_energy_var, local_energy_var], window)

    def _plot_energy_convergence(ax):
        mask = np.ones((50,)) / 50
        ax.plot(np.convolve(iters_local_energy, mask, mode='valid'),
                np.convolve(local_energy, mask, mode='valid'),
                label=r"$\langle E_\mathrm{loc} \rangle$", color=cols["local energy"], linewidth=0.75, ls='--')
        # ax.plot(iters_local_energy, local_energy, label=r"$\langle E_\mathrm{loc} \rangle$", color=cols["local energy"], linewidth=0.75, ls='--')
        ax.plot(iters_energy, energy, label=r"$E$", color=cols["energy"], linewidth=1)
        ax.set_ylabel("Energy (Hartree)")
        ax.set_xlabel("Iterations")

        e_mid = (molecule.hf_energy + molecule.fci_energy) / 2
        e_range = 1.05 * (molecule.hf_energy - molecule.fci_energy)

        ax.set_ylim(top=e_mid + e_range / 2, bottom=e_mid - e_range / 2)

        ax.axhline(molecule.hf_energy, 0, 1, ls='--', color=cols["HF"], linewidth=0.75, label="Hartree-Fock")
        ax.axhline(molecule.fci_energy + chem_acc, 0, 1, ls='--', color=cols["chem acc"], linewidth=0.75, label="Chem. Acc.")
        ax.axhline(molecule.fci_energy, 0, 1, ls='--', color=cols["FCI"], linewidth=0.75, label="FCI")

        ax.legend(loc="upper right", bbox_to_anchor=(1, 1, 0., 0.))

        return ax

    def _plot_energy_error(ax):
        e_min = molecule.fci_energy

        ax.plot(iters_local_energy, local_energy - e_min, linewidth=0.75, color=cols["local energy"], ls='--')
        if iters_energy[0] is not None:
            ax.plot(iters_energy, energy - e_min, color=cols["energy"], linewidth=1)
        ax.set_ylabel("E err. (H)")
        #     ax.set_xlabel("Iterations")
        ax.set_yscale("log")

        ax.axhline(molecule.hf_energy - e_min, 0, 1, ls='--', color=cols["HF"], linewidth=0.75, label="Hartree-Fock")
        ax.axhline(molecule.fci_energy + chem_acc - e_min, 0, 1, ls='--', color=cols["chem acc"], linewidth=0.75, label="Chem. Acc.")
        ax.axhline(molecule.fci_energy - e_min, 0, 1, ls='--', color=cols["FCI"], linewidth=0.75, label="FCI")

        return ax

    def _plot_local_energy_var(ax):
        ax.plot(iters_local_energy_var, local_energy_var, color=cols["local energy"], linewidth=1)
        ax.set_ylabel(r"$\sigma^2(E_\mathrm{loc})$")
        ax.set_xlabel("Iterations")
        ax.set_yscale("log")

        return ax

    with plt.style.context('seaborn-paper', after_reset=True):
        fig = plt.figure(figsize=(6, 2.5), constrained_layout=True)
        gs = gridspec.GridSpec(ncols=5, nrows=2, figure=fig)

        ax1 = fig.add_subplot(gs[:, :3])
        try:
            ax1 = _plot_energy_convergence(ax1)
        except:
            pass

        ax2 = fig.add_subplot(gs[0, 3:])
        try:
            ax2 = _plot_energy_error(ax2)
        except:
            pass

        ax3 = fig.add_subplot(gs[1, 3:])
        try:
            ax3 = _plot_local_energy_var(ax3)
        except:
            pass

    if print_summary:
        E_min = np.min(energy)
        print(f"Minimum energy : {E_min:.5f} Hartree")

        print(f'\tBelow Hartree-Fock ({molecule.hf_energy:.5f} Hartree) : {E_min < molecule.hf_energy}')
        print(f'\tBelow CCSD ({molecule.ccsd_energy:.5f} Hartree) : {E_min < molecule.ccsd_energy}')
        print(f'\tBelow FCI ({molecule.fci_energy:.5f} Hartree) : {E_min < molecule.fci_energy}')

        if molecule.fci_energy + chem_acc > E_min:
            print(f"\nChemical accuracy achieved!\n\tNAQS energy : {np.min(energy):.4f} < {(molecule.fci_energy + chem_acc):.4f}")
        else:
            print(f"Not reaching chemical accuracy...\n\tNAQS energy : {np.min(energy):.4f} >= {(molecule.fci_energy + chem_acc):.4f}")

    return fig

def plot_wavefunction(wavefunction, hilbert_args=None, n_states=None, log_scale=True):
    if hilbert_args is None:
        states, states_idx = wavefunction.hilbert.get_basis(ret_idxs=True)
    else:
        states, states_idx = wavefunction.hilbert.get_subspace(*hilbert_args, ret_idxs=True)
    amps = wavefunction.amplitude(states).detach()
    phase = wavefunction.phase(states).detach()
    probs = amps.pow(2)

    if n_states is None:
        n_states = 2**wavefunction.hilbert.N
    x_idxs = np.arange(n_states)
    state_idxs = states_idx.numpy()
    plot_idxs = np.argsort(probs)[-n_states:]

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))

    ax0.bar(x_idxs, probs[plot_idxs])
    ax0.set_xlabel("State idx.")
    ax0.set_ylabel("Prob.")
    if log_scale:
        ax0.set_yscale("log")

    ax1.bar(x_idxs, phase[plot_idxs] / np.pi)
    ax1.set_xlabel("State idx.")
    ax1.set_ylabel("Phase (/Pi).")

    for ax in [ax0, ax1]:
        ax.set_xticks(x_idxs)
        ax.set_xticklabels(state_idxs[plot_idxs])

    fig.tight_layout()

    return fig