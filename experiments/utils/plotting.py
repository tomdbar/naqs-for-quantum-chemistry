import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

from src.optimizer.utils import LogKey

try:
    import seaborn as sns
    sns.set()
    print("Set seaborn plotting defaults.")
except:
    print("seaborn not found.")

def plot_vmc(vmc, molecule, title=""):
    chem_acc = 1.6e-3

    energy_log = vmc.log[LogKey.E]
    local_energy_log = vmc.log[LogKey.E_LOC]

    iters_energy = np.array([i for i, e in energy_log])
    energy = np.array([e for i, e in energy_log])
    iters_local_energy = np.array([i for i, e in local_energy_log])
    local_energy = np.array([e for i, e in local_energy_log])

    print(f"Minimum energy : {min(energy):.5f} Hartree")

    offset = 0

    with plt.style.context('seaborn-paper', after_reset=True):
        fig, ax1 = plt.subplots(figsize=(4, 2.5))

        zero_level = 1.0001 * molecule.fci_energy

        ax1.plot(iters_energy, abs(energy - zero_level), label = "E", linewidth=1)
        ax1.plot(iters_local_energy, abs(local_energy - zero_level), label = "E_loc", linewidth=0.75, ls='--')
        #         ax2.set_xlabel("Iter")
        ax1.set_ylabel("<E> (arb.)")
        # ax2.set_yscale("log")

        ax1.set_ylim(top=0.9999 * molecule.hf_energy - zero_level, bottom=0)
        ax1.axhline(molecule.hf_energy - zero_level, 0, 1, ls='--', color='r', linewidth=0.75, label="Hartree-Fock")
        ax1.axhline(molecule.fci_energy - zero_level, 0, 1, ls='--', color='k', linewidth=0.75, label="FCI")
        ax1.axhline(molecule.fci_energy + chem_acc - zero_level, 0, 1, ls='--', color='g', linewidth=0.75,
                    label="Chem. Acc.")

        ax1.legend(loc="center right", bbox_to_anchor=(0.4, 0.2, 0., 0.))

        ax1.set_title(title)

        ax2 = plt.axes([0, 0, 1, 1])
        # Manually set the position and relative size of the inset axes within ax1
        ip = InsetPosition(ax1, [0.625, 0.625, 0.35, 0.35])
        ax2.set_axes_locator(ip)

        ax2.plot(iters_energy, energy + offset, label="NAQS")

        ax2.set_xlabel("Iter")
        ax2.set_ylabel("<Energy>")

        ax2.axhline(molecule.hf_energy + offset, 0, 1, ls='--', color='r', linewidth=0.75, label="Hartree-Fock")
        ax2.axhline(molecule.fci_energy + offset, 0, 1, ls='--', color='k', linewidth=0.75, label="FCI")
        ax2.axhline(molecule.fci_energy + chem_acc + offset, 0, 1, ls='--', color='g', linewidth=0.75,
                    label="Chem. Acc.")

        ax2.set_ylim(1.025 * molecule.fci_energy + offset, np.max(energy) + offset)

    return fig

