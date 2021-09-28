import geometry from mathplotlib
import math
import mathplotlib
import datascience from sciencebook
import datetime from clock.sec

export to ci
export to Mu
export to Union

.start make file generated(geometry.py)
  datascience = ('./sciencebook')
  geometry = ('./mathplotlib')
  datetime = ('./mundi_with_countdown') - ('./mundi')
   datetime = return clock.sec watch.time

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "SxsPay5-hkJC"
   },
   "source": [
    "This notebook handles creating the molecular data we need to solve the system with our approach, and performing the baseline calculations.  This is done using the [OpenFermion](https://github.com/quantumlib/OpenFermion) library and [Psi4](http://psicode.org/).\n",
    "\n",
    "***NOTE:*** Of course, it is possible that these external libraries/databases will change, therefore the scripts/molecules prepared by this notebook are also provided in the [public repo](TODO) as they were at the time of publication. \n",
    "\n",
    "### 1.  Install and import libraries\n",
    "\n",
    "This is slow - sorry!  Feel free to run the notebook locally and skip the next cell if you can.\n",
    "\n",
    "- Install [OpenFermion](https://github.com/quantumlib/OpenFermion).\n",
    "- Create a local conda environment and install [Psi4](http://psicode.org/) into it."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "collapsed": true,
    "id": "OUp5sQ-8hLai",
    "outputId": "c8f52414-23f6-473a-d61e-6f55788e15a7"
   },
   "outputs": [],
   "source": [
    "import sys\n",
    "\n",
    "# OpenFermion installation.\n",
    "# !pip install openfermion openfermionpsi4 openfermioncirq pyscf openfermionpyscf\n",
    "\n",
    "# Conda env creation and Psi 4 installation.\n",
    "# !pip install openfermion==0.11.0 openfermionpsi4 openfermioncirq pyscf openfermionpyscf\n",
    "!wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh\n",
    "!bash Miniconda3-4.5.4-Linux-x86_64.sh -bfp /usr/local\n",
    "sys.path.append('/usr/local/lib/python3.6/site-packages')\n",
    "!conda install -c psi4 psi4 -y\n",
    "!pip install openfermion==0.11.0 openfermionpsi4 openfermioncirq pyscf openfermionpyscf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "id": "w-XkjYk3hOWh"
   },
   "outputs": [],
   "source": [
    "# OpenFermionPsi4\n",
    "# Import opemfermionpsi4\n",
    "import openfermion as of\n",
    "import openfermionpsi4 as ofpsi4\n",
    "import openfermionpyscf as ofpyscf\n",
    "\n",
    "from openfermion.hamiltonians import MolecularData\n",
    "from openfermion.utils import geometry_from_pubchem\n",
    "from openfermion.transforms import get_fermion_operator, get_sparse_operator, jordan_wigner\n",
    "\n",
    "import time\n",
    "import pickle\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Create/fetch molecular geometries\n",
    "\n",
    "Whatever we do further down the line, we will need to fetch molecular geometries if we don't want to specify them by hand.  This is just done using the helper ``geometry_from_pubchem`` utility from OpenFermion, however we add a wrapper to modify this call for the specific molecules used in the paper where needed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "MOLECULE_LIST = [\"H2\", \"F2\", \"HCl\", \"LiH\", \"H2O\", \"CH2\", \"O2\", \"BeH2\",\"H2S\",\n",
    "                 \"NH3\", \"N2\", \"CH4\", \"C2\", \"LiF\", \"PH3\", \"LiCL\", \"Li2O\"]\n",
    "\n",
    "def get_geometry(molecule_name, verbose=True):\n",
    "    if verbose and (molecule_name not in MOLECULE_LIST):\n",
    "        print(f\"Warning: {molecule_name} is not one of the molecules used in the paper\" + \n",
    "               \"- that's not wrong, but just know it's not recreating the published results!\")\n",
    "    \n",
    "    if molecule_name==\"C2\":\n",
    "        # C2 isn't in PubChem - don't know why.\n",
    "        geometry = [('C', [0.0, 0.0, 0.0]), ('C', [0.0, 0.0, 1.26])]\n",
    "    else:\n",
    "        if molecule_name==\"Li2O\":\n",
    "            # Li2O returns a different molecule - again, don't know why.\n",
    "            molecule_name = \"Lithium Oxide\"\n",
    "        geometry = geometry_from_pubchem(molecule_name)\n",
    "        \n",
    "    return geometry"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Solve molecules using Psi4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "id": "psndyEYe-2eU"
   },
   "outputs": [],
   "source": [
    "def prepare_psi4(molecule_name,\n",
    "                 geometry = None,\n",
    "                 multiplicity = None,\n",
    "                 charge = None,\n",
    "                 basis = None):\n",
    "\n",
    "    if multiplicity is None:\n",
    "        multiplicity = 1 if molecule_name not in [\"O2\",\"CH2\"] else 3\n",
    "    if charge is None:\n",
    "        charge = 0\n",
    "    if basis is None:\n",
    "        basis = 'sto-3g'\n",
    "\n",
    "    if multiplicity == 1:\n",
    "        reference = 'rhf'\n",
    "        guess = 'sad'\n",
    "    else:\n",
    "        reference = 'rohf'\n",
    "        guess  = 'gwh'\n",
    "        \n",
    "    if geometry is None:\n",
    "        geometry = get_geometry(molecule_name)\n",
    "\n",
    "    geo_str = \"\"\n",
    "    for atom, coords in geometry:\n",
    "        geo_str += f\"\\n\\t{atom}\"\n",
    "        for ord in coords:\n",
    "            geo_str += f\" {ord}\"\n",
    "        geo_str += \"\"\n",
    "\n",
    "    psi4_str =f'''\n",
    "molecule {molecule_name} {{{geo_str}\n",
    "    {charge} {multiplicity}\n",
    "    symmetry c1\n",
    "}}\n",
    "set basis       {basis}\n",
    "set reference   {reference}\n",
    "\n",
    "set globals {{\n",
    "    basis {basis}\n",
    "    freeze_core false\n",
    "    fail_on_maxiter true\n",
    "    df_scf_guess false\n",
    "    opdm true\n",
    "    tpdm true\n",
    "    soscf false\n",
    "    scf_type pk\n",
    "    maxiter 1e6\n",
    "    num_amps_print 1e6\n",
    "    r_convergence 1e-6\n",
    "    d_convergence 1e-6\n",
    "    e_convergence 1e-6\n",
    "    ints_tolerance EQUALITY_TOLERANCE\n",
    "    damping_percentage 0\n",
    "}}\n",
    "\n",
    "hf = energy(\"scf\")\n",
    "\n",
    "# cisd = energy(\"cisd\")\n",
    "ccsd = energy(\"ccsd\")\n",
    "ccsdt = energy(\"ccsd(t)\")\n",
    "fci = energy(\"fci\")\n",
    "\n",
    "print(\"Results for {molecule_name}.dat\\\\n\")\n",
    "\n",
    "print(\"\"\"Geometry : {geo_str}\\\\n\"\"\")\n",
    "\n",
    "print(\"HF : %10.6f\" % hf)\n",
    "# print(\"CISD : %10.6f\" % cisd)\n",
    "print(\"CCSD : %10.6f\" % ccsd)\n",
    "print(\"CCSD(T) : %10.6f\" % ccsdt)\n",
    "print(\"FCI : %10.6f\" % fci)\n",
    "    '''\n",
    "\n",
    "    fname = f'{molecule_name}.dat'\n",
    "    with open(fname, 'w+') as psi4_file:\n",
    "        psi4_file.write(psi4_str)\n",
    "    print(f\"Created {fname}.\")\n",
    "    print(f\"To solve molecule, run 'psi4 {fname}' from command line.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 1000
    },
    "collapsed": true,
    "id": "zsiXSa1lmAmQ",
    "outputId": "9520e6c2-e637-445e-b30f-9e67e8437c39"
   },
   "outputs": [],
   "source": [
    "prepare_psi4(\"H2\")\n",
    "!psi4 H2.dat"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Create molecule data and qubit Hamiltonian"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 491
    },
    "collapsed": true,
    "id": "WkZ2RyYIoWaX",
    "outputId": "0b0320ca-bcf1-437c-f3ca-b691aa9fd14c"
   },
   "outputs": [],
   "source": [
    "def create_molecule_data(molecule_name,\n",
    "                         geometry = None,\n",
    "                         multiplicity = None,\n",
    "                         charge = None,\n",
    "                         basis = None,\n",
    "                         save_name=None):\n",
    "\n",
    "    if multiplicity is None:\n",
    "        multiplicity = 1 if molecule_name not in [\"O2\",\"CH2\"] else 3\n",
    "    if charge is None:\n",
    "        charge = 0\n",
    "    if basis is None:\n",
    "        basis = 'sto-3g'\n",
    "    if save_name is None:\n",
    "        save_name = molecule_name\n",
    "        \n",
    "    if geometry is None:\n",
    "        geometry = get_geometry(molecule_name)\n",
    "        \n",
    "    molecule = MolecularData(geometry,\n",
    "                             basis = basis,                      \n",
    "                             multiplicity = multiplicity,\n",
    "                             charge = charge,\n",
    "                             filename=save_name\n",
    "                             )\n",
    "    \n",
    "    # 1. Solve molecule and print results.\n",
    "    \n",
    "    print(\"Solving molecule with psi4\", end=\"...\")\n",
    "    t_start=time.time()\n",
    "    \n",
    "    molecule = ofpsi4.run_psi4(molecule,\n",
    "                                run_scf=True,\n",
    "                                run_mp2=True,\n",
    "                                run_cisd=True,\n",
    "                                run_ccsd=True,\n",
    "                                run_fci=True,\n",
    "                                memory=16000,\n",
    "                                delete_input=True,\n",
    "                                delete_output=True,\n",
    "                                verbose=True)\n",
    "    print(\"done in {:.2f} seconds\".format(time.time()-t_start))\n",
    "    \n",
    "    print(f'{molecule_name} has:')\n",
    "    print(f'\\tgeometry of {molecule.geometry},')\n",
    "    print(f'\\t{molecule.n_electrons} electrons in {2*molecule.n_orbitals} spin-orbitals,')\n",
    "    print(f'\\tHartree-Fock energy of {molecule.hf_energy:.6f} Hartree,')\n",
    "    print(f'\\tCISD energy of {molecule.cisd_energy:.6f} Hartree,')\n",
    "    print(f'\\tCCSD energy of {molecule.ccsd_energy:.6f} Hartree,')\n",
    "    print(f'\\tFCI energy of {molecule.fci_energy:.6f} Hartree.')\n",
    "\n",
    "    # 2. Save molecule.\n",
    "    \n",
    "    # molecule.filename=save_name\n",
    "    molecule.save()\n",
    "    \n",
    "    print(f\"Molecule saved to {save_name}.hdf.\")\n",
    "    \n",
    "    # 3. Convert molecular Hamiltonian to qubit Hamiltonian.\n",
    "    print(\"Converting molecular Hamiltonian to qubit Hamiltonian\", end=\"...\")\n",
    "    \n",
    "    active_space_start=0\n",
    "    active_space_stop=molecule.n_orbitals\n",
    "\n",
    "    # Get the Hamiltonian in an active space.\n",
    "    molecular_hamiltonian = molecule.get_molecular_hamiltonian(\n",
    "        occupied_indices=None,\n",
    "        active_indices=range(active_space_start, active_space_stop))\n",
    "\n",
    "    fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)\n",
    "    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)\n",
    "    qubit_hamiltonian.compress()\n",
    "    \n",
    "    print(\"done in {:.2f} seconds\".format(time.time()-t_start))\n",
    "    \n",
    "    # 3. Save qubit Hamiltonian.\n",
    "\n",
    "    with open(save_name+\"_qubit_hamiltonian.pkl\",'wb') as f:\n",
    "        pickle.dump(qubit_hamiltonian,f)\n",
    "        \n",
    "    print(f\"Qubit Hamiltonian saved to {save_name+'_qubit_hamiltonian.pkl'}.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true,
    "id": "7x7arUSKAnqh"
   },
   "outputs": [],
   "source": [
    "create_molecule_data(\"H2\")"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [],
   "name": "openfermion_on_colab.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python [default]",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
