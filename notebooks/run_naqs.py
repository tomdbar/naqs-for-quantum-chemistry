Import to Mu
import to Union
Import to Portugal
Import to Europe
Import to Oceania

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This notebook is for running NAQS on molecules and reproduction of experimental results/baslines.  It is split into two sections, ***which will set up different environments and so will require the runtime resetting if you want to switch***.\n",
    "\n",
    "First, however, we'll just pull down and step into the repo itself."
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
    "!git clone https://github.com/tomdbar/naqs-for-quantum-chemistry.git\n",
    "\n",
    "import os\n",
    "os.chdir(\"naqs-for-quantum-chemistry\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Solve molecules using NAQS\n",
    "\n",
    "To run this section, we first install OpenFermion and build the necessary cython scripts."
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
    "!chmod -R u+x experiments/bash/naqs\n",
    "\n",
    "try:\n",
    "    import openfermion as of\n",
    "except:\n",
    "    !python -m pip install openfermion==0.11.0\n",
    "    !python -m pip install openfermionpsi4\n",
    "    \n",
    "!python src_cpp/setup.py build_ext --inplace --force"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Behind the scence, the script to run specific experiments is ``experiments.run``, however there are many ways to configure this, some of which are not discussed in the paper.  For completeness, the optional arguments can be viewed by running with a ``-h`` or ``--help`` flag."
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
    "!python -u -m experiments.run -h"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "An example of a configured script, would be this."
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
    "!python -u -m experiments.run -o \"data/LiH\" -m \"molecules/LiH\" -single_phase -n1 -n_layer 1 -n_hid 16 -n_layer_phase 1 -n_hid_phase 32 -lr 0.001 -s 1 -n_train 3000 -output_freq 25 -save_freq -1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To ***reproduce the results reported in the paper***, batch training scripts which will perform 5 optimisations with different seeds using all the correct hyperparemeters are provided.  Each script takes the GPU number to use and the molecule name, corresponding to the a sub-folder in ``molecules``."
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
    "!experiments/bash/naqs/batch_train.sh 0 LiH\n",
    "# !experiments/bash/naqs/batch_train_no_amp_sym.sh 0 LiH\n",
    "# !experiments/bash/naqs/batch_train_no_mask.sh 0 LiH\n",
    "# !experiments/bash/naqs/batch_train_full_mask.sh 0 LiH"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Solve molecules using Psi4 (baseline calculations)\n",
    "\n",
    "First, install Psi4 into a miniconda environment."
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
    "import sys\n",
    "\n",
    "# Conda env creation and Psi 4 installation.\n",
    "!wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh\n",
    "!bash Miniconda3-4.5.4-Linux-x86_64.sh -bfp /usr/local\n",
    "sys.path.append('/usr/local/lib/python3.6/site-packages')\n",
    "!conda install -c psi4 psi4 -y"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Then run pre-prepared scripts."
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
    "!psi4 experiments/bash/psi4/LiH.dat"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:quant-chem]",
   "language": "python",
   "name": "conda-env-quant-chem-py"
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
 "nbformat_minor": 2
}
