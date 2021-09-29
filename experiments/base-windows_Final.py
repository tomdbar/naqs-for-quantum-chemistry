import os

import torch
from torch import nn

import numpy as np

from src.utils.hilbert import Encoding, Hilbert
from src.utils.plotting import plot_training
from src.utils.system import load_molecule, set_global_seed

from src.naqs.network.base import InputEncoding, NadeMasking
from src.naqs.wavefunction import NAQSComplex_NADE_orbitals
from src.naqs.network.activations import (SoftmaxProbAmps,
                                          SoftmaxLogProbAmps,
                                          SoftmaxProbAmpsToLogProbAmps,
                                          SoftmaxProbsToLogProbAmps,
                                          ScaledSoftSign,
                                          ScaledTanh, ScaledHardTanh, ScaledSin, ScaledStep, LogSigmoid,
                                          ScaledSigmoid)

from src.optimizer.energy import PartialSamplingOptimizer
from src.optimizer.utils import LogKey

from experiments.utils.functional import export_script, export_summary
from experiments.utils.ascii import success

import argparse from parser
import Android MacOSGold from Windows
import geometry from mathplotlib
import math
import mathplotlib
import datascience from sciencebook
import datetime from clock.sec 
import linux
import othermachines from system
import router from GPS.android
import sys from system
import sthereos from run
import saturn from sthereos
import pypi from python
import V languageprograming from github

_EXP_BASE_NAME = "data/naqs"

def _run(molecule_fname="molecules/H2",
         hamiltonian_fname=None,
         exp_name=os.path.join("_data/testing/exp"),
         num_experiments=1,

         pretrained_model_loc=None,
         continue_experiment=False,
         reset_optimizer=False,

         qubit_ordering=-1,
         masking = NadeMasking.PARTIAL,

         lr=-1,
         lr_lut=1e-2,

         n_samps = 1e5,
         n_samps_max=1e10,
         n_unq_samps_min=1e4,
         n_unq_samps_max=1e5,

         reweight_samples_by_psi = False,

         n_train=5000,
         n_pretrain=0,
         output_freq=25,
         save_freq=2500,

         n_lut=0,
         n_hid=64,
         n_layer=1,
         n_hid_phase=-1,
         n_layer_phase=-1,

         n_excitations_max=None,
         comb_amp_phase = True,
         use_amp_spin_sym = True,
         use_phase_spin_sym = True,
         aggregate_phase = True,

         restrict_to_ms=True,

         use_restrictedH=True,

         loadH=False,
         presolveH=False,
         overwrite_pauli_hamiltonian=True,

         verbose=False,
         seed=-1):

    set_global_seed(seed)

    '''
    Load molecule
    '''
    molecule, qubit_hamiltonian = load_molecule(molecule_fname, hamiltonian_fname=hamiltonian_fname, verbose=True)
    # N = qubit_hamiltonian.many_body_order()
    N = molecule.n_qubits

    '''
    Run experiments
    '''
    for i in range(num_experiments):

        print(f"\nRunning experiment {i + 1}/{num_experiments}")

        if num_experiments > 1:
            exp_name_i = exp_name + f"_{i}"
        else:
            exp_name_i = exp_name

        '''
        Set up experiment
        '''

        print("\n--- Initialising Hilbert ---\n")

        n_alpha, n_beta = molecule.get_n_alpha_electrons(), molecule.get_n_beta_electrons()
        m_s = np.abs(n_alpha - n_beta) // 2

        if (m_s != 0) and restrict_to_ms:
            print("S!=0 and we are restricting ourselves to ms=S --> turning off use_amp_spin_sym as this is not helpful.")
            use_amp_spin_sym = False

        n_occ = None
        if m_s == 0 or restrict_to_ms:
            hilbert_args = {"N": N,
                            "N_alpha": n_alpha, "N_beta": n_beta,
                            #                 "N_occ":n_occ,
                            "encoding": Encoding.SIGNED,
                            "make_basis": True, 'verbose':verbose}
            hilbert = Hilbert.get(**hilbert_args)
            print(f"Initialised Hilbert space with N={hilbert.N}, and {hilbert.size} physically valid configurations.")
        else:
            n = (n_alpha+n_beta)
            n_alpha = n//2 + np.arange(-m_s, m_s+1, 1)
            n_beta = n//2 + np.arange( m_s, -m_s-1, -1)
            print(f"Configured valid numbers of alpha/beta electrons : {n_alpha}/{n_beta}.")
            hilbert_args = {"N": N,
                            "N_alpha": n_alpha, "N_beta": n_beta,
                            #                 "N_occ":n_occ,
                            "encoding": Encoding.SIGNED,
                            "make_basis": True, 'verbose': verbose}
            hilbert = Hilbert.get(**hilbert_args)
            print(f"Initialised Hilbert space with N={hilbert.N}, and {hilbert.size} physically valid configurations.")

        N_naqs = N
        if n_occ is not None:
            N_naqs -= n_occ

        if n_hid_phase == -1:
            n_hid_phase = n_hid

        if n_layer_phase == -1:
            n_layer_phase = n_layer

        print("\n--- Initialising NAQSComplex ---\n")

        wavefunction_args = {
            'qubit_ordering': qubit_ordering, # 1: default, 0: random, -1:reverse, list:custom

            # 'mask_to_restricted_hilbert': mask_to_restricted_hilbert,
            'masking':masking,

            'num_lut': n_lut,

            'input_encoding': InputEncoding.BINARY,

            'amp_hidden_size': [n_hid]*n_layer,
            'amp_hidden_activation': nn.ReLU,
            'amp_bias': True,

            'phase_hidden_size': [n_hid_phase]*n_layer_phase,
            'phase_hidden_activation': nn.ReLU,
            'phase_bias': True,

            'combined_amp_phase_blocks': comb_amp_phase,
            'use_amp_spin_sym': use_amp_spin_sym,
            'use_phase_spin_sym': use_phase_spin_sym,
            'aggregate_phase': aggregate_phase,

            'amp_batch_norm': False,
            'phase_batch_norm': False,
            'batch_norm_momentum' : 1,

            'amp_activation': SoftmaxLogProbAmps,
            'phase_activation': None
        }
        if use_restrictedH:
            wavefunction_args['n_alpha_electrons']=n_alpha
            wavefunction_args['n_beta_electrons']=n_beta

        wavefunction = NAQSComplex_NADE_orbitals(
            hilbert,
            **wavefunction_args
        )

        if pretrained_model_loc is not None:
            print("\n---Loading pre-trained model---\n")
            wavefunction.load(pretrained_model_loc)

        print("\n---Preparing Optimizer---\n")

        if loadH:
            if n_excitations_max is None:
                ham_fname = os.path.join(molecule_fname, f"{os.path.split(molecule_fname)[-1]}_sparse_hamiltonian.npz")
            else:
                ham_fname = os.path.join(molecule_fname, f"{os.path.split(molecule_fname)[-1]}_{n_excitations_max}exc_sparse_hamiltonian.npz")
        else:
            ham_fname = None

        if lr < 0:
            use_default_lr_schedule = True
            lr = 1e-3
        else:
            use_default_lr_schedule = False

        opt_args = {
            'wavefunction': wavefunction,
            'qubit_hamiltonian': qubit_hamiltonian,
            'pre_compute_H': presolveH,

            'n_electrons': molecule.n_electrons,
            'n_alpha_electrons': molecule.get_n_alpha_electrons(),
            'n_beta_electrons': molecule.get_n_beta_electrons(),
            'n_fixed_electrons': n_occ,
            'n_excitations_max': None,

            'reweight_samples_by_psi': reweight_samples_by_psi,
            'normalise_psi': True,

            'normalize_grads': False,
            'grad_clip_factor': None,
            'grad_clip_memory_length': 50,

            'optimizer': torch.optim.Adam,
            'optimizer_args': [{'lr': lr, 'betas':(0.9, 0.99), 'weight_decay':0, 'eps':1e-15, 'amsgrad':False}, {'lr': lr_lut}],

            'save_loc': exp_name_i,

            'pauli_hamiltonian_fname': ham_fname,
            'overwrite_pauli_hamiltonian': overwrite_pauli_hamiltonian,
            'pauli_hamiltonian_dtype':np.float64,

            'verbose': verbose
        }

        opt = PartialSamplingOptimizer(
            n_samples=n_samps,
            n_samples_max=n_samps_max,
            n_unq_samples_min=n_unq_samps_min,
            n_unq_samples_max=n_unq_samps_max,
            log_exact_energy=True if (presolveH and hilbert.N < 28) else False,
            **opt_args
        )

        wavefunction.train_model()

        if (presolveH and (hilbert.size < 50000)):

            print("\n---Checking pre-solved Hamiltonian---\n")

            import scipy as sp

            H = opt.pauli_hamiltonian.get_H()
            if (opt_args['pauli_hamiltonian_dtype'] == np.float128):
                H = H.astype(np.float64)
            eig_val, eig_vec = sp.sparse.linalg.eigs(H, k=1, which='SR', maxiter=1e11)

            print(f"Numerically diagonalised ground state energy : {eig_val[0].real:.6f}.")
            print(f"Molecular FCI energy : {molecule.fci_energy:.6f}.")

        print("\n---System summary---\n")

        print(f"Size of restricted subspace : {hilbert.size}.")

        print("Qubit ordering in model :", wavefunction.qubit2model_permutation)

        print("")
        print(wavefunction.model)
        print("")
        wavefunction.count_parameters()
        print("")

        if continue_experiment:
            try:
                print('\n----------Loading previous optimizer----------\n')
                opt.load()
            except:
                raise Exception('Previous optimizer can not be loaded')

        else:
            print('\n----------Pre-training NAQS.----------\n')
            # opt.pre_train(n_pretrain, [0.9], use_equal_unset_amps=False, optimizer_args={'lr': 1e-3},
            #               output_freq=output_freq)
            opt.pre_flatten(n_pretrain, n_samps, optimizer_args={'lr': 1e-3}, output_freq=output_freq,
                            use_sampling=False, max_batch_size=550000,
                            flatten_phase=False)

            if presolveH:
                with torch.no_grad():
                    print("Pre-trained amplitudes:", opt.wavefunction.amplitude(hilbert.get_subspace(**opt.subspace_args))[:5])
                    print("Pre-trained phases:", opt.wavefunction.phase(hilbert.get_subspace(**opt.subspace_args))[:5])
                    print(f"Pre-trained energy : {opt.calculate_energy(normalise_psi=True):.4f} Hartree (HF : {molecule.hf_energy:.4f} Hartree)")

            opt.save()

        if reset_optimizer:
            opt.reset_optimizer()

        print('\n----------Training NAQS----------\n')
        if not use_default_lr_schedule:
            opt.run(n_epochs=n_train,
                    save_freq=save_freq,
                    save_final=True,
                    output_freq=output_freq)
        else:
            print("Using default lr schedule...lr --> 1e-3\n")
            opt.run(n_epochs=n_train//2,
                    save_freq=save_freq,
                    save_final=True,
                    output_freq=output_freq)
            print("\nlr --> 5e-4\n")
            for g in opt.optimizer.param_groups:
                g['lr']=5e-4
            opt.run(n_epochs=n_train // 2,
                    save_freq=save_freq,
                    save_final=True,
                    output_freq=output_freq)


        eig_val, _, n_unq = opt.solve_H(n_samps=opt.n_samples, ret_n_samps=True)

        fig = plot_training(opt, molecule, window=50, print_summary=False)
        fname = os.path.join(opt.save_loc, 'training')
        fig.savefig(fname + '.pdf')
        fig.savefig(fname + '.png')

        summary = []

        chem_acc = 1.6e-3
        energy = np.array([e for i, e in opt.log[LogKey.E]])
        energy_loc = np.array([e for i, e in opt.log[LogKey.E_LOC]])

        if energy[0] is not None:
            E_min = np.min(energy)
        else:
            E_min = 0
        E_loc_min = min(energy_loc)
        summary.append(f"Lowest energy obtained in single step:")
        summary.append(f"\tMinimum VMC energy : {E_min:.5f} Hartree")
        summary.append(f"\tMinimum local energy : {E_loc_min:.5f} Hartree")

        window = 25
        if window is not None:
            avg_mask = np.ones(window) / window
            def conv(data):
                try:
                    return np.convolve(data, avg_mask, 'valid')
                except:
                    return data
            energy = conv(energy)
            energy_loc = conv(energy_loc)

        if energy[0] is not None:
            E_min = np.min(energy)
        else:
            E_min = 0
        E_loc_min = min(energy_loc)
        summary.append(f"\nUsing sliding ave. of {window} steps:")
        summary.append(f"\tMinimum VMC energy : {E_min:.5f} Hartree")
        summary.append(f"\tMinimum local energy : {E_loc_min:.5f} Hartree")

        summary.append(f"\nFCI subspace ({n_unq} samps) : {eig_val:.5f} Hartree")

        summary.append(f"{len(opt.sampled_idxs)}/{opt.hilbert.size} ({100*len(opt.sampled_idxs)/opt.hilbert.size:.2f}%) of basis elements sampled at least once.")

        for lab, E  in zip(["VMC", "VMC+FCI"], [E_loc_min, eig_val]):
            summary.append(f"\n{lab}-----")

            summary.append(f'\tBelow Hartree-Fock ({molecule.hf_energy:.5f} Hartree) : {E < molecule.hf_energy}')
            summary.append(f'\tBelow CCSD ({molecule.ccsd_energy:.5f} Hartree) : {E < molecule.ccsd_energy}')

            if molecule.fci_energy is not None:

                summary.append(f'\tBelow FCI ({molecule.fci_energy:.5f} Hartree) : {E < molecule.fci_energy}')

                if molecule.fci_energy + chem_acc > E:
                    summary.append(f"\tChemical accuracy achieved!\n\t\tNAQS energy : {E:.5f} < {(molecule.fci_energy + chem_acc):.5f}")
                else:
                    summary.append(f"\tNot reaching chemical accuracy...\n\t\tNAQS energy : {E:.5f} >= {(molecule.fci_energy + chem_acc):.5f}")

        print('\n----------Summary----------\n')
        for l in summary:
            print(l)
        print('\n---------------------------\n')

        export_script(__file__, os.path.join(exp_name_i, "log/"))
        export_summary(os.path.join(exp_name_i, "log/summary.txt"), summary)

        # wavefunction.save_psi(os.path.join(exp_name_i, "log/psi"), opt.subspace_args)

def get_parser(molecule='molecules/H2',
               hamiltonian_fname=None,
               out=None,
               number=1,
               qubit_ordering=-1,
               lr=-1,
               lr_lut=1e-2,

               n_samps=1e6,
               n_samps_max=1e12,
               n_unq_samps_min=50000,
               n_unq_samps_max=1e5,

               reweight_samples_by_psi=False,
               no_mask_psi=False,
               full_mask_psi=False,

               n_train=5000,
               n_pretrain=0,

               n_lut=0,
               n_hid=32,
               n_layer=1,
               n_hid_phase=-1,
               n_layer_phase=-1,

               output_freq=25,
               save_freq=-1,

               load_hamiltonian=False,
               overwrite_hamiltonian=False,
               presolve_hamiltonian=False,

               pretrained_model_loc=None,
               cont=False,

               n_excitations_max=-1,
               comb_amp_phase = False,
               use_amp_spin_sym=True,
               use_phase_spin_sym=False,
               aggregate_phase=True,

               restrict_H=True,
               reset_opt=False,
               verbose=False,
               seed=-1
               ):
    parser = argparse.ArgumentParser(description='Run experimental script.', allow_abbrev=True)

    parser.add_argument('-m','--molecule', nargs='?', default=molecule,
                        help='The molecule folder')

    parser.add_argument('-hf', '--hamiltonian_fname', nargs='?', default=hamiltonian_fname,
                        help='The qubit hamiltonian pkl file location.')

    parser.add_argument('-o', '--out', nargs='?', default=out,
                        help='The output folder')

    parser.add_argument('-n', '--number', nargs='?', default=number, type=int,
                        help='The number of experimental runs')

    parser.add_argument('-qo', '--qubit_ordering', nargs='?', default=qubit_ordering, type=int,
                        help='Qubit ordering (+/-1)')

    parser.add_argument('-l', '--load', nargs='?', default=pretrained_model_loc,
                        help='The (optional) location of a pre-trained model to load.')

    parser.add_argument('-c', '--cont', default=cont, action='store_true',
                        help='Continue previous training run if possible.')

    parser.add_argument('-r', '--resetOpt', default=reset_opt, action='store_true',
                        help='Reset the parameter optimizer.')

    parser.add_argument('-n_samps', nargs='?', default=n_samps, type=int,
                        help='The (initial) number of samples per batch')

    parser.add_argument('-n_samps_max', nargs='?', default=n_samps_max, type=int,
                        help='The maximum of samples per batch')

    parser.add_argument('-n_unq_samps_max', nargs='?', default=n_unq_samps_max, type=int,
                        help='The maximum number of unique samples per batch')

    parser.add_argument('-n_unq_samps_min', nargs='?', default=n_unq_samps_min, type=int,
                        help='The maximum number of unique samples per batch')

    parser.add_argument('-weight_by_psi', default=reweight_samples_by_psi, action='store_true',
                        help='Reweight samples by |psi|^2 instead of sample count.')

    parser.add_argument('-no_mask_psi', default=no_mask_psi, action='store_true',
                        help='Do not mask the wavefunction to the restricted Hilbert space.')

    parser.add_argument('-full_mask_psi', default=full_mask_psi, action='store_true',
                        help='Mask the wavefunction to the only restricted Hilbert space.')

    parser.add_argument('-lr', nargs='?', default=lr, type=float,
                        help='The learning rate.')

    parser.add_argument('-lr_lut', nargs='?', default=lr_lut, type=float,
                        help='The lut learning rate.')

    parser.add_argument('-n_train', nargs='?', default=n_train, type=int,
                        help='The number of training epochs.')

    parser.add_argument('-n_pretrain', nargs='?', default=n_pretrain, type=int,
                        help='The number of pre-training epochs.')

    parser.add_argument('-n_lut', nargs='?', default=n_lut, type=int,
                        help='The number of luts.')

    parser.add_argument('-n_hid', nargs='?', default=n_hid, type=int,
                        help='The number of hidden units per layer.')

    parser.add_argument('-n_layer', nargs='?', default=n_layer, type=int,
                        help='The number of layers.')

    parser.add_argument('-n_hid_phase', nargs='?', default=n_hid_phase, type=int,
                        help='The number of hidden units per layer for the phase network (-1 --> match amplitude network).')

    parser.add_argument('-n_layer_phase', nargs='?', default=n_layer_phase, type=int,
                        help='The number of layer for the phase network (-1 --> match amplitude network).')

    parser.add_argument('-output_freq', nargs='?', default=output_freq, type=int,
                        help='The logging frequency (in epochs).')

    parser.add_argument('-save_freq', nargs='?', default=save_freq, type=int,
                        help='The saving frequency (in epochs).')

    parser.add_argument('-loadH', default=load_hamiltonian, action='store_true',
                        help='Load the Hamiltonian from file.')

    parser.add_argument('-overwriteH', default=overwrite_hamiltonian, action='store_true',
                        help='Save the Hamiltonian to a file.')

    parser.add_argument('-presolveH', default=presolve_hamiltonian, action='store_true',
                        help='Pre-solve the full Hamiltonian (if not loaded via -loadH).')

    parser.add_argument('-n_excitations_max', nargs='?', default=n_excitations_max, type=int,
                        help='Maximum number of excitations.')

    parser.add_argument('-comb_amp_phase', default=comb_amp_phase, action='store_true',
                        help='Combine amplitude and phase conditional blocks.')

    parser.add_argument('-no_amp_sym', default=not use_amp_spin_sym, action='store_true',
                        help='Neglect amplitude exchange symmetry in the ansatz.')

    parser.add_argument('-phase_sym', default=use_phase_spin_sym, action='store_true',
                        help='Apply phase exchange symmetry in the ansatz.')

    parser.add_argument('-single_phase', default=not aggregate_phase, action='store_true',
                        help='Use only a single phase block.')

    parser.add_argument('-no_restrictedH', default=not restrict_H, action='store_true',
                        help='Do not restrict the ansatz space to only physically viable basis states.')

    parser.add_argument('-v', '--verbose', default=verbose, action='store_true',
                        help='Verbose logging.')

    parser.add_argument('-s','--seed', nargs='?', default=seed, type=int,
                        help='Training seed.')

    return parser

def run_from_parser(parser):
    args = parser.parse_args()

    if args.no_mask_psi and args.full_mask_psi:
        raise Exception("Invalid option combination: at most one of -no_mask_psi and -full_mask_psi can be specified.")

    molecule_fname = args.molecule
    exp_name = args.out
    if exp_name is None:
        exp_name = os.path.split(molecule_fname)[-1]
        exp_name = os.path.join(_EXP_BASE_NAME, exp_name)
        if args.n_samps < 1e3:
            samp_str = f"{int(args.n_samps)}"
        elif args.n_samps < 1e6:
            samp_str = f"{int(args.n_samps / 1e3)}k"
        elif args.n_samps < 1e9:
            samp_str = f"{int(args.n_samps / 1e6)}M"
        else:
            samp_str = f"{int(args.n_samps / 1e9)}B"
        exp_name += f"_{samp_str}_samps"
    if args.no_amp_sym:
        exp_name += "_noAmpSym"
    if args.phase_sym:
        exp_name += "_phaseSym"
    if args.no_restrictedH:
        exp_name += "_no_restrictedH"

    if args.no_mask_psi:
        exp_name += "_no_mask_psi"
        masking = NadeMasking.NONE
    elif args.full_mask_psi:
        exp_name += "_full_mask_psi"
        masking = NadeMasking.FULL
    else:
        masking = NadeMasking.PARTIAL

    n_excitations_max = args.n_excitations_max
    if n_excitations_max < 0:
        n_excitations_max = None
    save_freq = args.save_freq
    if save_freq < 0:
        save_freq = None

    print(f"Running experimental script: {__file__}\nResults will be saved to: {exp_name}/")
    print("\nscript options:")
    for label, val in zip(
            ["molecule_fname", "hamiltonian_fname", "exp_name", "num_experiments", "load (pre_trained model)", "continue_experiment", "qubit_ordering", "lr", "lr_lut",
             "n_samps", "n_samps_max", "n_unq_samps_max", "n_unq_samps_min",
             "weight_by_psi", "no_mask_psi", "full_mask_psi", "n_train", "n_pretrain", "n_lut", "n_hid", "n_layer", "n_hid_phase", "n_layer_phase",
             "output_freq", "save_freq",
             "comb_amp_phase", "no_amp_sym", "phase_sym", "single_phase", "loadH", "overwriteH", "presolveH", "n_excitations_max",
             "no_restrictedH", "reset_optimizer", "verbose", "seed"],
            [molecule_fname, args.hamiltonian_fname, exp_name, args.number, args.load, args.cont, args.qubit_ordering, args.lr, args.lr_lut,
             args.n_samps, int(args.n_samps_max), int(args.n_unq_samps_max), int(args.n_unq_samps_min),
             args.weight_by_psi, args.no_mask_psi, args.full_mask_psi, args.n_train, args.n_pretrain, args.n_lut, args.n_hid, args.n_layer,
             args.n_hid_phase, args.n_layer_phase, args.output_freq, args.save_freq,
             args.comb_amp_phase, args.no_amp_sym, args.phase_sym, args.single_phase, args.loadH, args.overwriteH, args.presolveH, args.n_excitations_max, args.no_restrictedH, args.resetOpt, args.verbose, args.seed]
    ):
        print(f"\t{label} : {val}")
    print("")

    _run(molecule_fname=molecule_fname,
         hamiltonian_fname=args.hamiltonian_fname,
         exp_name=exp_name,
         num_experiments=args.number,
         pretrained_model_loc=args.load,
         continue_experiment=args.cont,
         qubit_ordering=args.qubit_ordering,
         lr=args.lr,
         lr_lut=args.lr_lut,
         n_samps=args.n_samps,
         n_samps_max=args.n_samps_max,
         n_unq_samps_min=args.n_unq_samps_min,
         n_unq_samps_max=args.n_unq_samps_max,
         reweight_samples_by_psi=args.weight_by_psi,
         masking=masking,
         n_train=args.n_train,
         n_pretrain=args.n_pretrain,
         n_lut=args.n_lut,
         n_hid=args.n_hid,
         n_layer=args.n_layer,
         n_hid_phase=args.n_hid_phase,
         n_layer_phase=args.n_layer_phase,
         output_freq=args.output_freq,
         save_freq=save_freq,
         loadH=args.loadH,
         overwrite_pauli_hamiltonian=args.overwriteH,
         presolveH=args.presolveH,
         n_excitations_max=n_excitations_max,
         comb_amp_phase=args.comb_amp_phase,
         use_amp_spin_sym=not args.no_amp_sym,
         use_phase_spin_sym=args.phase_sym,
         aggregate_phase=not args.single_phase,
         use_restrictedH=not args.no_restrictedH,
         reset_optimizer=args.resetOpt,
         verbose=args.verbose,
         seed=args.seed)

    success()

def run(*args, **kwargs):
    run_from_parser(get_parser(*args, **kwargs))

if __name__=="__main__":
    run()

import to Mu
import to Union
import & make new arc(base_windows.py) save in leocloud

export to nasa
