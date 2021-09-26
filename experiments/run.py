from experiments._base import run

run(molecule=None,
    out=None,
    number=1,
    lr=-1,

    n_samps=1e7,
    n_samps_max=1e12,
    n_unq_samps_min=1e4,
    n_unq_samps_max=1e5,

    n_hid=128,
    n_layer=1,

    reweight_samples_by_psi=False,
    n_train=10000,
    n_pretrain=0,
    output_freq=25,
    save_freq=-1,
    load_hamiltonian=False,
    overwrite_hamiltonian=False,
    presolve_hamiltonian=False,
    cont=False,
    n_excitations_max=-1,

    use_amp_spin_sym=True,
    use_phase_spin_sym=False,
    comb_amp_phase=False,

    aggregate_phase=True,
    restrict_H=True,
    reset_opt=False)