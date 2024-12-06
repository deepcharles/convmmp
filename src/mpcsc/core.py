import numpy as np
from joblib import Parallel, delayed
from numba import njit
from scipy.signal import find_peaks
from scipy.sparse.linalg import LinearOperator, lsqr


@njit
def compute_path(k_path, width, depth):
    # Table III from https://arxiv.org/pdf/1308.4791
    temp = k_path
    out = np.empty(depth, dtype=np.int32)
    for k in range(depth):
        out[k] = temp % width
        temp = temp // width
    return out


def convolve_with_dict(dictionary, activations):
    # reconstruct from activations and dictionary
    # activations, shape (n_samples_valid, n_atoms)
    # dictionary, shape (n_atoms, n_samples_atom, n_dims)
    # out, shape (n_samples, n_dims)
    n_atoms, n_samples_atom, n_dims = dictionary.shape
    n_samples_valid, n_atoms = activations.shape
    n_samples = n_samples_valid + n_samples_atom - 1
    out = np.zeros((n_samples, n_dims), dtype=np.float64)
    for k_dim in range(n_dims):
        for k_atom in range(n_atoms):
            out[:, k_dim] += np.convolve(
                activations[:, k_atom], dictionary[k_atom, :, k_dim], mode="full"
            )
    return out


def correlate_with_dict(dictionary, signal):
    # reconstruct from activations and dictionary
    # signal, shape (n_samples, n_atoms)
    # dictionary, shape (n_atoms, n_samples_atom, n_dims)
    # out, shape (n_samples_valid, n_atoms)
    n_atoms, n_samples_atom, n_dims = dictionary.shape
    n_samples, n_dims = signal.shape
    n_samples_valid = n_samples - n_samples_atom + 1
    out = np.zeros((n_samples_valid, n_atoms), dtype=np.float64)
    for k_dim in range(n_dims):
        for k_atom in range(n_atoms):
            out[:, k_atom] += np.correlate(
                signal[:, k_dim], dictionary[k_atom, :, k_dim], mode="valid"
            )
    return out


@njit
def convolve_with_dict_sp(dictionary, time_idxs, atom_idxs, active_vals, n_samples):
    n_atoms, n_samples_atom, n_dims = dictionary.shape
    out = np.zeros((n_samples, n_dims))
    n_nnz = time_idxs.shape[0]
    for k_nnz in range(n_nnz):
        start = time_idxs[k_nnz]
        end = start + n_samples_atom
        val = active_vals[k_nnz]
        k_atom = atom_idxs[k_nnz]
        out[start:end] += val * dictionary[k_atom]
    return out


@njit
def correlate_with_dict_sp(dictionary, time_idxs, atom_idxs, signal):
    n_atoms, n_samples_atom, n_dims = dictionary.shape
    n_nnz = time_idxs.shape[0]
    out = np.empty(n_nnz)
    for k_nnz in range(n_nnz):
        start = time_idxs[k_nnz]
        end = start + n_samples_atom
        k_atom = atom_idxs[k_nnz]
        atom = dictionary[k_atom]
        out[k_nnz] = np.sum(signal[start:end] * atom)
    return out


@njit
def approx_signal(n_samples, dictionary, time_idxs, atom_idxs):
    # place atoms with amplitude 1 at the correct locations
    n_atoms, n_samples_atom, n_dims = dictionary.shape
    approx = np.zeros((n_samples, n_dims), dtype=np.float64)
    for k_atom in range(time_idxs.size):
        start = time_idxs[k_atom]
        end = start + n_samples_atom
        atom = dictionary[atom_idxs[k_atom]]
        approx[start:end] += atom
    return approx


def csc_single_path(signal, dictionary, path, distance=20, debug=False):
    # signal, shape (n_samples,)
    # dictionary, shape (n_samples_atom, n_atoms)

    n_atoms, n_samples_atom, n_dims = dictionary.shape
    (n_samples, n_dims) = signal.shape
    n_times_valid = n_samples - n_samples_atom + 1

    k_iter = 0
    residual = signal
    mse = np.square(residual).mean()
    selected_time_idxs = list()
    selected_atom_idxs = list()

    if debug:
        residual_list = [mse]

    for path_element in path:
        # compute all correlations, shape (n_samples_valid, n_atoms)
        all_correlations = correlate_with_dict(signal=residual, dictionary=dictionary)
        # the ranking of each atom at each time stamp, shape (n_samples_valid, n_atoms)
        argmax_inds = all_correlations.argmax(axis=1, keepdims=True)
        # maximum correlation for each time stamp, shape (n_samples_valid,)
        all_correlations_maxpooled = all_correlations.max(axis=1)
        # time indexes of the largest correlations, in descending order
        time_idxs, _ = find_peaks(all_correlations_maxpooled, distance=distance)
        order_by_correlation = np.argsort(all_correlations_maxpooled[time_idxs])
        time_idxs = time_idxs[order_by_correlation[::-1]]
        # atom indexes of the largest correlations, in descending order
        atom_idxs = argmax_inds[time_idxs].squeeze()

        # keep time and atom index that correspond to the path
        selected_time_idxs.append(time_idxs[path_element])
        selected_atom_idxs.append(atom_idxs[path_element])

        # update the residual
        approx = approx_signal(
            n_samples,
            dictionary,
            atom_idxs=np.array(selected_atom_idxs),
            time_idxs=np.array(selected_time_idxs),
        )
        residual = signal - approx
        mse = np.square(residual).mean()
        k_iter += 1

        if debug:
            print(f"{ k_iter = } ({path_element}), { mse = :.2f}")
            print(f"selected time {time_idxs[path_element]}")
            print(f"selected atom {atom_idxs[path_element]}")
            residual_list.append(mse)

    if debug:
        return approx, selected_time_idxs, selected_atom_idxs, residual_list

    return approx, selected_time_idxs, selected_atom_idxs


csc_single_path_delayed = delayed(csc_single_path)


def multipathcsc(
    signal,
    dictionary,
    n_atoms_to_find=3,
    distance=20,
    width=3,
    n_paths=5,
    debug=False,
    n_jobs=-2,
):
    # signal, shape (n_samples, n_dims)
    # dictionary, shape (n_atoms, n_samples_atom, n_dims)
    # output: (approx, activations, path)
    # approx, shape (n_samples, n_dims,)
    # activations, shape (n_samples_valid, n_atoms)
    # path, tuple, optimal path
    all_paths = [
        compute_path(k_path=k_path, width=width, depth=n_atoms_to_find)
        for k_path in range(n_paths)
    ]
    all_results = Parallel(n_jobs=n_jobs)(
        csc_single_path_delayed(
            signal=signal,
            dictionary=dictionary,
            distance=distance,
            path=path,
            debug=False,
        )
        for path in all_paths
    )

    all_squared_errors = list()
    for k_path in range(n_paths):
        approx, *_ = all_results[k_path]
        se = np.sum(np.square(signal - approx))
        all_squared_errors.append(se)

    k_best = np.argmin(all_squared_errors)
    approx, selected_time_idxs, selected_atom_idxs = all_results[k_best]
    path = all_paths[k_best]
    if debug:
        return (
            approx,
            np.array(selected_time_idxs),
            np.array(selected_atom_idxs),
            path,
        ), dict(squared_error=all_squared_errors[k_best], k_path=k_best)
    return (
        approx,
        np.array(selected_time_idxs),
        np.array(selected_atom_idxs),
        path,
    )
