import numpy as np
from numba import jit, complex128
from scipy.stats import unitary_group

def ruc_channel(ρ):
    """
    Quantum channel corresponding to fixed depth random
    unitary circuit.

    ρ is a numpy array of shape `[q,q,...q]` (`2*depth` times),
    where q is the local Hilbert space dimension.

    We assume `depth` row indices followed by `depth` column indices.

    Returns: output density matrix of the same shape
    """

    shape = ρ.shape
    depth = len(shape) // 2
    q = shape[0]
    gates = random_gates(q, depth)
    conj_gates = [contig_conj_transpose(gate) for gate in gates]
    conj_gates.reverse()

    # First trace out the final index
    ρ = np.trace(ρ, axis1=depth-1, axis2=2*depth-1)

    # Add two new indices of size 1
    ρ = np.expand_dims(ρ, axis=depth-1)
    ρ = np.expand_dims(ρ, axis=0)

    # We have to pass the shape tuple and gates into the decorated functions
    ρ_out = mat_mul(ρ, shape, gates, conj_gates)

    return ρ_out

# @jit(nopython=True)
def mat_mul(ρ, out_shape, gates, conj_gates):

    ρ_out = np.zeros(out_shape, dtype=np.complex128)

    for out_idx in np.ndindex(*out_shape):
        ρ_out_elem = 0
        for in_idx, ρ_elem in np.ndenumerate(ρ):
            res = matrix_product(out_idx, in_idx, gates, conj_gates)
            res *= ρ_elem
            ρ_out_elem += res

        ρ_out[out_idx] = ρ_out_elem

    return ρ_out


# @jit(nopython=True)#, locals={'gate_matrix': complex128[:,::1]})
def matrix_product(out_idx, in_idx, gates, conj_gates):

    q = gates[0].shape[0]
    depth = len(gates)

    left_vec = np.zeros(q, dtype=np.complex128)
    left_vec[0] = 1

    for idx, gate in enumerate(gates):
        # Doing this without slices removed a Numba performance warning
        gate_matrix = gate[in_idx[idx], out_idx[idx]]
        left_vec = np.dot(left_vec, gate_matrix)

    for idx, gate in enumerate(conj_gates, 1):
        # Annoyingly, Numba doesn't seem to like negative indices, so we do this.
        gate_matrix = gate[in_idx[2 * depth - idx], out_idx[2 * depth - idx]]
        left_vec = np.dot(left_vec, gate_matrix)

    right_vec = np.zeros(q, dtype=np.complex128)
    right_vec[0] = 1

    res = np.dot(left_vec, right_vec)
    return res

def random_gates(q, depth):
    return [np.ascontiguousarray(unitary_group.rvs(q ** 2).reshape(4 * [q]).transpose(1, 2, 0, 3)) for _ in range(depth)]

def contig_conj_transpose(array):
    return np.ascontiguousarray(array.conj().transpose(0, 1, 3, 2))
