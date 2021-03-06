import math
import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import eigh

"""
The depth of a circuit is equal to the number of gates. The number of indices in an ancilla state is one less.

Gates are indexed from the top to the bottom of circuit (going __back__ in time).

For consistency with apply_gates we don't use the first gate, and check that the number of gates is one more 
than the number of row indices.

Indexing of a unitary U_{ab,cd} is like this

c   d
|   |
-----
| U |
-----
|   |
a   b

"""


def cptp_map(ρ, gates):
    """
    Quantum channel corresponding to fixed depth circuit defined by gates.

    ρ is a numpy array of shape `[q,q,...q]` (`2*depth` times), where q is the local Hilbert space dimension.

    We assume `depth` row alternating with `depth` column indices: [r0, c0, r1, c1, ...]

    Returns: output density matrix of the same shape
    """

    assert (len(gates) == len(ρ.shape) // 2 + 1), "Number of gates must be one more than the number of row indices."

    # Remove the first gate
    gates = gates[1:]
    # Trace out the first index
    ρ = np.trace(ρ, axis1=0, axis2=1)

    # We are going to denote 'a,b,...' for 'in' indices, `x,y,.... for 'out'
    # Capitals for contractions between unitaries
    # After contraction we move indices to the end using ellipsis notation
    # After going through all the gates the indices are back in their starting position.

    ρ = np.einsum('AaxC,BbyC,ab...->AB...xy', gates[0], gates[0].conj(), ρ,
                  optimize=['einsum_path', (0, 1), (0, 1)])

    for gate in gates[1:-1]:
        ρ = np.einsum('AaxC,BbyD,CDab...->AB...xy', gate, gate.conj(), ρ,
                      optimize=['einsum_path', (0, 2), (0, 1)])

    # Note that at this step we set the initial state to be Neel-like in order that
    # conserving circuits still do something interesting
    ρ = np.einsum('xC,yD,CD...->...xy', gates[-1][0, 1], gates[-1][0, 1].conj(), ρ,
                  optimize=['einsum_path', (0, 2), (0, 1)])

    return ρ


def apply_gates(state, gates):
    """
    Apply unitary gates to ancilla states.
    Returns state of ancilla AND two physical sites (as the first two indices)
    """

    assert (len(gates) == len(state.shape) + 1), "Number of gates must be one more than the number of state indices."

    state = np.einsum('Aast,a...->A...st', gates[0], state)

    for gate in gates[1:-1]:
        state = np.einsum('AaxB,Ba...->A...x', gate, state)

    state = np.einsum('xB,B...->...x', gates[-1][0, 0], state)

    return state


def next_step(state, gates):
    """
    Return the probabilities of each (s1, s2) pair of physical indices, along with the ancilla states that go with each.
    """

    state = apply_gates(state, gates)
    q = state.shape[0]
    probs = []
    states = []
    for phys_index in np.ndindex(q, q):
        ancilla_state = state[phys_index]
        prob = square_norm(ancilla_state)
        ancilla_state = ancilla_state / np.sqrt(prob)
        states.append(ancilla_state)
        probs.append(prob)

    return probs, states

def tensor_trace(tensor):
    """
    Calculate the matrix trace of a tensor with rows and column multi-indices as [r0, c0, r1, c1, ...]
    """
    while len(tensor.shape) != 0:
        tensor = np.einsum("aa...->...", tensor)

    return tensor


def tensor_transpose(tensor):
    """
    Calculate the transpose of a tensor with rows and column multi-indices as [r0, c0, r1, c1, ...]
    """
    num_indices = len(tensor.shape)
    trans_perm = np.arange(num_indices).reshape([-1,2])[:,::-1].flatten()
    tensor = tensor.transpose(trans_perm)

    return tensor

def trace_square(tensor):
    """
    Calculate the trace of the square of a tensor with rows and column multi-indices as [r0, c0, r1, c1, ...]
    """
    num_indices = len(tensor.shape)
    return np.tensordot(tensor, tensor_transpose(tensor), axes=num_indices)

def inner_product(state1, state2):
    """
    Calculate the inner product of two states.
    """
    assert(state1.shape == state2.shape)
    num_indices = len(state1.shape)
    return np.tensordot(state1.conj(), state2, axes=num_indices)

def square_norm(state):
    return inner_product(state, state).real

def pure_ρ(state):
    """
    Return a pure density matrix corresponding to the input state in format [r0, c0, r1, c1, ...]
    """
    num_indices= len(state.shape)
    ρ =  np.tensordot(state, state.conj(), axes=0)
    index_order =  np.arange(2 * num_indices).reshape(2, -1).T.flatten()
    return np.transpose(ρ, axes=index_order)

def tensor_to_matrix(tensor):
    num_row_indices = len(tensor.shape) // 2
    q = tensor.shape[0]
    tensor = tensor.transpose(list(range(0, 2 * num_row_indices, 2)) + list(range(1, 2 * num_row_indices, 2)))
    return tensor.reshape(2 * [q ** num_row_indices])


def matrix_to_tensor(matrix, q):
    num_row_indices = int(math.log(matrix.shape[0], q))
    tensor = matrix.reshape(2 * num_row_indices * [q])
    axis_order = np.arange(2 * num_row_indices).reshape([2, -1]).T.reshape([-1])
    return tensor.transpose(axis_order)


def random_gates(q, depth):
    return unitary_group.rvs(q ** 2, size=depth).reshape([depth, q, q, q, q])

def conserving_gates(depth):
    """
    Random conserving gates for spin-1/2
    """
    gates = np.zeros([depth, 2, 2, 2 ,2], dtype=np.complex)
    phases = np.exp(2j * np.pi * np.random.rand(depth, 2))
    gates[:, 0, 0, 0, 0] =  phases[:, 0]
    gates[:, 1, 1, 1, 1] = phases[:, 1]
    u2_gates = unitary_group.rvs(2, size=depth)
    gates[:, 0, 1, 0, 1] = u2_gates[:, 0, 0]
    gates[:, 1, 0, 1, 0] = u2_gates[:, 1, 1]
    gates[:, 0, 1, 1, 0] = u2_gates[:, 0, 1]
    gates[:, 1, 0, 0, 1] = u2_gates[:, 1, 0]

    return gates



def random_ρ(q, depth):
    """
    Generate a random density matrix with row and column multi-indices as [r0, c0, r1, c1, ...]
    """
    random_matrix = np.random.rand(*2*[q**depth])
    random_matrix = random_matrix @ random_matrix.transpose()
    random_tensor = matrix_to_tensor(random_matrix, q)
    ρ = random_tensor / tensor_trace(random_tensor)
    return ρ


def random_state(q, depth):
    """
    Generate a random normalized state
    """
    random_state = np.random.rand(* depth * [q])
    return random_state / np.sqrt(square_norm(random_state))



