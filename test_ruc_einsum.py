import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from ruc_einsum import *


class TestRandomCircuitFunctions(unittest.TestCase):

    def testRandomGatesAreUnitary(self):
        q = 2
        depth = 4
        gates = random_gates(q, depth)
        gates_as_matrices = [gate.reshape(2*[q**2]) for gate in gates]
        uudag = [gate @ gate.conj().transpose() for gate in gates_as_matrices]
        [assert_allclose(prod, np.identity(q**2, dtype=complex), atol=1e-14) for prod in uudag]

    def testGatesAreCContiguous(self):
        q = 2
        depth = 4
        gates = random_gates(q, depth)
        for gate in gates:
            assert(gate.flags['C_CONTIGUOUS'] == True)

    def testIdentityGatesRepeatedGivesPureState(self):
        q = 2
        depth = 4
        id_gates = [np.identity(q ** 2).reshape([q, q, q, q]) for _ in range(depth)]
        rho = random_ρ(q, depth)
        for _ in range(2):
            rho = cptp_map(rho, id_gates)

        output_evals = eigh(tensor_to_matrix(rho), eigvals_only=True)
        expected = np.zeros(q**depth)
        expected[-1] = 1.
        assert_almost_equal(expected, output_evals)

    def testCPTPMapTracePreserving(self):
        q = 2
        depth = 4
        input_rho = random_ρ(q, depth)
        input_trace = tensor_trace(input_rho)
        output_rho = cptp_map(input_rho, random_gates(q, depth))
        output_trace = tensor_trace(output_rho)
        assert_almost_equal(input_trace, output_trace)

    def testCPTPMapHermiticityPreserving(self):
        q = 2
        depth = 3
        input_rho = random_ρ(q, depth)
        input_rho = (input_rho + tensor_conj_transp(input_rho)) / 2
        output_rho = cptp_map(input_rho, random_gates(q, depth))
        assert_almost_equal(output_rho, tensor_conj_transp(output_rho))

    def testCPTPMapPositive(self):
        """
        Use Choi form to test for positivity
        """
        pass

    def testRandomRhoUnitTrace(self):
        q = 2
        depth = 3
        rho = random_ρ(q, depth)
        trace = tensor_trace(rho)
        assert_almost_equal(trace, 1.)

    def testApplyGatesPreservesNorm(self):
        q = 2
        depth = 3
        gates = random_gates(q, depth)
        state = random_state(q, depth)
        state = apply_gates(state, gates)
        assert_almost_equal(1., np.einsum("...,...", state, state))
