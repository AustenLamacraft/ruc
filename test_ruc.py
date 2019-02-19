import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from ruc import *

class TestRandomCircuitFunctions(unittest.TestCase):

    def testRandomGatesAreUnitary(self):
        q = 2
        depth = 4
        gates = random_gates(q, depth)
        gates_as_matrices = [gate.reshape(2*[q**2]) for gate in gates]
        uudag = [gate @ gate.conj().transpose() for gate in gates_as_matrices]
        [assert_allclose(prod, np.identity(q**2, dtype=complex), atol=1e-14) for prod in uudag]

    def testRandomConservingGatesAreUnitary(self):
        depth = 4
        gates = conserving_gates(depth)
        gates_as_matrices = [gate.reshape(2*[4]) for gate in gates]
        uudag = [gate @ gate.conj().transpose() for gate in gates_as_matrices]
        [assert_allclose(prod, np.identity(4, dtype=complex), atol=1e-14) for prod in uudag]

    def testGatesAreCContiguous(self):
        q = 2
        depth = 4
        gates = random_gates(q, depth)
        for gate in gates:
            assert(gate.flags['C_CONTIGUOUS'] == True)

    def testIdentityGatesRepeatedGivesPureState(self):
        q = 2
        depth = 5
        id_gates = [np.identity(q ** 2).reshape([q, q, q, q]) for _ in range(depth)]
        rho = random_ρ(q, depth - 1)
        for _ in range(2):
            rho = cptp_map(rho, id_gates)

        output_evals = eigh(tensor_to_matrix(rho), eigvals_only=True)
        expected = np.zeros(q**depth)
        expected[-1] = 1.
        assert_almost_equal(expected, output_evals)

    def testCPTPMapTracePreserving(self):
        q = 2
        depth = 4
        input_rho = random_ρ(q, depth - 1)
        input_trace = tensor_trace(input_rho)
        output_rho = cptp_map(input_rho, random_gates(q, depth))
        output_trace = tensor_trace(output_rho)
        assert_almost_equal(input_trace, output_trace)

    def testCPTPMapHermiticityPreserving(self):
        q = 2
        depth = 4
        input_rho = random_ρ(q, depth - 1)
        input_rho = (input_rho + tensor_transpose(input_rho).conj()) / 2
        output_rho = cptp_map(input_rho, random_gates(q, depth))
        assert_almost_equal(output_rho, tensor_transpose(output_rho).conj())

    def testCPTPMapAndApplyGatesConsistent(self):
        q = 2
        depth = 5
        gates = random_gates(q, depth)
        input_state = random_state(q, depth - 1)
        input_rho = pure_ρ(input_state)
        out_1 = cptp_map(input_rho, gates)
        output_state = apply_gates(input_state, gates)

        # Use the q^2 output states to make a density matrix
        out_2 = np.tensordot(np.transpose(output_state, list(range(2, depth + 1)) + [0, 1]), output_state.conj(), axes=2)
        # Now put the axes in the right order
        index_order = np.arange(2 * (depth - 1)).reshape(2, -1).T.flatten()
        out_2 = np.transpose(out_2, index_order)
        assert_almost_equal(out_1, out_2)


    def testApplyGatesPreservesNorm(self):
        q = 2
        depth = 4
        gates = random_gates(q, depth)
        state = random_state(q, depth - 1)
        state = apply_gates(state, gates)
        assert_almost_equal(1., inner_product(state, state))


    def testNextStepGivesProbabilities(self):
        q = 2
        depth = 4
        gates = random_gates(q, depth)
        input_state = random_state(q, depth - 1)
        probs, _ = next_step(input_state, gates)
        assert_almost_equal(1., sum(probs))

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

    def testTraceSquareComputesTraceSquare(self):
        q = 2
        depth = 3
        rho = random_ρ(q, depth)
        purity = trace_square(rho)
        matrix_rho = tensor_to_matrix(rho)
        matrix_purity = np.trace(matrix_rho @ matrix_rho)
        assert_almost_equal(purity, matrix_purity)

    def testRandomRhoIsImpure(self):
        q = 2
        depth = 3

        for _ in range(10):
            rho = random_ρ(q, depth)
            purity = trace_square(rho)
            assert purity < 1.

    def testPureRhoIsPure(self):
        q = 2
        depth = 4
        state = random_state(q, depth)
        rho = pure_ρ(state)
        assert_almost_equal(1., trace_square(rho))


    def testInnerProductsLessThanUnity(self):
        q = 2
        depth = 4
        for _ in range(10):
            state1 = random_state(q, depth)
            state2 = random_state(q, depth)
            assert(np.abs(inner_product(state1, state2)) < 1.)
