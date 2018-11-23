import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from ruc_einsum import random_gates, ruc_channel, tensor_trace, tensor_conj_transp, random_ρ


class TestRandomCircuitFunctions(unittest.TestCase):

    def testRandomGatesAreUnitary(self):
        q = 2
        depth = 3
        gates = random_gates(q, depth)
        gates_as_matrices = [gate.reshape(2*[q**2]) for gate in gates]
        uudag = [gate @ gate.conj().transpose() for gate in gates_as_matrices]
        [assert_allclose(prod, np.identity(q**2, dtype=complex), atol=1e-14) for prod in uudag]

    def testGatesAreCContiguous(self):
        q = 2
        depth = 3
        gates = random_gates(q, depth)
        for gate in gates:
            assert(gate.flags['C_CONTIGUOUS'] == True)

    def testRucChannelTracePreserving(self):
        q = 2
        depth = 3
        input_rho = np.random.rand(*2*depth*[q])
        input_trace = tensor_trace(input_rho)
        output_rho = ruc_channel(input_rho)
        output_trace = tensor_trace(output_rho)
        assert_almost_equal(input_trace, output_trace)

    def testRucChannelHermiticityPreserving(self):
        q = 2
        depth = 3
        input_rho = np.random.rand(*2 * depth * [q])
        input_rho = (input_rho + tensor_conj_transp(input_rho)) / 2
        output_rho = ruc_channel(input_rho)
        assert_almost_equal(output_rho, tensor_conj_transp(output_rho))

    def testRucChannelPositive(self):
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