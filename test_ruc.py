import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from ruc import random_gates, ruc_channel


class TestRandomCircuitFunctions(unittest.TestCase):

    def testRandomGatesAreUnitary(self):
        q = 2
        depth = 3
        gates = random_gates(q, depth)
        gates_as_matrices = [gate.transpose(2,0,1,3).reshape(2*[q**2]) for gate in gates]
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
        random_matrix = np.random.rand(*2*[q**depth])
        random_matrix = (random_matrix + random_matrix.transpose()) / 2
        input_trace = np.trace(random_matrix)
        input_tensor = random_matrix.reshape(2*depth*[q])
        output_tensor = ruc_channel(input_tensor)
        output_trace = np.trace(output_tensor.reshape(2*[q**depth]))
        assert_almost_equal(input_trace, output_trace)

    def testRucChannelHermiticityPreserving(self):
        q = 2
        depth = 3
        random_matrix = np.random.rand(*2 * [q ** depth])
        random_matrix = (random_matrix + random_matrix.transpose()) / 2
        input_tensor = random_matrix.reshape(2 * depth * [q])
        output_tensor = ruc_channel(input_tensor)
        output_matrix = output_tensor.reshape(2 * [q ** depth])
        assert_almost_equal(output_matrix, output_matrix.conj().transpose())

    def testRucChannelPositive(self):
        """
        Use Choi form to test for positivity
        """
        pass

