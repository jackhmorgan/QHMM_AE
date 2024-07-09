'''
Copyright 2024 Jack Morgan

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import unittest
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from QHMM_AE import TrainableHQMM
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

class expectedTrainableCircuit(QuantumCircuit):
    def __init__(self):
        super().__init__(2,2)
        ansatz = EfficientSU2(num_qubits=self.num_qubits, su2_gates=['ry'], entanglement='linear')
        self.compose(RealAmplitudes(num_qubits=1, parameter_prefix='phi'), inplace=True)
        self.compose(ansatz, inplace=True)
        self.measure(1,0)
        self.reset(1)
        self.compose(ansatz, inplace=True)
        self.measure(1,1)
        self.reset(1)
    
class expectedStatePrepCircuit(QuantumCircuit):
    def __init__(self):
        super().__init__(4,1)
        initial_state = RealAmplitudes(num_qubits=1, parameter_prefix='phi').to_gate()
        ansatz = EfficientSU2(num_qubits=2, su2_gates=['ry'], entanglement='linear').to_gate()
        processing_circuit = QuantumCircuit(3)
        processing_circuit.x([-1])
        processing_circuit = processing_circuit.to_gate()
        self.append(initial_state,[0])
        self.append(ansatz, [0,1])
        self.append(ansatz, [0,2])
        self.append(processing_circuit, [1,2,3])

class TestTrainableHQMM(unittest.TestCase):

    def setUp(self):
        # Example parameters for testing
        self.num_time_steps = 2
        self.num_qubits = 2
        self.measurement_qubits = [1]
        self.initial_state = RealAmplitudes(num_qubits=1, parameter_prefix='phi')
        self.ansatz = EfficientSU2(num_qubits=self.num_qubits, su2_gates=['ry'], entanglement='linear')

        self.hqmm = TrainableHQMM(
            num_qubits=self.num_qubits,
            initial_state=self.initial_state,
            ansatz=self.ansatz,
            num_time_steps=self.num_time_steps,
            measurement_qubits=self.measurement_qubits,
            su2_gates=['ry'],
            entanglement='linear'
        )

    def test_initialization(self):
        # Check if the model parameters are set correctly
        self.assertEqual(self.hqmm.num_time_steps, self.num_time_steps)
        self.assertEqual(self.hqmm.num_qubits, self.num_qubits)
        self.assertEqual(self.hqmm.initial_state, self.initial_state)
        self.assertEqual(self.hqmm.ansatz, self.ansatz)
        self.assertEqual(len(self.hqmm.measurement_qubits), len(self.measurement_qubits))
        expected = expectedTrainableCircuit()

        params = np.random.uniform(0, 8*np.pi, expected.num_parameters)
        expected.assign_parameters(params, inplace=True)
        circ = self.hqmm.assign_parameters(params)
        simulator = AerSimulator()
        expected_dist = simulator.run(transpile(expected, simulator)).result().results[0].data.counts
        circ_dist = simulator.run(transpile(circ, simulator)).result().results[0].data.counts

        for key, value in expected_dist.items():
            np.testing.assert_allclose(
                value, circ_dist.get(key, 0), atol=25
            )

    def test_to_state_prep(self):
        expected_sp = expectedStatePrepCircuit()
        params = np.random.uniform(0, 8*np.pi, expected_sp.num_parameters)
        expected_sp.assign_parameters(params, inplace=True)
        # Create a processing circuit and check the to_state_prep method
        processing_circuit = QuantumCircuit(self.num_time_steps * len(self.measurement_qubits) + 1)
        processing_circuit.x(-1)

        circ, objective = self.hqmm.to_state_prep(processing_circuit=processing_circuit, objective=0)
        circ.assign_parameters(params, inplace=True)
        self.assertIsInstance(circ, QuantumCircuit)
        self.assertAlmostEqual(Operator(circ), Operator(expected_sp))

if __name__ == '__main__':
    unittest.main()
