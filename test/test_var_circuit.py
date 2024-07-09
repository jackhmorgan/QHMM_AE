import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator
from qiskit.circuit.library import QFT
from QHMM_AE import VarCircuit, precision_from_growths, AdderBaseQFT, OneStepGrowths

class expectedVar(QuantumCircuit):
    def _AdderBaseQFT(self,
                      value, 
                      num_val_qubits, 
                      fractional_precision):
        ''' Adds the constant value in Fourier basis.
        '''
        circ_a = QuantumCircuit(num_val_qubits)
        for i in range(num_val_qubits):
            lam = value * np.pi * 2**(fractional_precision - i)
            circ_a.p(lam, i)   
        return circ_a.to_gate(label='Add_value')
    
    def _OneStepGrowths(self,
                       growths, 
                       num_val_qubits,
                       fractional_precision):
        ''' Adds the appropriate growth possibility in each regime.
        '''
        num_state_qubits = int(np.ceil(np.log2(len(growths))))
        circ_g = QuantumCircuit(num_val_qubits+num_state_qubits)
        for ctrl in range(len(growths)):
            if growths[ctrl] != 0:
                add = AdderBaseQFT(growths[ctrl], 
                                num_val_qubits=num_val_qubits,
                                fractional_precision=fractional_precision)
                circ_g.append(add.control(ctrl_state=ctrl, num_ctrl_qubits=num_state_qubits), circ_g.qubits)
        return circ_g.to_gate(label='Add_growth')

    def __init__(self):

        growths = [0, 0.25]
        num_time_steps = 4
        loss = 0.3
        num_val_qubits = 4
        fractional_precision = 2
        num_state_qubits = 1


        iQ = QFT(num_val_qubits, approximation_degree=0, do_swaps=False, inverse=True, insert_barriers=False).to_gate()
        circ = QuantumCircuit(num_val_qubits+(num_state_qubits*num_time_steps))
        circ.h(circ.qubits[(num_state_qubits*num_time_steps):])

        osg = self._OneStepGrowths(growths=growths,
                            num_val_qubits=num_val_qubits,
                            fractional_precision=fractional_precision)
        val_qubits = circ.qubits[-num_val_qubits:]
        for ts in range(num_time_steps):
            start_ci = ts*num_state_qubits
            control_qubits = [start_ci + i for i in range(num_state_qubits)]
            circ.append(osg, control_qubits+val_qubits)

    
        loss_adjustment = -2**(-1-fractional_precision)
        comparator = self._AdderBaseQFT(value = loss_adjustment-loss,
                                    num_val_qubits=num_val_qubits,
                                    fractional_precision=fractional_precision,
                                    )
        circ.append(comparator, val_qubits)

        

        circ.append(iQ, qargs=val_qubits)

        super().__init__(circ.num_qubits)
        self.compose(circ, inplace=True)


class TestVarCircuit(unittest.TestCase):
    
    def test_var_circuit_structure(self):
        num_time_steps = 4
        growths = [0, 0.25]
        loss = 0.3
        num_val_qubits = 4
        insert_barriers = True

        circ = VarCircuit(num_time_steps=num_time_steps,
                          growths=growths,
                          loss=loss,
                          num_val_qubits=num_val_qubits,
                          insert_barriers=insert_barriers)

        self.assertIsInstance(circ, QuantumCircuit)
        num_state_qubits = int(np.ceil(np.log2(len(growths))))
        expected_num_qubits = num_val_qubits + (num_state_qubits * num_time_steps)
        self.assertEqual(circ.num_qubits, expected_num_qubits)

        example = expectedVar()
        self.assertAlmostEqual(Operator(circ), Operator(example))
 

    def test_precision_from_growths(self):
        growths = [0.0, 0.25]
        num_val_qubits = 4
        num_time_steps = 4

        precision = precision_from_growths(growths=growths,
                                           num_val_qubits=num_val_qubits,
                                           num_time_steps=num_time_steps)
        expected_precision = 2
        self.assertEqual(precision, expected_precision)

    def test_adder_base_qft(self):
        value = 0.5
        num_val_qubits = 3
        fractional_precision = 1

        adder = AdderBaseQFT(value=value, 
                             num_val_qubits=num_val_qubits,
                             fractional_precision=fractional_precision)
        self.assertIsInstance(adder, Gate)

    def test_one_step_growths(self):
        growths = [0.0, 0.25]
        num_val_qubits = 3
        fractional_precision = 1

        osg = OneStepGrowths(growths=growths, 
                             num_val_qubits=num_val_qubits,
                             fractional_precision=fractional_precision)
        self.assertIsInstance(osg, Gate)

if __name__ == '__main__':
    unittest.main()
