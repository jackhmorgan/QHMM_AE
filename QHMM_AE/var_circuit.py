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
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library.generalized_gates import UCRYGate
from qiskit.circuit.library import QFT
from typing import List

def VarCircuit(num_time_steps: int,
           growths: List[int],
           loss: float,
           num_val_qubits: int,
           insert_barriers: bool = False,
           ):
    """
    The function `VarCircuit` creates a circuit that is useful for calculating the Value at Risk (VaR)
    of a portfolio of loan based on changes in its value over time.
    
    :param num_time_steps: The `num_time_steps` parameter specifies the number of time steps in the
    VarCircuit function.
    :type num_time_steps: int
    :param growths: The `growths` parameter represents a list of possible log returns
    of the asset price for each potential state at each time step.
    :type growths: List[int]
    :param loss: The `loss` parameter represents log return we want to compare our asset 
    performance to
    :type loss: float
    :param num_val_qubits: The `num_val_qubits` parameter represents the
    number of qubits used to store the value of the asset in the circuit.
    :type num_val_qubits: int
    :param insert_barriers: The `insert_barriers` parameter in the `VarCircuit` function is a boolean
    flag that determines whether barriers should be inserted between each step in the quantum circuit,
    defaults to False.
    :type insert_barriers: bool (optional)
    :return: The function `VarCircuit` creates a quantum circuit that inputs a series of qubits representing the
    change in the value of an underlying asset, and outputs an amplitude on the final qubit that corresponds 
    to the probability the underlying asset is worth less than the loss parameter at the end of the time period.
    """

    # number of qubits holding the emission index for each time step
    num_state_qubits = int(np.ceil(np.log2(len(growths))))

    # number of bits after the decimal used to store the value of the portfolio
    fractional_precision = precision_from_growths(growths=growths,
                                                  num_val_qubits=num_val_qubits, 
                                                  num_time_steps=num_time_steps,
                                                  )

    
    # create circuit in Fourier Basis
    circ = QuantumCircuit(num_val_qubits+(num_state_qubits*num_time_steps))
    circ.h(circ.qubits[(num_state_qubits*num_time_steps):])

    # define the gate to add the log returns at each time step
    osg = OneStepGrowths(growths=growths,
                         num_val_qubits=num_val_qubits,
                         fractional_precision=fractional_precision)
    val_qubits = circ.qubits[-num_val_qubits:]

    # loop over time steps
    for ts in range(num_time_steps):
        if insert_barriers:
            circ.barrier()
        start_ci = ts*num_state_qubits
        control_qubits = [start_ci + i for i in range(num_state_qubits)]
        circ.append(osg, control_qubits+val_qubits)

    if insert_barriers:
        circ.barrier()

    # compare to loss
    if loss != 0:
        loss_adjustment = -2**(-1-fractional_precision)
        comparator = AdderBaseQFT(value = loss_adjustment-loss,
                                  num_val_qubits=num_val_qubits,
                                  fractional_precision=fractional_precision,
                                  )
        circ.append(comparator, val_qubits)

        if insert_barriers:
            circ.barrier()

    # return to computational basis
    iQ = QFT(num_val_qubits, approximation_degree=0, do_swaps=False, inverse=True, insert_barriers=False).to_gate() 
    circ.append(iQ, qargs=val_qubits)

    return circ

def AdderBaseQFT(value: float,
                num_val_qubits: int, 
                fractional_precision: int):
    """
    The function `AdderBaseQFT` adds a constant value in Fourier basis using quantum circuits.
    
    :param value: The `value` parameter represents the constant value that you want to add in the
    Fourier basis
    :param num_val_qubits: The `num_val_qubits` parameter represents the number of qubits used to encode
    the value in the quantum circuit. It determines the precision and range of values that can be
    represented in the quantum computation
    :param fractional_precision: The `fractional_precision` parameter represents the number of binary bits
    after the decimal used to represent the constant value in the Fourier basis.
    :return: The function `AdderBaseQFT` returns a Gate implementing the addition of a constant value in
    the Fourier basis.
    """
    circ_a = QuantumCircuit(num_val_qubits)
    for i in range(num_val_qubits):
        lam = value * np.pi * 2**(fractional_precision - i)
        circ_a.p(lam, i)   
    return circ_a.to_gate(label='Add_value')

def OneStepGrowths(growths: list[float], 
                   num_val_qubits: int,
                   fractional_precision: int):
    """
    This function adds the appropriate growth possibility in each regime by creating a quantum circuit
    gate.
    
    :param growths: The `growths` parameter is a list that contains growth possibilities for different
    regimes. Each element in the list represents a growth possibility for a specific regime.
    :param num_val_qubits: The `num_val_qubits` parameter represents the number of qubits used to encode
    the values in the quantum circuit.
    :param fractional_precision: The `fractional_precision` parameter in the `OneStepGrowths` function
    represents the number of fractional bits used in the representation of growth values. It is used in
    the `AdderBaseQFT` function to determine the precision of the addition operation for each growth
    possibility.
    :return: The function `OneStepGrowths` returns a Gate that represents the
    appropriate growth possibilities in each regime based on the input parameters `growths`,
    `num_val_qubits`, and `fractional_precision`. The gate is labeled as 'Add_growth' and is constructed
    by adding growth possibilities using the `AdderBaseQFT` function for each non-zero growth value in
    the
    """
    num_state_qubits = int(np.ceil(np.log2(len(growths))))
    circ_g = QuantumCircuit(num_val_qubits+num_state_qubits)
    for ctrl in range(len(growths)):
        if growths[ctrl] != 0:
            add = AdderBaseQFT(growths[ctrl], 
                               num_val_qubits=num_val_qubits,
                               fractional_precision=fractional_precision)
            circ_g.append(add.control(ctrl_state=ctrl, num_ctrl_qubits=num_state_qubits), circ_g.qubits)
    return circ_g.to_gate(label='Add_growth')

def precision_from_growths(growths,
                            num_val_qubits,
                            num_time_steps):
    """
    The function calculates the fractional precision required based on growth values, number of value
    qubits, and number of time steps.
    
    :param growths: The `growths` parameter is a list of potential growths in one period of time
    :param num_val_qubits: The `num_val_qubits` parameter represents the total number of qubits used for
    representing the total value in the circuit.
    :param num_time_steps: The `num_time_steps` parameter represents the number of time steps in a
    growth simulation
    :return: The function `precision_from_growths` returns the ideal fractional precision calculated based on
    the input parameters `growths`, `num_val_qubits`, and `num_time_steps`.
    """
    max_growth = num_time_steps*(max(growths)-min(growths))
    integer_precision = num_val_qubits

    while max_growth < 2**(integer_precision-2):
        integer_precision-= 1

    while max_growth >= 2**(integer_precision):
        integer_precision+= 1
    
    fractional_precision = num_val_qubits - integer_precision
    return fractional_precision
