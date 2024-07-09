from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import BlueprintCircuit, EfficientSU2
from qiskit.circuit import ParameterVector, Gate
from qiskit.circuit.quantumregister import Qubit
from typing import List, Tuple

# This class `TrainableHQMM` is a blueprint circuit with trainable parameters, allowing for
# customization of the ansatz and initial state.
class TrainableHQMM(BlueprintCircuit):
    def __init__(self,
                 num_qubits = None,
                 ansatz = None,
                 initial_state = None,
                 num_time_steps = 1,
                 measurement_qubits = [-1],
                 **kwargs,
                 ):
        """
        This class is a template for a ansatz based Quantum Hidden Markov model circuit.
        
        :param num_qubits: The `num_qubits` parameter specifies the number of qubits in the quantum
        circuit
        :param ansatz: The `ansatz` parameter is the parameterized QuantumCircuit or gate that learns the
        emission and transition behavior of the system for one time step.
        :param initial_state: The `initial_state` parameter is the QuantumCircuit or Gate that that prepares or learns 
        how to prepare the steady state of the system.
        :param num_time_steps: The `num_time_steps` parameter specifies the number of time steps, which determines the number of measurements
        :param measurement_qubits: The `measurement_qubits` parameter is a list that specifies the
        qubits on which measurements will be performed. Defaults to only the final qubit in the ansatz
        """
        super().__init__()
        self._num_time_steps = num_time_steps
        self._sorted_paramters = ParameterVector('x')
        self._initial_state=initial_state

        # set default ansatz
        if ansatz == None:
            if 'su2_gates' in kwargs.keys() and 'entanglement' in kwargs.keys():
                ansatz = EfficientSU2(num_qubits=num_qubits, 
                                                su2_gates=kwargs['su2_gates'],
                                                entanglement=kwargs['entanglement'])
        self._ansatz = ansatz
            
        # check that num_qubits, initial_state, and ansatz agree
        self.num_qubits, self._initial_state, self._ansatz = derive_num_qubits_initial_state_ansatz(
            num_qubits, initial_state, ansatz
        )

        # set measurement qubits and classical bits
        self.measurement_qubits = measurement_qubits
        num_clbits = len(self.measurement_qubits)*self.num_time_steps
        self.add_register(ClassicalRegister(num_clbits, name="c"))

    def _build(self):
        '''
        Constructs Quantum Hidden Markov Model circuit.
        '''
        super()._build()

        # add initial state qubits
        self.initial_state_qubits = self.qubits[:self.initial_state.num_qubits]
        self.compose(self.initial_state, qubits = self.initial_state_qubits, inplace=True)

        # loop ansatz and measurement over the number of time steps
        clbit = 0
        for _ in range(self.num_time_steps):
            self.compose(self.ansatz, inplace=True)
            self.measure(self.measurement_qubits, range(clbit, clbit + len(self.measurement_qubits)))
            self.reset(self.measurement_qubits)
            clbit += len(self.measurement_qubits)
    
    def _check_configuration(self, raise_on_failure=True):
        '''This method is required for a BlueprintCircuit parent class. Not yet implementing any checks.
        '''
        return True
    
    def to_state_prep(self,
                      processing_circuit: QuantumCircuit,
                      objective: int) -> QuantumCircuit:
        '''Function to retun a MC state prep circuit with the trained ansatz.

        Args:
            processing_circuit: a circuit that uses the output distribution of the Markov Chain
                to prepare an amplitude of interest on one objective qubit.
            objective: the index of the objective qubit of the processing circuit
        
        Returns:
            A Tuple with the Unitary Quantum Circuit that combines the Markov chain circuit and the processing circuit, 
                and the index of the objective qubit in the new circuit.
        
        '''
        
        self._build()

        # convert circuits to Gates. Will make the result cooperate with the qiskit_alorithms Amplitude Estimator.
        subcircuits = [self.initial_state, self.ansatz, processing_circuit]
        for i, subcircuit in enumerate(subcircuits):
            if not isinstance(subcircuit, Gate):
                subcircuits[i] = subcircuit.to_gate()

        # number of emission qubits per time step in new circuit
        num_mq = len(self.measurement_qubits)
        circ = QuantumCircuit(self.num_qubits-num_mq+processing_circuit.num_qubits, processing_circuit.num_clbits)

        # Markov chain qubits
        mc_qubits = [qubit._index for qubit in self.qubits if qubit not in self.measurement_qubits]

        # initial state qubits
        is_qubits = [qubit._index for qubit in self.initial_state_qubits]

        # append initial state circuit
        circ.append(subcircuits[0], is_qubits)

        # append the ansatz for each time step
        for layer in range(self.num_time_steps):
            mq = [len(mc_qubits)+layer*num_mq + i for i in range(num_mq)]
            circ.append(subcircuits[1], mc_qubits+mq)

        # append the processing circuit
        circ.append(subcircuits[2], range(len(mc_qubits), circ.num_qubits), range(circ.num_clbits))

        # define the objective qubit of the new circuit
        objective = len(mc_qubits)+objective
        return circ, objective

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in this circuit.

        Returns:
            The number of qubits.
        """
        return super().num_qubits
    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits. If num_qubits is set
        the feature map and ansatz are adjusted to circuits with num_qubits qubits.

        Args:
            num_qubits:  The number of qubits, a positive integer.
        """
        if self.num_qubits != num_qubits:
            # invalidate the circuit
            self._invalidate()
            self.qregs: List[QuantumRegister] = []
            if num_qubits is not None and num_qubits > 0:
                self.qregs = [QuantumRegister(num_qubits, name="q")]
                (self.num_qubits, self._feature_map, self._ansatz) = derive_num_qubits_initial_state_ansatz(num_qubits, self._initial_state, self._ansatz)

    @property
    def initial_state(self) -> QuantumCircuit:
        """Returns initial_state.

        Returns:
            The initial_state.
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: QuantumCircuit) -> None:
        """Set the initial_state circuit.

        Args:
            initial_state: The initial_state circuit.
        """
        if type(self._initial_state) != QuantumCircuit:
            # invalidate the circuit
            self._invalidate()

    @property
    def ansatz(self) -> QuantumCircuit:
        """Returns ansatz.

        Returns:
            The ansatz.
        """
        return self._ansatz

    @ansatz.setter
    def ansatz(self, ansatz: QuantumCircuit) -> None:
        """Set the ansatz.

        Args:
            ansatz: The ansatz.
        """
        if self.ansatz != ansatz:
            # invalidate the circuit
            self._invalidate()

    @property
    def measurement_qubits(self) -> List[Qubit]:
        '''Returns measurement qubits.
        
        Returns:
            measurement qubits.'''
        return self._measurement_qubits
    
    @measurement_qubits.setter
    def measurement_qubits(self, measurement_qubits: List[Qubit]|QuantumRegister|List[int]) -> None:
        '''Sets the measurement qubits. Will change the format to a list of qubits'''
        if isinstance(measurement_qubits,List):
            if all(isinstance(item, Qubit) for item in measurement_qubits):
                self._measurement_qubits = measurement_qubits
            if all(isinstance(item, int) for item in measurement_qubits):
                self._measurement_qubits = [self.qubits[i] for i in measurement_qubits]
        elif isinstance(measurement_qubits, QuantumRegister):
            self._measurement_qubits = [qubit for qubit in measurement_qubits]
        else:
            raise ValueError('measurement_qubits must be a list of integers, list of qubits, or QuantumRegister')
    
    @property
    def num_time_steps(self)->int:
        '''Returns the number of time steps.
        
        Returns:
            The number of time steps of the MC circuit.
        '''
        return self._num_time_steps
    @num_time_steps.setter
    def num_time_steps(self, num_time_steps)->None:
        '''Set the number of time steps.
        
        Args:
            num_time_steps: The number of time steps'''
        self._num_time_steps = int(num_time_steps)
        
def derive_num_qubits_initial_state_ansatz(
    num_qubits: int | None = None,
    initial_state: QuantumCircuit | None = None,
    ansatz: QuantumCircuit | None = None,
) -> Tuple[int, QuantumCircuit, QuantumCircuit]:
    """
    Derives a correct number of qubits, initial state, and ansatz from the parameters.

    If the number of qubits is not ``None``, then the initial state and ansatz are adjusted to this
    number of qubits if required. If such an adjustment fails, an error is raised. Also, if the
    feature map or ansatz or both are ``None``, then :class:`~qiskit.circuit.library.ZZFeatureMap`
    and :class:`~qiskit.circuit.library.RealAmplitudes` are created respectively. If there's just
    one qubit, :class:`~qiskit.circuit.library.ZFeatureMap` is created instead.

    If the number of qubits is ``None``, then the number of qubits is derived from the feature map
    or ansatz. Both the feature map and ansatz in this case must have the same number of qubits.
    If the number of qubits of the feature map is not the same as the number of qubits of
    the ansatz, an error is raised. If only one of the feature map and ansatz are ``None``, then
    :class:`~qiskit.circuit.library.ZZFeatureMap` or :class:`~qiskit.circuit.library.RealAmplitudes`
    are created respectively.

    If all the parameters are none an error is raised.

    Args:
        num_qubits: Number of qubits.
        feature_map: A feature map.
        ansatz: An ansatz.

    Returns:
        A tuple of number of qubits, feature map, and ansatz. All are not none.

    Raises:
        ValueError: If correct values can not be derived from the parameters.
    """

    # check num_qubits, feature_map, and ansatz
    if num_qubits in (0, None) and initial_state is None and ansatz is None:
        raise ValueError(
            "Need at least one of number of qubits, initial state, or ansatz!"
        )

    if num_qubits not in (0, None):
        if initial_state is None:
            initial_state = default_initial_state(num_qubits)
        if ansatz is None:
            ansatz = EfficientSU2(num_qubits, 
                                  su2_gates=['rz','rx'],
                                  entanglement='linear',
                                  )
    else:
        if initial_state is not None and ansatz is not None:
            if initial_state.num_qubits > ansatz.num_qubits:
                raise ValueError(
                    f"The number of qubit in the initial state circuit ({initial_state.num_qubits}) "
                    f"exceeds that of the ansatz ({ansatz.num_qubits})!"
                )
            num_qubits = ansatz.num_qubits
        else:
            num_qubits = ansatz.num_qubits
            initial_state = default_initial_state(num_qubits)

    return num_qubits, initial_state, ansatz


def _adjust_num_qubits(circuit: QuantumCircuit, circuit_name: str, num_qubits: int) -> None:
    """
    Tries to adjust the number of qubits of the circuit by trying to set ``num_qubits`` properties.

    Args:
        circuit: A circuit to adjust.
        circuit_name: A circuit name, used in the error description.
        num_qubits: A number of qubits to set.

    Raises:
        ValueError: if number of qubits can't be adjusted.

    """
    try:
        circuit.num_qubits = num_qubits
    except AttributeError as ex:
        raise ValueError(
            f"The number of qubits {circuit.num_qubits} of the {circuit_name} does not match "
            f"the number of qubits {num_qubits}, and the {circuit_name} does not allow setting "
            "the number of qubits using `num_qubits`."
        ) from ex
    
def default_initial_state(num_qubits) -> QuantumCircuit:
    if not num_qubits % 2 == 0:
        raise ValueError('num qubits must be even to use the MaxMixedState') 
    
    half_qubits = round(num_qubits/2)
    circ = QuantumCircuit(num_qubits)
    for qubit in range(0, half_qubits):
        circ.h(qubit)
        circ.cx(qubit, qubit+half_qubits)
        circ.reset(qubit+half_qubits)
    return circ