import unittest
from QHMM_AE import ClassicalVaR
from unittest.mock import MagicMock

# Assuming the function ClassicalVaR is already defined

# Mock class for ClassicalHMM
class MockHMM:
    def __init__(self, observations):
        self.model = MagicMock()
        self.model.sample = MagicMock(return_value=(observations, None))

class TestClassicalVaR(unittest.TestCase):
    def test_var_calculation(self):
        # Define the observations for the mock HMM
        observations = [[0], [1], [0], [1], [2]]
        
        # Create a mock HMM instance with these observations
        hmm = MockHMM(observations)
        
        # Define other parameters
        num_time_steps = 5
        num_samples = 10
        growths = [0.1, -0.2, 0.05]
        loss = 0.0
        
        # Call the function
        result = ClassicalVaR(hmm, num_time_steps, num_samples, growths, loss)
        
        # Check the result
        self.assertEqual(result, 1.0)

    def test_var_no_loss(self):
        # Define the observations for the mock HMM
        observations = [[0], [0], [0], [0], [0]]
        
        # Create a mock HMM instance with these observations
        hmm = MockHMM(observations)
        
        # Define other parameters
        num_time_steps = 5
        num_samples = 10
        growths = [0.1, 0.1, 0.1]
        loss = 1.0
        
        # Call the function
        result = ClassicalVaR(hmm, num_time_steps, num_samples, growths, loss)
        
        # Check the result
        self.assertEqual(result, 0.0)

    def test_var_mixed(self):
        # Define the observations for the mock HMM
        observations = [[0], [1], [0], [1], [2], [2], [0], [1], [0], [1]]
        
        # Create a mock HMM instance with these observations
        hmm = MockHMM(observations)
        
        # Define other parameters
        num_time_steps = 5
        num_samples = 10
        growths = [0.1, 0.2, 0.05]
        loss = 0.1
        
        # Call the function
        result = ClassicalVaR(hmm, num_time_steps, num_samples, growths, loss)
        
        # Check the result
        self.assertTrue(0 <= result <= 1)

if __name__ == '__main__':
    unittest.main()