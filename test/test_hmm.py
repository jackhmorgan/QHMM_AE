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
from hmmlearn import hmm
from QHMM_AE import ClassicalHMM

class TestClassicalHMM(unittest.TestCase):

    def setUp(self):
        # Example matrices for testing
        self.transition_matrix = np.array([[0.7, 0.3],
                                           [0.4, 0.6]])

        self.emission_matrix = np.array([[0.9, 0.1],
                                         [0.2, 0.8]])

        self.initial_probabilities = np.array([0.6, 0.4])

        self.hmm = ClassicalHMM(self.transition_matrix, self.emission_matrix, self.initial_probabilities)

    def test_initialization(self):
        # Check if the model parameters are set correctly
        self.assertEqual(self.hmm.num_states, 2)
        self.assertEqual(self.hmm.num_outcomes, 2)
        np.testing.assert_array_almost_equal(self.hmm.model.startprob_, self.initial_probabilities)
        np.testing.assert_array_almost_equal(self.hmm.model.transmat_, self.transition_matrix)
        np.testing.assert_array_almost_equal(self.hmm.model.emissionprob_, self.emission_matrix)

    def test_generate_distribution(self):
        # Generate a distribution
        num_time_steps = 3
        num_samples = 1000
        distribution = self.hmm.generate_distribution(num_time_steps, num_samples)

        # Check if the distribution is a dictionary
        self.assertIsInstance(distribution, dict)

        # Check if the keys are binary strings of correct length
        for key in distribution.keys():
            self.assertEqual(len(key), num_time_steps)

        # Check if the values are integers
        for value in distribution.values():
            self.assertIsInstance(value, int)

        # Check if the total number of samples matches
        total_samples = sum(distribution.values())
        self.assertEqual(total_samples, num_samples)

if __name__ == '__main__':
    unittest.main()
