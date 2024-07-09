import numpy as np
from .hmm import ClassicalHMM

def ClassicalVaR(hmm: ClassicalHMM, 
                 num_time_steps: int,
                 num_samples: int,
                 growths: list[float],
                 loss: float,
                 ):
    """
    The function ClassicalVaR calculates the Value at Risk (VaR) using a classical Hidden Markov Model
    (HMM) approach based on growths, loss threshold, and number of samples.
    
    :param hmm: ClassicalHMM is a class representing a classical Hidden Markov Model
    :type hmm: ClassicalHMM
    :param num_time_steps: The `num_time_steps` parameter represents the
    number of time steps for which the model will generate observations and calculate the total growth
    based on the provided growth rates
    :type num_time_steps: int
    :param num_samples: The `num_samples` parameter represents the number of samples to generate in 
    order to estimate the Value at Risk (VaR). It determines how many times the simulation will be run 
    to calculate the proportion of samples where the total growth is less than the specified loss.
    :type num_samples: int
    :param growths: The `growths` parameter represents a list of growth values corresponding to different observations.
    :type growths: list[float]
    :param loss: The `loss` parameter represents the threshold value for total growth. If the total growth 
    calculated for a sample is less than this threshold value, it is considered a loss and `loss_sample` is 
    incremented. The function then returns the proportion of samples that resulted in growth less than the loss.
    :type loss: float
    :return: the proportion of samples where the total growth is less than the specified loss value.
    """
    model = hmm.model
    loss_sample = 0
    for _ in range(num_samples):
        total_growth = 0
        observations = model.sample(num_time_steps)[0]
        for observation in observations:
            total_growth += growths[observation[0]]
        if total_growth < loss:
            loss_sample += 1

    return loss_sample / num_samples