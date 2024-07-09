Hidden Markov Models (HMM) are a statistical tool used to predict the likelihood of a sequence of outcomes that depend on an unkown hidden state. We are interested in estimating the Value at Risk (VaR) of a portfolio where the growth of said portfolio follows a HMM.

Quantum computers offer two distinct advantages over classical algorithms for this application. Reference [1] shows that a Quantum Hidden Markov Model (QHMM) can use fewer model parameters than a HMM to capture equally complex behavior. Once the QHMM model is trained, a quantum computer can calculate the VaR with a quadratic speedup over classical Monte Carlo methods via Quantum Amplitude Estimation (QAE) [2].

The existing QHMM implementations like [3] utilize mid-circuit measurement, which means that the distribution they produce is not unitary and thus cannot be used in an estimation with QAE. Additionally, our implementation differs from the literature in that we treat the initial state of the system as its own trainable parameter. This means that the learned transition circuit can be repeated for an arbitrary number of time steps without the need to retrain the ansatz.

This repository offers the TrainableQHMM class, which is a tool that crates a QHMM with mid-circuit measurement and the trainable initial state structure. The class also creates a unitary circuit that is compatible with QAE using the trained circuit. See example.ipynb for a complete workflow.

References:

1. Monras, A., Beige, A., & Wiesner, K. (2010). Hidden quantum Markov models and non-adaptive read-out of many-body states arXiv preprint arXiv:1002.2337.
2. Egger, D. J., Gutiérrez, R. G., Mestre, J. C., & Woerner, S. (2020). Credit risk analysis using quantum computers. IEEE Transactions on Computers, 70(12), 2136-2145.
3. Markov, V., Rastunkov, V., Deshmukh, A., Fry, D., & Stefanski, C. (2022). Implementation and learning of quantum hidden markov models. arXiv preprint arXiv:2212.03796.
