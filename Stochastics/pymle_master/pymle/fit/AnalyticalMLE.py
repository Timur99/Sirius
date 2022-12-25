import numpy as np
from typing import List, Tuple

from pymle.fit.LikelihoodEstimator import LikelihoodEstimator
from pymle.TransitionDensity import TransitionDensity


class AnalyticalMLE(LikelihoodEstimator):
    def __init__(self,
                 sample: np.ndarray,
                 param_bounds: List[Tuple],
                 dt: float,
                 density: TransitionDensity):
        """
        Maximimum likelihood estimator based on some analytical represenation for the transition density.
        e.g. ExactDensity, EulerDensity, ShojiOzakiDensity, etc.
        :param sample: array, a single path draw from some theoretical model
        :param param_bounds: list of tuples, one tuple (lower,upper) of bounds for each parmater
        :param dt: float, the time step size
        :param density: transition density of some kind, attached to a model
        """
        super().__init__(sample=sample, param_bounds=param_bounds, dt=dt, model=density.model)
        self._density = density

    def log_likelihood_negative(self, params: np.ndarray) -> float:
        self._model.params = params
        return -np.sum(np.log(np.maximum(self._min_prob,
                                         self._density(x0=self._sample[:-1], 
                                                       xt=self._sample[1:],
                                                       t=self._dt))))

