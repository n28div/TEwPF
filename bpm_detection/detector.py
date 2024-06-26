from __future__ import annotations
from typing import Union, List, Tuple, Dict

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from more_itertools import sliding_window
from tqdm import tqdm
import pyswarms as ps

from bpm_detection.priors import PRIORS
from bpm_detection.estimators import ESTIMATORS


class PeriodicBPMDetector(BaseEstimator, ClassifierMixin):
  def __init__(self, 
               min_bpm: int = 30, 
               max_bpm: int = 300, 
               bpm_step: float = 0.1,
               alpha: float = 1,
               beta: float = 1,
               gamma: float = 1,
               smooth_time_window: float = 3, 
               smooth_bpm_window: float = 1, 
               bpm_prior: str = "uniform",
               estimator: str = "median",
               workers: int = 1,
               verbose: bool = False,
               prior_kwargs: Dict = {}):
    """
    Initialise the BPM detector with provided parameters.

    Args:
        min_bpm (int, optional): Minimum BPM checked. Defaults to 30.
        max_bpm (int, optional): Maximum BPM checked. Defaults to 300.
        bpm_step (float, optional): Step between each point in the range of BPMs checked. Defaults to 0.1.
        alpha (float, optional): Weight for the tatum cosine function. Defaults to 1.
        beta (float, optional): Weight for double tatum cosine function. Defaults to 1.
        gamma (float, optional): Weight for half tatum cosine function. Defaults to 1.
        smooth_time_window (float, optional): Number of annotations smoothed. Defaults to 3.
        smooth_bpm_window (float, optional): Number of BPMs smoothed. Defaults to 1.
        bpm_prior (str, optional): Prior applied to BPMs. Defaults to "uniform".
        estimator (str, optional): Estimator used to extract the global BPM. Defaults to "median".
        workers (int, optional): Number of workers for parallel execution. Defaults to 1.
        verbose (bool, optional): Show progress bar while predicting dataset. Defaults to False.
        prior_kwargs (Dict, optional): Arguments for prior. Defaults to {}.
    """
    self.min_bpm = min_bpm
    self.max_bpm = max_bpm
    self.bpm_step = bpm_step

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma

    self.smooth_time_window = smooth_time_window
    self.smooth_bpm_window = smooth_bpm_window
    
    self.bpm_prior = bpm_prior
    assert bpm_prior in PRIORS, f"Prior {bpm_prior} not in {PRIORS.keys()}."
    self.bpm_prior_func = PRIORS[bpm_prior]
    self.prior_kwargs = prior_kwargs

    self.estimator = estimator
    assert estimator in ESTIMATORS, f"Estimator {estimator} not in {ESTIMATORS.keys()}."
    self.estimator_func = ESTIMATORS[estimator]

    self.workers = workers
    self.verbose = verbose


  def fit(self, *args, **kwars):
    return self


  def predict(self, X: Union[np.array, List[Tuple[List[float], List[float]]]]) -> List[float]:
    """
    Predict the BPM for a set of annotations.

    Args:
        X (Union[np.array, List[Tuple[List[float], List[float]]]]): Input annotations
          expressed as a list of tuples in the form (time annotations, duration annotations)

    Returns:
        List[float]: BPM detected for each sample.
    """
    locals = []
    globals = []
    bpms = np.arange(self.min_bpm, self.max_bpm, self.bpm_step)

    def estimate(x: Tuple[np.array, np.array]) -> Tuple[float, Tuple[Union[float, np.array]]]:
      times, duration = x
      times = times - times[0]
      
      tiled_times = np.tile(times, (bpms.shape[0], 1))
      score = self.alpha * np.cos(((bpms.reshape(-1, 1) * np.pi) / 60)  * tiled_times) ** 4
      score += self.beta * np.cos(((bpms.reshape(-1, 1) * np.pi) / 120)  * tiled_times) ** 4
      score += self.gamma * np.cos(((bpms.reshape(-1, 1) * np.pi) / 240)  * tiled_times) ** 4
           
      score_meter = [3, 4]
      score = np.stack([
        score + ((np.cos(((bpms.reshape(-1, 1) * np.pi) / (60 * m))  * tiled_times) ** 2) * duration)
        for m in score_meter])

      score = np.apply_along_axis(gaussian_filter1d, 2, score, sigma=(1 / self.bpm_step) * self.smooth_bpm_window, truncate=1)
      score = np.apply_along_axis(gaussian_filter1d, 1, score, sigma=self.smooth_time_window, truncate=1)

      meter_idx = score.sum(axis=2).max(axis=1).argmax()
      score = score[meter_idx]
      meter = score_meter[meter_idx]

      prior = self.bpm_prior_func(bpms, **self.prior_kwargs)
      prior = (prior - prior.min() + 1e-20) / (prior.max() - prior.min() + 1e-20)
      score *= prior.reshape(-1, 1)      

      loc = bpms[score.cumsum(axis=1).argmax(axis=0)]
      glob = self.estimator_func(loc)
      return glob, (loc, bpms, score, meter)

    with ThreadPoolExecutor(self.workers) as executor:
        results = [x for x in tqdm(executor.map(estimate, X), total=len(X), disable=not self.verbose)]
    
    return results


class OptimisationBPMDetector(BaseEstimator, ClassifierMixin):
  def __init__(self, 
               min_bpm: int = 30, 
               max_bpm: int = 300, 
               alpha: float = 1,
               beta: float = 1,
               gamma: float = 1,
               time_window: int = 3,
               estimator: str = "median",
               workers: int = 1,
               verbose: bool = False):
    """
    Initialise the BPM detector with provided parameters.

    Args:
        min_bpm (int, optional): Minimum BPM checked. Defaults to 30.
        max_bpm (int, optional): Maximum BPM checked. Defaults to 300.
        alpha (float, optional): Weight for the tatum cosine function. Defaults to 1.
        beta (float, optional): Weight for double tatum cosine function. Defaults to 1.
        gamma (float, optional): Weight for half tatum cosine function. Defaults to 1.
        time_window (int, optional): Number of annotations in a time window. Defaults to 3.
        workers (int, optional): Number of workers for parallel execution. Defaults to 1.
        estimator (str, optional): Estimator used to extract the global BPM. Defaults to "median".
        verbose (bool, optional): Show progress bar while predicting dataset. Defaults to False.
    """
    self.min_bpm = min_bpm
    self.max_bpm = max_bpm
    self.time_window = time_window
    self.workers = workers
    self.verbose = verbose

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma

    self.estimator = estimator
    assert estimator in ESTIMATORS, f"Estimator {estimator} not in {ESTIMATORS.keys()}."
    self.estimator_func = ESTIMATORS[estimator]

  def fit(self, *args, **kwars):
    return self

  def predict(self, X: Union[np.array, List[Tuple[List[float], List[float]]]]) -> List[float]:
    """
    Predict the BPM for a set of annotations.

    Args:
        X (Union[np.array, List[Tuple[List[float], List[float]]]]): Input annotations
          expressed as a list of tuples in the form (time annotations, duration annotations)

    Returns:
        List[float]: BPM detected for each sample.
    """

    def f(bpm, times, duration):
      fitness = self.alpha * np.cos(((bpm * np.pi) / 60)  * times) ** 4
      fitness += self.beta * np.cos(((bpm * np.pi) / 120)  * times) ** 4
      fitness += self.gamma * np.cos(((bpm * np.pi) / 30)  * times) ** 4
      return -1 * fitness.sum()

    def estimate(X):
      times, duration = X
      times = times - times[0]

      x0 = (self.max_bpm - self.min_bpm) / 2
      tw = min(len(times), self.time_window)

      est = [minimize(f, x0,
                      args=(t, d), 
                      options={"xatol": 0, "fatol": 0},
                      bounds=[(self.min_bpm, self.max_bpm)], 
                      method="nelder-mead").x[0]
        for t, d in zip(sliding_window(times, tw), 
                        sliding_window(duration, tw))]
      glob = self.estimator_func(est)
      return glob, est
    
    with ThreadPoolExecutor(self.workers) as executor:
        results = [x for x in tqdm(executor.map(estimate, X), total=len(X), disable=not self.verbose)]
    
    return results


class PsoBPMDetector(BaseEstimator, ClassifierMixin):
  def __init__(self, 
               min_bpm: int = 30, 
               max_bpm: int = 300, 
               alpha: float = 1,
               beta: float = 1,
               gamma: float = 1,
               time_window: int = 3,
               c1: float = 0.5,
               c2: float = 0.5,
               w: float = 0.5,
               estimator: str = "median",
               workers: int = 1,
               verbose: bool = False):
    """
    Initialise the BPM detector with provided parameters.

    Args:
        min_bpm (int, optional): Minimum BPM checked. Defaults to 30.
        max_bpm (int, optional): Maximum BPM checked. Defaults to 300.
        alpha (float, optional): Weight for the tatum cosine function. Defaults to 1.
        beta (float, optional): Weight for double tatum cosine function. Defaults to 1.
        gamma (float, optional): Weight for half tatum cosine function. Defaults to 1.
        time_window (int, optional): Number of annotations in a time window. Defaults to 3.
        c1 (float, optional): Cognitive parameter for PSO. Defaults to 0.5.
        c2 (float, optional): Social parameter for PSO. Defaults to 0.5.
        w (float, optional): Inertia parameter for PSO. Defaults to 0.5.
        workers (int, optional): Number of workers for parallel execution. Defaults to 1.
        estimator (str, optional): Estimator used to extract the global BPM. Defaults to "median".
        verbose (bool, optional): Show progress bar while predicting dataset. Defaults to False.
    """
    self.min_bpm = min_bpm
    self.max_bpm = max_bpm
    self.time_window = time_window
    self.workers = workers
    self.verbose = verbose

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma

    self.c1 = c1
    self.c2 = c2
    self.w = w

    self.estimator = estimator
    assert estimator in ESTIMATORS, f"Estimator {estimator} not in {ESTIMATORS.keys()}."
    self.estimator_func = ESTIMATORS[estimator]

  def fit(self, *args, **kwars):
    return self

  def predict(self, X: Union[np.array, List[Tuple[List[float], List[float]]]]) -> List[float]:
    """
    Predict the BPM for a set of annotations.

    Args:
        X (Union[np.array, List[Tuple[List[float], List[float]]]]): Input annotations
          expressed as a list of tuples in the form (time annotations, duration annotations)

    Returns:
        List[float]: BPM detected for each sample.
    """

    def f(bpm, times, duration):
      fitness = self.alpha * np.cos(((bpm * np.pi) / 60)  * times) ** 4
      fitness += self.beta * np.cos(((bpm * np.pi) / 120)  * times) ** 4
      fitness += self.gamma * np.cos(((bpm * np.pi) / 30)  * times) ** 4
      return -1 * fitness.sum()

    def estimate(X):
      times, duration = X
      times = times - times[0]

      x0 = (self.max_bpm - self.min_bpm) / 2
      tw = min(len(times), self.time_window)

      opt = ps.single.GlobalBestPSO(
        n_particles=10, 
        dimensions=1,
        options={ "c1": self.c1, "c2": self.c2, "w": self.w }, 
        bounds=([self.min_bpm], [self.max_bpm]))

      est = [opt.optimize(f, iters=100, times=t, duration=d, verbose=False)[1]
        for t, d in zip(sliding_window(times, tw), 
                        sliding_window(duration, tw))]
      glob = self.estimator_func(est)
      return glob, est
    
    with ThreadPoolExecutor(self.workers) as executor:
        results = [x for x in tqdm(executor.map(estimate, X), total=len(X), disable=not self.verbose)]
    
    return results

