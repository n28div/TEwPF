import numpy as np
from madmom.features.tempo import TempoEstimationProcessor

def estimate_onset(times: np.array, fs: int = 200) -> np.array:
  """
  Estimate an onset signal from the specified times, using
  the specified sampling frequency.

  Args:
      times (np.array): Time annotations
      fs (int, optional): Sampling frequency. Defaults to 200.

  Returns:
      np.array: Onset signal
  """
  signal = np.zeros(int(times.max() * fs) + 1)
  signal[(times * Fs).astype(int)] = 1
  return signal
  
def comb_estimate(times: np.array, fps: int = 200) -> float:
  """
  Estimate BPM using the method from [1].

  [1] Sebastian Böck, Florian Krebs and Gerhard Widmer, 
    "Accurate Tempo Estimation based on Recurrent Neural Networks and Resonating Comb Filters" ISMIR, 2015

  Args:
      times (np.array): Times annotations.
      fps (int, optional): 
        Frame Per Second. See madmom library for more info. Defaults to 200.
        fps is used as sampling frequency to estimate the annotations
        onset.

  Returns:
      float: BPM estimate.
  """
  onset = estimate_onset(times, fps)
  processor = TempoEstimationProcessor(fps=fps, method="comb")
  return proc(onset)[0, 0]

def acf_estimate(times: np.array, fps: int = 200) -> float:
  """
  Estimate BPM using the method from [1].

  [1] Sebastian Böck and Markus Schedl, 
    "Enhanced Beat Tracking with Context-Aware Neural Networks",
    Proceedings of the 14th International Conference on Digital Audio Effects (DAFx), 2011.

  Args:
      times (np.array): Times annotations.
      fps (int, optional): 
        Frame Per Second. See madmom library for more info. Defaults to 200.
        fps is used as sampling frequency to estimate the annotations
        onset.

  Returns:
      float: BPM estimate.
  """
  onset = estimate_onset(times, fps)
  processor = TempoEstimationProcessor(fps=fps, method="acf")
  return proc(onset)[0, 0]