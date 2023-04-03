import numpy as np

def uniform_prior(bpms: np.array) -> np.array:
  """
  Assign to each BPM a uniform weight (all 1s).

  Args:
      bpms (np.array): BPMs to be scored.

  Returns:
      np.array: Uniform prior
  """
  return np.ones_like(bpms)

def gaussian_prior(bpms: np.array, mu: int = 100, sigma: float = 40) -> np.array:
  """
  Compute BPM prior by using a Gaussian distribution.

  Args:
      bpms (np.array): BPMs to be scored.
      mu (int, optional): Gaussian BPM mean. Defaults to 100.
      sigma (float, optional): Gaussian BPM deviation. Defaults to 40.

  Returns:
      np.array: Gaussian prior over the specified bpms
  """
  prior = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((bpms - mu)**2 / sigma**2))
  return prior

def parncutt_prior(bpms: np.array, mu: int = 100, sigma: float = 0.8) -> np.array:
  """
  Cognitive BPM prior as formalised by Parncutt in [1]

  [1] Parncutt, R. (1994). 
    A perceptual model of pulse salience and metrical accent in musical rhythms. 
    Music perception

  Args:
      bpms (np.array): BPMs to be scored.
      mu (int, optional): Prior mean. Defaults to 100.
      sigma (float, optional): Prior sigma. Defaults to 0.8.

  Returns:
      np.array: Parncutt prior over the specified bpms
  """
  saliency = np.exp(-0.5 * ((1 / sigma) * np.log10(bpms / mu))**2)
  return saliency

def resonance_prior(bpms: np.array, bpm_ext: int = 120, beta: float = 1.12):
  """
  Cognitive BPM prior as formalised by van Noorden & Moelants  in [1].

  [1] Van Noorden, L., & Moelants, D. (1999). 
    Resonance in the perception of musical pulse. 
    Journal of New Music Research

  Args:
    bpms (np.array): BPMs to be scored.
    bpm_ext (int, optional): External resonant BPM. Defaults to 120.
    beta (float, optional): Beta parameter. Defaults to 1.12.

  Returns:
    np.array: Resonance prior over the specified bpms
  """
  f_ext = 60 / bpm_ext
  f_0 = 60 / bpms
  resonance = (1 / np.sqrt((f_0**2 - f_ext**2)**2 + beta*(f_ext**2)))
  resonance = (resonance - resonance.min()) / (resonance.max() - resonance.min())
  return resonance

PRIORS = {
  "uniform": uniform_prior,
  "gaussian": gaussian_prior,
  "parncutt": parncutt_prior,
  "resonance": resonance_prior
}