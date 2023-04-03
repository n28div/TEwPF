import numpy as np

def median(local_bpms: np.array) -> float:
  """
  Estimate the BPM as the median of local BPMs.

  Args:
      local_bpms (np.array): Local BPMs array.

  Returns:
      float: Estimated global BPM.
  """
  return np.median(local_bpms)

def histogram(local_bpms: np.array) -> float:
  """
  Estimate BPM by using an histogram of the local bpms in bins.
  The bin size is automatically estimated by numpy.

  Args:
      local_bpms (np.array): Local BPMs array

  Returns:
      float: Estimated global BPM.
  """
  count, bins = numpy.histogram(local_bpms)
  return bins[count.argmax()]

ESTIMATORS = {
  "median": median,
  "histogram": histogram,
}