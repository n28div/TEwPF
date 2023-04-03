import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def plot_tatum_f(bpm: int, ax: Axes = None) -> Axes:
  """
  Plot the function used to detect the tatum.

  Args:
      bpm (int): BPM value that is plotted.
      ax (Axes, optional): Axes on which the plot is drawn. Defaults to None.

  Returns:
      plt.Axes: Axes with BPM plotted
  """
  x = np.linspace(0, 5, 1000)
  y = np.cos(((bpm * np.pi) / 30)  * x)**4
  y += np.cos(((bpm * np.pi) / 120)  * x)**4

  if ax is None:
    _, ax = plt.subplots(figsize=(10, 3))

  ax.set_title(f"$f$ at {bpm} BPM")
  ax.plot(x, y, label="$f$")

  beat_x = np.arange(0, 5, (1 / (bpm / 60)))
  ax.vlines(beat_x, 0, y.max(), linestyles="--", colors="r", label="beat")
  ax.set_xlabel("time (s)")
  ax.set_ylabel("amplitude")
  
  return ax