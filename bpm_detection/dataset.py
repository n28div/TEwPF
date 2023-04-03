from typing import Tuple
import numpy as np
import pandas as pd

import mirdata
from mirdata.core import Track

def extract_beatles(track: Track) -> Tuple[str, np.array, np.array, int]:
  """
  Extract a track from the Beatles dataset

  Args:
      track (Track): Track from the Beatles dataset.

  Returns:
      Tuple[str, np.array, np.array, int]: 
        tuple composed of (track metadata, onsets, durations, BPM)
  """
  chords = track.chords.labels
  times, duration, beats = zip(
      *[(t, d, b) 
      for c, t, d, b in zip(track.chords.labels, track.chords.intervals[:, 0], 
                            track.chords.intervals[:, 1] - track.chords.intervals[:, 0], 
                            track.beats.times) 
      if c not in ["N", "X"]])

  beats_times = np.array(beats)
  bpm = np.median((1 / (beats_times[1:] - beats_times[:-1] )) * 60)

  return (f"The Beatles - {track.title}", np.array(times), np.array(duration), bpm)

def extract_rwc(track: Track) -> Tuple[str, np.array, np.array, int]:
  """
  Extract a track from the RWC dataset

  Args:
      track (Track): Track from the RWC dataset.

  Returns:
      Tuple[str, np.array, np.array, int]: 
        tuple composed of (track metadata, onsets, durations, BPM)
  """
  chords = track.chords.labels
  times, duration = zip(
    *[(t, d) 
    for c, t, d in zip(track.chords.labels, 
                       track.chords.intervals[:, 0], 
                       track.chords.intervals[:, 1] - track.chords.intervals[:, 0]) 
    if c not in ["N", "X"]])

  return (f"{track.artist} - {track.title}", 
          np.array(times), 
          np.array(duration), 
          int(track.tempo))

def load_dataset(name: str) -> pd.DataFrame:
  """
  Load a specific dataset from mirdata

  Args:
      name (str): Mirdata dataset

  Returns:
      pd.DataFrame: Dataset DataFrame
  """
  dataset = mirdata.initialize(name)
  dataset.download()

  if name == "beatles":
    extract = extract_beatles
  elif name == "rwc_popular":
    extract = extract_rwc

  data = []
  for _, t in dataset.load_tracks().items():
    try:
      data.append(extract(t))
    except:
      pass
    
  df = pd.DataFrame(data, 
                    columns=["title", "time", "duration", "bpm"])
  df["source"] = name
  return df
