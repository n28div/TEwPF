import pandas as pd

import os
import joblib
import argparse

from bpm_detection.dataset import load_dataset
from bpm_detection import BPMDetector
from bpm_detection.metrics import accuracy
from bpm_detection.priors import PRIORS
from bpm_detection.estimators import ESTIMATORS

import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, nargs="+", choices=["beatles", "rwc_popular"], required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--its", type=int, default=10)
parser.add_argument("--cv", type=int, default=5)

if __name__ == "__main__":
  args = parser.parse_args()
  data_df = pd.concat([load_dataset(d) for d in args.dataset])

  X = [(row.time, row.duration) for _, row in data_df.iterrows()]
  y = data_df.bpm.to_numpy()
  X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      train_size=0.75,
                                                      random_state=args.seed)


  def objective(trial):   
    detector = BPMDetector(
      min_bpm=trial.suggest_int("min_bpm", 10, 50),
      max_bpm=trial.suggest_int("max_bpm", 180, 300),
      bpm_step=trial.suggest_float("bpm_step", 0.01, 1.0),
      smooth_time_window=trial.suggest_float("smooth_time_window", 0.5, 10),
      smooth_bpm_window=trial.suggest_float("smooth_bpm_window", 0.5, 10),
      bpm_prior=trial.suggest_categorical("bpm_prior", list(PRIORS.keys())),
      estimator=trial.suggest_categorical("estimator", list(ESTIMATORS.keys())),
      workers=os.cpu_count()
    )

    score =  cross_val_score(estimator=detector, 
                             X=X_train, 
                             y=y_train, 
                             scoring=make_scorer(accuracy, greater_is_better=True, accuracy_type="1"),
                             cv=args.cv,
                             n_jobs=-1).mean()
    
    return score

  study = optuna.create_study()
  study.optimize(objective, n_trials=args.its)
  study.trials_dataframe().to_csv(args.out)