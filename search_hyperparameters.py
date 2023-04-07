import pandas as pd
import numpy as np

import os
import joblib
import argparse
import random

from bpm_detection.dataset import load_dataset
from bpm_detection import PeriodicBPMDetector, OptimisationBPMDetector
from bpm_detection.metrics import accuracy
from bpm_detection.priors import PRIORS
from bpm_detection.estimators import ESTIMATORS

import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer

def search_periodic(X, y, iterations, cv):
  results = {}
  for estimator in ESTIMATORS:
    for prior in PRIORS:
      def objective(trial):
        if prior == "gaussian":
          bpm_kwargs = {
            "mu": trial.suggest_int("gaussian_prior_mu", 50, 250),
            "sigma": trial.suggest_float("gaussian_prior_sigma", 1, 1000),
          }
        elif prior == "parncutt":
          bpm_kwargs = {
            "mu": trial.suggest_int("parncutt_prior_mu", 10, 300),
            "sigma": trial.suggest_float("parncutt_prior_sigma", 0.1, 10),
          }
        elif prior == "resonance":
          bpm_kwargs = {
            "bpm_ext": trial.suggest_int("resonance_prior_bpm_ext", 50, 250),
            "beta": trial.suggest_float("resonance_prior_beta", 0.1, 10),
          }
        else:
          bpm_kwargs = {}

        detector = PeriodicBPMDetector(
          min_bpm=trial.suggest_int("min_bpm", 10, 50),
          max_bpm=trial.suggest_int("max_bpm", 240, 300),
          bpm_step=trial.suggest_float("bpm_step", 0.1, 1.0),
          alpha=trial.suggest_float("alpha", 0.0, 1.0),
          beta=trial.suggest_float("beta", 0.0, 1.0),
          gamma=trial.suggest_float("gamma", 0.0, 1.0),
          smooth_time_window=trial.suggest_float("smooth_time_window", 0.5, 10),
          smooth_bpm_window=trial.suggest_float("smooth_bpm_window", 0.5, 10),
          bpm_prior=prior,
          estimator=estimator,
          workers=os.cpu_count(),
          prior_kwargs=bpm_kwargs
        )

        def accuracy_wrapper(y, y_pred, **kwargs):
          y_pred = [elem[0] for elem in y_pred]
          return accuracy(y, y_pred, "1")

        score =  cross_val_score(estimator=detector, X=X, y=y, 
                                scoring=make_scorer(accuracy_wrapper, greater_is_better=True),
                                cv=cv, n_jobs=-1).mean()
        
        return score

      study = optuna.create_study(direction="maximize")
      study.optimize(objective, n_trials=iterations)
      results[f"{estimator}_{prior}"] = study.trials_dataframe()
  return results

def search_optimisation(X, y, iterations, cv):
  results = {}
  for estimator in ESTIMATORS:
    def objective(trial):
      detector = OptimisationBPMDetector(
        min_bpm=trial.suggest_int("min_bpm", 10, 50),
        max_bpm=trial.suggest_int("max_bpm", 180, 300),
        alpha=trial.suggest_float("alpha", 0.0, 1.0),
        beta=trial.suggest_float("beta", 0.0, 1.0),
        gamma=trial.suggest_float("gamma", 0.0, 1.0),
        time_window=trial.suggest_int("time_window", 1, 10),
        estimator=estimator,
        workers=os.cpu_count(),
      )

      def accuracy_wrapper(y, y_pred, **kwargs):
        y_pred = [elem[0] for elem in y_pred]
        return accuracy(y, y_pred, "1")

      score = cross_val_score(estimator=detector, X=X, y=y, 
                              scoring=make_scorer(accuracy_wrapper, greater_is_better=True),
                              cv=cv, n_jobs=-1).mean()
      
      return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=iterations)
    results[estimator] = study.trials_dataframe()
  return results

MODEL_SEARCH_FN = {
  "periodic": search_periodic,
  "optimisation": search_optimisation,
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, nargs="+", choices=["beatles", "rwc_popular"], required=True)
parser.add_argument("--model", type=str, choices=list(MODEL_SEARCH_FN.keys()), required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--its", type=int, default=10)
parser.add_argument("--cv", type=int, default=5)

if __name__ == "__main__":
  args = parser.parse_args()

  data_path = os.path.join(args.out, "data.pkl")
  if not os.path.exists(args.out):
    os.mkdir(args.out)

  if os.path.exists(data_path):
    X_train, X_test, y_train, y_test = joblib.load(data_path)  
  else:
    data_df = pd.concat([load_dataset(d) for d in args.dataset])
    X = [(row.time, row.duration) for _, row in data_df.iterrows()]
    y = data_df.bpm.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8,
                                                        shuffle=True,
                                                        random_state=args.seed)
    joblib.dump((X_train, X_test, y_train, y_test), data_path)

  results = MODEL_SEARCH_FN[args.model](X_train, y_train, args.its, args.cv)
  for name, df in results.items():
    df.to_csv(os.path.join(args.out, f"{name}.csv"))
