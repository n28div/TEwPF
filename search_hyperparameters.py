import pandas as pd

import os
import joblib
import argparse

from bpm_detection.dataset import load_dataset
from bpm_detection import BPMDetector
from bpm_detection.metrics import accuracy
from bpm_detection.priors import PRIORS
from bpm_detection.estimators import ESTIMATORS

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, nargs="+", choices=["beatles", "rwc_popular"], required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--its", type=int, default=10)
parser.add_argument("--cv", type=int, default=10)

if __name__ == "__main__":
  args = parser.parse_args()
  data_df = pd.concat([load_dataset(d) for d in args.dataset])

  X = [(row.time, row.duration) for _, row in data_df.iterrows()]
  y = data_df.bpm.to_numpy()
  X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      train_size=0.75,
                                                      random_state=args.seed)

  opt = BayesSearchCV(
    BPMDetector(workers=os.cpu_count()),
    {
      "min_bpm": Real(10, 50),
      "max_bpm": Real(180, 300),
      "bpm_step": Real(0.01, 1.0),
      "smooth_time_window": Real(0.5, 10),
      "smooth_bpm_window": Real(0.5, 10),
      "bpm_prior": Categorical(list(PRIORS.keys())),
      "estimator": Categorical(list(ESTIMATORS.keys())),
    },
    scoring=make_scorer(accuracy, greater_is_better=True, accuracy_type="1"),
    n_iter=args.its,
    verbose=True,
    cv=args.cv,
    refit=False,
    random_state=args.seed)

  # executes bayesian optimization
  opt.fit(X_train, y_train)
  
  joblib.dump(opt, args.out)

  print(pd.concat([
    pd.DataFrame(opt.cv_results_["params"]), 
    pd.DataFrame(opt.cv_results_["mean_test_score"], 
    columns=["Accuracy"])], axis=1))