"""Model-family registry for the raw-feature classifiers: one place that
knows how to build, save, and load each supported classifier type, so
train_raw_classifier.py and voting_raw_agent.py don't need to duplicate
per-library special-casing. Add a new family here and it's immediately
usable by both the trainer and the voting agent -- no changes needed
elsewhere.

Each family is saved as 26 files (one per letter) under
<family>_raw_models/<letter>.<ext>, mirroring approach/catboot's original
catboost_raw_models/ layout.
"""
from __future__ import annotations

import os

FAMILIES = ["catboost", "xgboost", "lightgbm", "random_forest", "logreg"]


def model_dir(models_root: str, family: str) -> str:
    return os.path.join(models_root, f"{family}_raw_models")


def build_classifier(family: str, seed: int):
    if family == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                                    loss_function="Logloss", random_seed=seed, verbose=False)
    if family == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                              eval_metric="logloss", random_state=seed, verbosity=0)
    if family == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                               random_state=seed, verbosity=-1)
    if family == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, max_depth=12, random_state=seed, n_jobs=-1)
    if family == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=seed)
    raise ValueError(f"unknown family: {family}")


def file_ext(family: str) -> str:
    return {"catboost": "cbm", "xgboost": "json", "lightgbm": "txt",
            "random_forest": "joblib", "logreg": "joblib"}[family]


def save_classifier(clf, family: str, path: str):
    if family == "catboost":
        clf.save_model(path)
    elif family == "xgboost":
        clf.save_model(path)
    elif family == "lightgbm":
        clf.booster_.save_model(path)
    else:  # random_forest, logreg -- plain sklearn estimators
        import joblib
        joblib.dump(clf, path)


def load_classifier(family: str, path: str):
    if family == "catboost":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier()
        clf.load_model(path)
        return clf
    if family == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier()
        clf.load_model(path)
        return clf
    if family == "lightgbm":
        import lightgbm as lgb
        return lgb.Booster(model_file=path)
    # random_forest, logreg
    import joblib
    return joblib.load(path)


def predict_proba_positive(family: str, clf, X):
    """Returns P(class=1) as a 1-D array, regardless of the underlying
    library's exact API shape."""
    if family == "lightgbm":
        # lightgbm.Booster (not the sklearn wrapper) predicts P(class=1) directly
        return clf.predict(X)
    return clf.predict_proba(X)[:, 1]
