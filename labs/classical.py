"""D4: classical ML on bundled digits; preprocessing is fitted inside Pipeline/CV."""
import argparse
import json
import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, adjusted_rand_score
from .common import seed_all, save_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="dummy + logistic only")
    p.add_argument("--test", action="store_true", help="Final held-out evaluation after model selection")
    a = p.parse_args(); seed_all(a.seed)
    X, y = load_digits(return_X_y=True)
    # The official digits dataset is bundled with sklearn; no download needed.
    Xdev, Xtest, ydev, ytest = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    models = {"dummy": DummyClassifier(strategy="most_frequent"),
              "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}
    if not a.quick:
        models.update({"knn": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
            "svm": make_pipeline(StandardScaler(), SVC(C=3)),
            "naive_bayes": GaussianNB(),
            "tree": DecisionTreeClassifier(max_depth=8, random_state=a.seed),
            "forest": RandomForestClassifier(n_estimators=100, random_state=a.seed, n_jobs=4),
            "boosting": HistGradientBoostingClassifier(max_iter=60, random_state=a.seed)})
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = {}
    for name, model in models.items():
        values = cross_val_score(model, Xdev, ydev, cv=cv, scoring="f1_macro", n_jobs=1)
        scores[name] = {"macro_f1_mean": values.mean().item(), "macro_f1_std": values.std().item()}
        print(name, json.dumps(scores[name]), flush=True)
    best = max(scores, key=lambda k: scores[k]["macro_f1_mean"])
    metrics = {"dataset": "sklearn_digits_8x8", "split_seed": 42,
               "cv": scores, "selected": best, "n_dev": len(ydev), "n_test": len(ytest)}
    # Unsupervised exploration uses development data only, without labels in fit.
    projection = make_pipeline(StandardScaler(), PCA(n_components=0.95))
    Z = projection.fit_transform(Xdev)
    clusters = KMeans(n_clusters=10, n_init=10, random_state=a.seed).fit_predict(Z)
    metrics["pca_components"] = Z.shape[1]
    metrics["development_cluster_ari"] = adjusted_rand_score(ydev, clusters)
    if a.test:
        model = clone(models[best]).fit(Xdev, ydev)
        pred = model.predict(Xtest)
        metrics.update(test_accuracy=accuracy_score(ytest, pred),
                       test_macro_f1=f1_score(ytest, pred, average="macro"),
                       confusion_matrix=confusion_matrix(ytest, pred).tolist())
    save_run("classical", a, metrics)


if __name__ == "__main__":
    main()
