"""
model.py - data loading, model training and prediction helpers for the Iris classifier.
No internet needed: the Iris dataset ships inside scikit-learn.
"""
import re

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
SPECIES = ["setosa", "versicolor", "virginica"]

# Each entry builds a fresh, untrained model. Scaling matters for SVM / KNN / Logistic Regression
# (they compare distances or weights), so those are wrapped in a pipeline with StandardScaler.
MODELS = {
    "SVM (RBF kernel)": lambda: make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    "K-Nearest Neighbors": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
}


def load_data():
    """Return a DataFrame with the 4 feature columns plus 'species' (text) and 'target' (0/1/2)."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    return df


def train(model_name, test_size=0.2, seed=42):
    """Split -> fit -> evaluate on the held-out test set."""
    df = load_data()
    X, y = df[FEATURES], df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    model = MODELS[model_name]().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    report = pd.DataFrame(classification_report(y_te, pred, target_names=SPECIES, output_dict=True)).T.round(3)
    return {
        "model": model, "accuracy": accuracy_score(y_te, pred),
        "confusion": confusion_matrix(y_te, pred), "report": report,
        "n_train": len(X_tr), "n_test": len(X_te),
        "wrong": X_te[pred != y_te].assign(actual=y_te[pred != y_te].map(dict(enumerate(SPECIES))),
                                          predicted=pd.Series(pred, index=X_te.index)[pred != y_te].map(dict(enumerate(SPECIES)))),
    }


def compare_models(test_size=0.2, seed=42, folds=5):
    """Test accuracy plus 5-fold cross-validation accuracy for every model."""
    df = load_data()
    X, y = df[FEATURES], df["target"]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for name in MODELS:
        scores = cross_val_score(MODELS[name](), X, y, cv=cv)
        rows.append({"Model": name, "Test accuracy": train(name, test_size, seed)["accuracy"],
                     "CV mean accuracy": scores.mean(), "CV std": scores.std()})
    return pd.DataFrame(rows).sort_values("CV mean accuracy", ascending=False).reset_index(drop=True)


def predict_one(model, values):
    """values: list of 4 numbers in FEATURES order -> (species, confidence, probabilities dict)."""
    X = pd.DataFrame([values], columns=FEATURES)
    proba = model.predict_proba(X)[0]
    i = int(proba.argmax())
    return SPECIES[i], float(proba[i]), dict(zip(SPECIES, proba))


def _norm(col):
    """'Sepal.Length', 'sepal_length_cm', 'sepal length (cm)' all become 'sepallength'."""
    return re.sub(r"[^a-z]", "", col.lower().replace("cm", ""))


def standardise_columns(df):
    """Map a user's CSV columns onto the 4 expected feature names. Raises ValueError if any is missing."""
    lookup = {_norm(f): f for f in FEATURES}
    mapping = {c: lookup[_norm(c)] for c in df.columns if _norm(c) in lookup}
    missing = [f for f in FEATURES if f not in mapping.values()]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}")
    return df.rename(columns=mapping)[FEATURES]


def predict_dataframe(model, df):
    """Predict every row of an uploaded CSV."""
    X = standardise_columns(df).apply(pd.to_numeric, errors="raise")
    proba = model.predict_proba(X)
    out = X.copy()
    out["predicted_species"] = [SPECIES[i] for i in proba.argmax(axis=1)]
    out["confidence"] = proba.max(axis=1).round(3)
    return out
