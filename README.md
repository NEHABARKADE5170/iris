# Iris Classifier (Streamlit)

Predict iris species (setosa, versicolor, virginica) from sepal and petal measurements.

## Run (Windows / VS Code terminal)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Files
- `model.py`  data loading, 5 models, training, evaluation, single and CSV prediction
- `app.py`    Streamlit app with 5 pages: Predict, Batch predict (CSV), Model comparison, Explore data, Learn the basics
- `requirements.txt`

## Features
- Choose the model, test-set size and random seed from the sidebar
- Live prediction with probability bars and the flower plotted among the dataset
- CSV upload for batch predictions with downloadable results
- Cross-validated comparison of all models, confusion matrix, per-class metrics, misclassified flowers
