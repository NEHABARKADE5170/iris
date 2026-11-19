"""
app.py - Iris Flower Classifier (Streamlit)
Run:  streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model import FEATURES, MODELS, SPECIES, compare_models, load_data, predict_dataframe, predict_one, train

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="wide")
COLORS = {"setosa": "#e74c3c", "versicolor": "#3498db", "virginica": "#2ecc71"}

df = load_data()


@st.cache_resource
def get_trained(model_name, test_size, seed):
    return train(model_name, test_size, seed)


@st.cache_data
def get_comparison(test_size, seed):
    return compare_models(test_size, seed)


# ------------------------------------------------------------ sidebar
st.sidebar.title("🌸 Iris Classifier")
page = st.sidebar.radio("Go to", ["Predict", "Batch predict (CSV)", "Model comparison", "Explore data", "Learn the basics"])
st.sidebar.divider()
st.sidebar.subheader("Model settings")
model_name = st.sidebar.selectbox("Model", list(MODELS), index=0)
test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.2, 0.05,
                              help="Share of the 150 flowers held back to test the model.")
seed = st.sidebar.number_input("Random seed", 0, 9999, 42, help="Changes which flowers land in train vs test.")
res = get_trained(model_name, test_size, int(seed))
st.sidebar.metric("Test accuracy", f"{res['accuracy']:.1%}")
st.sidebar.caption(f"Trained on {res['n_train']} flowers, tested on {res['n_test']}.")

# =========================================================== PREDICT
if page == "Predict":
    st.title("Predict the species of an iris flower")
    st.write("Set the four measurements. The model returns the species it thinks this flower is.")

    PRESETS = {"Typical setosa": [5.0, 3.4, 1.5, 0.2], "Typical versicolor": [5.9, 2.8, 4.3, 1.3],
               "Typical virginica": [6.6, 3.0, 5.6, 2.0]}
    for i, f in enumerate(FEATURES):
        st.session_state.setdefault(f"val_{i}", float(df[f].median()))

    def set_preset(vals):
        for i, v in enumerate(vals):
            st.session_state[f"val_{i}"] = v

    pc = st.columns(len(PRESETS))
    for col, (label, vals) in zip(pc, PRESETS.items()):
        col.button(label, on_click=set_preset, args=(vals,), width="stretch")

    left, right = st.columns([1, 1.3])
    with left:
        values = []
        for i, f in enumerate(FEATURES):
            lo, hi = float(df[f].min()), float(df[f].max())
            values.append(st.slider(f, round(lo - 0.5, 1), round(hi + 0.5, 1), key=f"val_{i}", step=0.1))
        species, conf, probs = predict_one(res["model"], values)
        st.success(f"Prediction: **Iris {species}**  ({conf:.0%} confident)")
        pdf = pd.DataFrame({"species": list(probs), "probability": list(probs.values())})
        st.plotly_chart(px.bar(pdf, x="probability", y="species", orientation="h", color="species",
                               color_discrete_map=COLORS, range_x=[0, 1])
                        .update_layout(height=220, showlegend=False, margin=dict(t=10, b=10)), width="stretch")
    with right:
        st.subheader("Where your flower sits among the 150 known flowers")
        fig = px.scatter(df, x="petal length (cm)", y="petal width (cm)", color="species",
                         color_discrete_map=COLORS, opacity=0.6)
        fig.add_trace(go.Scatter(x=[values[2]], y=[values[3]], mode="markers", name="Your flower",
                                 marker=dict(symbol="star", size=20, color="black", line=dict(width=2, color="white"))))
        st.plotly_chart(fig.update_layout(height=420), width="stretch")
        st.caption("Petal size separates the three species better than sepal size does.")

# =========================================================== BATCH
elif page == "Batch predict (CSV)":
    st.title("Batch prediction from a CSV file")
    st.write("Upload a CSV with these four columns (names like `sepal_length`, `Sepal.Length` or "
             "`sepal length (cm)` all work): " + ", ".join(f"`{f}`" for f in FEATURES))
    sample = df[FEATURES].sample(10, random_state=1).reset_index(drop=True)
    st.download_button("Download a sample CSV to try", sample.to_csv(index=False).encode(), "sample_iris.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        try:
            out = predict_dataframe(res["model"], pd.read_csv(up))
        except Exception as e:
            st.error(f"Could not read that file: {e}")
        else:
            st.success(f"Predicted {len(out)} flowers using {model_name}.")
            c = st.columns(3)
            for col, s in zip(c, SPECIES):
                col.metric(f"Iris {s}", int((out.predicted_species == s).sum()))
            st.dataframe(out, width="stretch")
            st.download_button("Download predictions", out.to_csv(index=False).encode(), "iris_predictions.csv", "text/csv")

# =========================================================== COMPARISON
elif page == "Model comparison":
    st.title("Model comparison")
    st.caption("Test accuracy uses one train/test split. Cross-validation (CV) repeats the test 5 times on different "
               "splits, so it is a steadier estimate.")
    comp = get_comparison(test_size, int(seed))
    st.dataframe(comp.style.format({"Test accuracy": "{:.1%}", "CV mean accuracy": "{:.1%}", "CV std": "{:.3f}"}),
                 hide_index=True, width="stretch")
    st.plotly_chart(px.bar(comp, x="Model", y="CV mean accuracy", error_y="CV std", range_y=[0.8, 1.0])
                    .update_layout(height=340), width="stretch")

    st.subheader(f"Details for {model_name}")
    a, b = st.columns(2)
    with a:
        cm = res["confusion"]
        fig = px.imshow(cm, text_auto=True, x=SPECIES, y=SPECIES, color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual", color="Flowers"))
        st.plotly_chart(fig.update_layout(height=360, title="Confusion matrix"), width="stretch")
    with b:
        st.write("**Precision / recall / F1 per species**")
        st.dataframe(res["report"], width="stretch")
        st.write("**Flowers the model got wrong**")
        if res["wrong"].empty:
            st.write("None on this test set.")
        else:
            st.dataframe(res["wrong"], width="stretch")

# =========================================================== EXPLORE
elif page == "Explore data":
    st.title("Explore the dataset")
    st.write(f"{len(df)} flowers, 50 per species, 4 measurements each.")
    st.dataframe(df.drop(columns="target"), width="stretch", height=260)
    a, b = st.columns(2)
    xf = a.selectbox("X axis", FEATURES, index=2)
    yf = b.selectbox("Y axis", FEATURES, index=3)
    st.plotly_chart(px.scatter(df, x=xf, y=yf, color="species", color_discrete_map=COLORS)
                    .update_layout(height=400), width="stretch")
    c, d = st.columns(2)
    with c:
        st.plotly_chart(px.box(df, x="species", y=xf, color="species", color_discrete_map=COLORS)
                        .update_layout(height=340, showlegend=False), width="stretch")
    with d:
        corr = df[FEATURES].corr().round(2)
        st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                                  title="Feature correlation").update_layout(height=340), width="stretch")
    st.write("**Summary statistics**")
    st.dataframe(df.groupby("species")[FEATURES].mean().round(2), width="stretch")

# =========================================================== LEARN
else:
    st.title("Learn the basics")
    st.markdown("""
### 1. The problem
You have a flower and four measurements of it. You want to know which of three species it is.
This is **classification**: pick one label from a fixed list.

### 2. Features and labels
- **Features** (the inputs): sepal length, sepal width, petal length, petal width, all in cm.
- **Label** (the answer we want): the species, one of setosa, versicolor, virginica.

A model is a function that maps features to a label. We do not write its rules by hand. We show it examples
where the answer is known and it finds the rules itself. That is **training**.

### 3. Why we split the data
If we test the model on the same flowers it learned from, it can simply memorise them and look perfect.
So we hold some flowers back (the **test set**) and only use them at the end. Accuracy on flowers the model
has never seen is the honest number. The *Test set size* slider in the sidebar controls this split.

### 4. Measuring how good it is
- **Accuracy**: the share of test flowers predicted correctly.
- **Confusion matrix**: a table of actual vs predicted species. The diagonal is correct, everything else is a mistake.
  It shows *which* species get mixed up (usually versicolor and virginica, because they overlap).
- **Precision**: when the model says "virginica", how often is it right?
- **Recall**: of all real virginica flowers, how many did it find?
- **Cross-validation**: repeat the train/test split several times and average, so one lucky split cannot fool you.

### 5. The models in this app
- **K-Nearest Neighbors**: look at the 5 most similar known flowers and let them vote.
- **SVM**: draw the boundary that separates the species with the widest possible gap.
- **Logistic Regression**: weigh each feature, turn the total into a probability for each species.
- **Decision Tree**: a chain of yes/no questions such as "petal length < 2.5?".
- **Random Forest**: many decision trees voting together.

### 6. Why scaling
SVM, KNN and Logistic Regression compare sizes of numbers. A feature with big numbers would dominate one with small
numbers, so we first rescale every feature to a similar range (StandardScaler). Trees do not need this.
""")
