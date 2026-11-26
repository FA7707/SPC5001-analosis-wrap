import streamlit as st
import sys
sys.path.append('../')
import CNN
from image_settings import generate
import export
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd

run = 'Train'
st.markdown("<h1 style='text-align: center;'>Training model</h1>", unsafe_allow_html=True)

st.subheader("Telescope")
telescope = st.segmented_control("", ["Roman", "JWST", "HST"], default="Roman", required=True, label_visibility="collapsed")
st.divider()

st.subheader("Dataset")
ring_col, arc_col, multiple_col, non_col = st.columns(4)
with ring_col:
    st.markdown("**Ring**")
    ring = st.number_input("Images", key="Ring1", min_value=20, max_value=10000, value=20)
    ring_seed= st.number_input("Seed", key="Ring2", value=10)
    
with arc_col:
    st.markdown("**Arc**")
    arc_lens= st.number_input("Images", key="Arc1", min_value=20, max_value=10000, value=20)
    arc_lens_seed= st.number_input("Seed", key="Arc2", value=20)
    
with multiple_col:
    st.markdown("**Multiple**")
    multiple_lens = st.number_input("Images", key="Multiple1", min_value=20, max_value=10000, value=20)
    multiple_lens_seed = st.number_input("Seed", key="Multiple2", value=30)

with non_col:
    st.markdown("**Non lens**")
    non_lens= st.number_input("Images", key="Non lens1", min_value=20, max_value=10000, value=20)
    non_lens_seed= st.number_input("Seed", key="Non lens2", value=40)
st.divider()

st.subheader("Training")
learning_rate, epochs = st.columns(2)
with learning_rate:
    st.markdown("**Learning rate**")
    lr = st.number_input("Learning rate", key="Learning rate", min_value=0.0, max_value=1.0, value=0.002, label_visibility="collapsed",format="%.4f")

with epochs:
    st.markdown("**Epochs**")
    n_epochs= st.number_input("Number of Epochs", key="Epochs", min_value=1, max_value=10000, value=100, label_visibility="collapsed")

if st.button("Train", type="primary", width="stretch"):
    with st.spinner("Generating Einstein rings"):
        generate('Ring', ring, ring_seed, telescope, run)
    with st.spinner("Generating Arc lenses"):
        generate('Arc', arc_lens, arc_lens_seed, telescope, run)
    with st.spinner("Generating Multiple lenses"):
        generate('Multiple', multiple_lens, multiple_lens_seed, telescope, run)
    with st.spinner("Generating non lenses"):
        generate('Non lens', non_lens, non_lens_seed, telescope, run)
    st.success("Training images generated")
    with st.spinner("Exporting images"):
        export.export(run)
    st.success("Images exported")
    with st.spinner("Training images"):
        X_prescaled, y = CNN.load_files(run)
        noise = CNN.measuring_noise(X_prescaled)
        X = CNN.preprocessing(X_prescaled, noise)
        X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t, y_val = CNN.tensors(X, y)
        cnn = CNN.build_model()
        cnn, J_train, J_val, acc_train, acc_val = CNN.train(cnn, X_train_t, y_train_t, X_val_t, y_val_t, n_epochs, lr)
        matrix_fig, actual, preds = CNN.matrix(X_val_t, y_val, cnn, "Validation")
        curve_fig = CNN.plot_curves(J_train, J_val, acc_train, acc_val, f"CNN ({CNN.count_parameters(cnn)} parameters)")
        st.divider()
        st.pyplot(curve_fig)
        st.divider()
        st.pyplot(matrix_fig)
        st.divider()
        
    fig, axs = plt.subplots(4, 10, figsize=(15, 6))
    for row in range(4):
        chosen = X[y == row][:10]
        for col in range(10):
            axs[row, col].imshow(chosen[col], cmap="bone", origin="lower", vmin=-1, vmax=5)
            axs[row, col].axis("off")
    st.pyplot(fig)
    st.divider()
        
    accuracy = metrics.accuracy_score(actual, preds)
    precision = metrics.precision_score(actual, preds, average=None)
    recall = metrics.recall_score(actual, preds, average=None)
    f1 = metrics.f1_score(actual, preds, average=None)
    st.markdown(f"**Accuracy:** {accuracy * 100:.1f}%")
    results = pd.DataFrame({"Precision": precision, "Recall": recall, "F1 score": f1}, index=["Ring", "Arc_lens", "Multi_lens", "Non_lens"])
    st.dataframe((results * 100).round(1))
    
    


    

