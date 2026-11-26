import streamlit as st
import sys
sys.path.append('../')
import CNN
from image_settings import generate
import export
import torch

run = 'Test'
st.markdown("<h1 style='text-align: center;'>Testing model</h1>", unsafe_allow_html=True)

st.subheader("Telescope")
telescope = st.segmented_control("", ["Roman", "JWST", "HST"], default="Roman", required=True, label_visibility="collapsed")
st.divider()

st.subheader("Dataset")
ring_col, arc_col, multiple_col, non_col = st.columns(4)
with ring_col:
    st.markdown("**Ring**")
    ring = st.number_input("Images", key="Ring1", min_value=20, max_value=10000, value=20)
    ring_seed= st.number_input("Seed", key="Ring2", value=50)
    
with arc_col:
    st.markdown("**Arc**")
    arc_lens= st.number_input("Images", key="Arc1", min_value=20, max_value=10000, value=20)
    arc_lens_seed= st.number_input("Seed", key="Arc2", value=60)
    
with multiple_col:
    st.markdown("**Multiple**")
    multiple_lens = st.number_input("Images", key="Multiple1", min_value=20, max_value=10000, value=20)
    multiple_lens_seed = st.number_input("Seed", key="Multiple2", value=70)

with non_col:
    st.markdown("**Non lens**")
    non_lens= st.number_input("Images", key="Non lens1", min_value=20, max_value=10000, value=20)
    non_lens_seed= st.number_input("Seed", key="Non lens2", value=80)

if st.button("Test", type="primary", width="stretch"):
    with st.spinner("Generating Einstein rings"):
        generate('Ring', ring, ring_seed, telescope, run)
    with st.spinner("Generating Arc lenses"):
        generate('Arc', arc_lens, arc_lens_seed, telescope, run)
    with st.spinner("Generating Multiple lenses"):
        generate('Multiple', multiple_lens, multiple_lens_seed, telescope, run)
    with st.spinner("Generating non lenses"):
        generate('Non lens', non_lens, non_lens_seed, telescope, run)
    st.success("Testing images generated")
    with st.spinner("Exporting images"):
        export.export(run)
    st.success("Images exported")
    with st.spinner("Testing images"):
        if torch.cuda.is_available():
            device = torch.device("cuda")  
        elif torch.backends.mps.is_available():
            device = torch.device("mps")       
        else:
            device = torch.device("cpu")

        X_prescaled, y = CNN.load_files(run)
        noise = CNN.load_noise()
        X = CNN.preprocessing(X_prescaled, noise)
        X_test = torch.tensor(X[:, None], dtype=torch.float32).to(device)
        cnn = CNN.build_model()
        load = torch.load("evaluation/Model_weights.pt", map_location=(device), weights_only=True,)
        cnn.load_state_dict(load)
        cnn.to(device)
        cnn.eval()

        with torch.no_grad():
            out = cnn(X_test)                          
        preds = out.cpu().numpy().argmax(axis=1)
    matrix_fig = CNN.matrix(X_test, y, cnn, 'Testing')
    st.pyplot(matrix_fig)
    
    


    

