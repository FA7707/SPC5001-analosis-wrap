import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../')
import pickle
import torch
import torch.nn as nn
from sklearn import metrics
import os
import CNN
from image_settings import generate

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

    path = ["results/datasets/Ring_eval_image_list.pickle", "results/datasets/Arc_lens_eval_image_list.pickle", "results/datasets/Multi_lens_eval_image_list.pickle", "results/datasets/Non_lens_eval_image_list.pickle"]
    
    images = {}
    for file_path in path:
        name = file_path.replace("results/datasets/", "").replace("_eval_image_list.pickle", "")
        with open(file_path, 'rb') as f:
            images[name] = pickle.load(f)
    
    all_images = []
    labels = []
    
    codes = {'Ring': 0, 'Arc_lens': 1, 'Multi_lens': 2, 'Non_lens': 3}
    for name, batch in images.items():
        all_images.extend(batch)
        labels.extend([codes[name]] * len(batch))

    

