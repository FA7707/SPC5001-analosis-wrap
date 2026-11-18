import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../')
from analosis.run import Run
import random
import pickle
import torch
import torch.nn as nn
from sklearn import metrics
import os

cosmology = {'id': 'planck18', 'H0': 67.4, 'Om': 0.315}
st.title("Testing model")                                  

ring = st.number_input("Number of ring images", min_value=20, max_value=10000, value=20)
ring_seed= st.number_input("Ring seed", value=5)

arc_lens= st.number_input("Number of arc images", min_value=20, max_value=10000, value=20)
arc_lens_seed= st.number_input("Arc seed", value=2)

multiple_lens = st.number_input("Number of multiple lens images", min_value=20, max_value=10000, value=20)
multiple_lens_seed = st.number_input("Multiple lens seed", value=3)

non_lens= st.number_input("Number of non lens images", min_value=20, max_value=10000, value=20)
non_lens_seed= st.number_input("Non lens seed", value=4)

def generate (lens, number_of_images, seed):
    np.random.seed(seed)
    random.seed(seed)

    if lens == 'Ring':
       image_settings = {'generate_image': True,
                  'number_of_images': number_of_images,
                  'image_name': 'Ring_eval',
                  'lens_light': True,
                  'source': True, # False = no background galaxy (non-lens images)
                  'truth_model': 'boxydisky',
                  'a4_max': 0.1,
                  'telescope': 'HST',
                  'band': None,
                  'maximum_shear': 0.025, # any larger means gamma_LOS > 0.05 which is challenging for the MCMC
                  'Einstein_radius_min': 0.5, # arcsec
                  'min_aspect_ratio_source': 0.6,
                  'min_aspect_ratio_baryons': 0.6,
                  'min_aspect_ratio_nfw': 0.6,
                  'minimum_source_offset_factor': 0.0, # in units of Einstein radius
                  'maximum_source_offset_factor': 0.15, # in units of Einstein radius, values from 0 - 0.15 give full/clear rings, values 0.2 - 0.35 give half a ring or a long arc, 0.4 - 0.5 give short arcs and a small counter image (doubles). 
                  'max_source_perturbations': 5, # 3 additional Sersics with amplitudes that are randomly 0-5% of the main source
                  'fixed_numpix': True,
                  'sigma_halo_offset': 0} # pc
    elif lens == 'Arc':
        image_settings = {'generate_image': True,
                  'number_of_images': number_of_images,
                  'image_name': 'Arc_lens_eval',
                  'lens_light': True,
                  'source': True, # False = no background galaxy (non-lens images)
                  'truth_model': 'boxydisky',
                  'a4_max': 0.1,
                  'telescope': 'HST',
                  'band': None,
                  'maximum_shear': 0.025, # any larger means gamma_LOS > 0.05 which is challenging for the MCMC
                  'Einstein_radius_min': 0.5, # arcsec
                  'min_aspect_ratio_source': 0.6,
                  'min_aspect_ratio_baryons': 0.6,
                  'min_aspect_ratio_nfw': 0.6,
                  'minimum_source_offset_factor': 0.25, # in units of Einstein radius
                  'maximum_source_offset_factor': 0.35, # in units of Einstein radius, values from 0 - 0.15 give full/clear rings, values 0.2 - 0.35 give half a ring or a long arc, 0.4 - 0.5 give short arcs and a small counter image (doubles). 
                  'max_source_perturbations': 5, # 3 additional Sersics with amplitudes that are randomly 0-5% of the main source
                  'fixed_numpix': True,
                  'sigma_halo_offset': 0} # pc
    elif lens == 'Multiple':
        image_settings = {'generate_image': True,
                  'number_of_images': number_of_images,
                  'image_name': 'Multi_lens_eval',
                  'lens_light': True,
                  'source': True, # False = no background galaxy (non-lens images)
                  'truth_model': 'boxydisky',
                  'a4_max': 0.1,
                  'telescope': 'HST',
                  'band': None,
                  'maximum_shear': 0.025, # any larger means gamma_LOS > 0.05 which is challenging for the MCMC
                  'Einstein_radius_min': 0.5, # arcsec
                  'min_aspect_ratio_source': 0.6,
                  'min_aspect_ratio_baryons': 0.6,
                  'min_aspect_ratio_nfw': 0.6,
                  'minimum_source_offset_factor': 0.45, # in units of Einstein radius
                  'maximum_source_offset_factor': 0.7, # in units of Einstein radius, values from 0 - 0.15 give full/clear rings, values 0.2 - 0.35 give half a ring or a long arc, 0.4 - 0.5 give short arcs and a small counter image (doubles). 
                  'max_source_perturbations': 5, # 3 additional Sersics with amplitudes that are randomly 0-5% of the main source
                  'fixed_numpix': True,
                  'sigma_halo_offset': 0} # pc
    elif lens == 'Non lens':
        image_settings = {'generate_image': True,
                  'number_of_images': number_of_images,
                  'image_name': 'Non_lens_eval',
                  'lens_light': True,
                  'source': False, # False = no background galaxy (non-lens images)
                  'truth_model': 'boxydisky',
                  'a4_max': 0.1,
                  'telescope': 'HST',
                  'band': None,
                  'maximum_shear': 0.025, # any larger means gamma_LOS > 0.05 which is challenging for the MCMC
                  'Einstein_radius_min': 0.5, # arcsec
                  'min_aspect_ratio_source': 0.6,
                  'min_aspect_ratio_baryons': 0.6,
                  'min_aspect_ratio_nfw': 0.6,
                  'minimum_source_offset_factor': 0.0, # in units of Einstein radius
                  'maximum_source_offset_factor': 0.15, # in units of Einstein radius, values from 0 - 0.15 give full/clear rings, values 0.2 - 0.35 give half a ring or a long arc, 0.4 - 0.5 give short arcs and a small counter image (doubles). 
                  'max_source_perturbations': 5, # 3 additional Sersics with amplitudes that are randomly 0-5% of the main source
                  'fixed_numpix': True,
                  'sigma_halo_offset': 0} # pc
    
    mcmc_settings = {'MCMC': False, 
                     'complexity': 'composite',
                     'number_of_runs': 3, 
                     'starting_index': 0, 
                     'sampler': 'EMCEE', 
                     'job_name': 'my_job_boxydisky', 
                     'n_burn': 500, 
                     'n_run': 500} 

    result = Run(cosmology, image_settings, mcmc_settings)
    
def matrix():
    path = "evaluation"
    os.makedirs(path, exist_ok=True)
    actual = y
    predicted = preds
    confusion_matrix = metrics.confusion_matrix(actual, predicted)

    df_matrix_values = pd.DataFrame()
    df_matrix_values['Actual'] = actual
    df_matrix_values['Predicted'] = predicted
    df_matrix_values.to_csv(os.path.join(path,'Matrix_values_test.csv'), index=False)
    
    fig, ax = plt.subplots(figsize=(6, 5)) 
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Ring', 'Arc_lens', 'Multi_lens', 'Non_lens'])
    cm_display.plot(ax=ax)
    ax.set_title("CNN confusion matrix (test set)", fontsize=14, pad=12)
    ax.set_xlabel("Predicted label", labelpad=10)
    ax.set_ylabel("True label", labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(path,'CNN Confusion Matrix test.png'), dpi=300)
    st.pyplot(fig)

if st.button("Test"):
    with st.spinner("Generating Einstein rings"):
        generate ('Ring', ring, ring_seed)
    with st.spinner("Generating Arc lenses"):
        generate ('Arc', arc_lens, arc_lens_seed)
    with st.spinner("Generating Multiple lenses"):
        generate ('Multiple', multiple_lens, multiple_lens_seed)
    with st.spinner("Generating non lenses"):
        generate ('Non lens', non_lens, non_lens_seed)
    st.success("Test images generated")

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
    
    y = np.array(labels)
    X_prescaled = np.stack(all_images)
    df = pd.read_csv("evaluation/noise.csv")
    noise = df['Noise'][0]
    scaled = X_prescaled / noise
    stretched = np.arcsinh(scaled)
    m = stretched.mean(axis=(1, 2), keepdims=True)
    s = stretched.std(axis=(1, 2), keepdims=True)
    normed = (stretched - m) / s
    X = normed
    
    X_test = torch.tensor(X[:, None], dtype=torch.float32)
    
    block1 = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),  
        nn.ReLU(),
        nn.MaxPool2d(2))
    
    block2 = nn.Sequential(
        nn.Conv2d(8, 16, kernel_size=3, padding=1),  
        nn.ReLU(),
        nn.MaxPool2d(2))
    
    block3 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=3, padding=1),  
        nn.ReLU(),
        nn.MaxPool2d(2))
    
    cnn = nn.Sequential(block1, block2, block3, nn.AdaptiveMaxPool2d(3), nn.Flatten(), nn.Linear(288,32), nn.ReLU(), nn.Linear(32,4))
    
    if torch.cuda.is_available():
        device = torch.device("cuda")      
    elif torch.backends.mps.is_available():
        device = torch.device("mps")       
    else:
        device = torch.device("cpu")
    
    load = torch.load("evaluation/Model_weights.pt", map_location=(device), weights_only=True,)
    cnn.load_state_dict(load)
    cnn.to(device)
    cnn.eval()
    
    data = (X_test)                               
    with torch.no_grad():
        out = cnn(data.to(device))                          
    preds = out.cpu().numpy().argmax(axis=1)
    matrix()
    
    class_ = ['Ring', 'Arc_lens', 'Multi_lens', 'Non_lens']

    accuracy = metrics.accuracy_score(y, preds)
    precision = metrics.precision_score(y, preds, average=None)
    recall = metrics.recall_score(y, preds, average=None)
    f1 = metrics.f1_score(y, preds, average=None)

    st.metric("Accuracy achieved:",f"{accuracy:.1%}")

    for i, team in enumerate(class_):  
        st.subheader(team)
        col1, col2, col3 = st.columns(3)            
        col1.metric("Precision achieved:",f"{precision[i]:.1%}")
        col2.metric("Recall achieved:",f"{recall[i]:.1%}")
        col3.metric("F1 score achieved:",f"{f1[i]:.1%}")

