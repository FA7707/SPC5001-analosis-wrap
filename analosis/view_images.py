import pandas as pd
import pickle 
import numpy as np
import os
import matplotlib as plt

path = ["results/datasets/Arc_lens_image_list.pickle", "results/datasets/Non_lens_image_list.pickle", "results/datasets/Ring_image_list.pickle", "results/datasets/Multi_lens_image_list.pickle"]

with open('results/datasets/Ring_image_list.pickle', 'rb') as f:
    images = pickle.load(f)
for i in images:
    i = i - i.min()
    i = i / i.max()
    
np.moveaxis(images, 0, -1)
np.arcsinh(images)

def view_images(run):
    path = [f"results/datasets/Arc_lens_{run}_image_list.pickle", f"results/datasets/Non_lens_{run}_image_list.pickle", f"results/datasets/Ring_image_{run}_list.pickle", f"results/datasets/Multi_lens_{run}image_list.pickle"]
    for i in path:
        with open(i, 'rb') as f:
        images = pickle.load(f)

        
    