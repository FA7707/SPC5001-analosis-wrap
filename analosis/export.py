import pandas as pd
import pickle 
import numpy as np
import os

csv = ["results/datasets/Arc_lens_input_kwargs.csv", "results/datasets/Non_lens_input_kwargs.csv", "results/datasets/Ring_input_kwargs.csv", "results/datasets/Multi_lens_input_kwargs.csv"]
path = ["results/datasets/Arc_lens_image_list.pickle", "results/datasets/Non_lens_image_list.pickle", "results/datasets/Ring_image_list.pickle", "results/datasets/Multi_lens_image_list.pickle"]

data = {}
stored_key = []
for i in csv:
    key = i.replace("results/datasets/", "").replace("_input_kwargs.csv", "")
    stored_key.append(key)
    with open(i) as f:
        data[key] = pd.read_csv(f)                         

images = {}
for i in path:
    key = i.replace("results/datasets/", "").replace("_image_list.pickle", "")
    with open (i,'rb') as f:
        images[key] = pickle.load(f)

def u(mass_bar, x_bar, virial_mass_nfw, x_nfw, y_nfw, y_bar, x_sl, y_sl, theta_E):
    x_cm = (mass_bar * x_bar + virial_mass_nfw * x_nfw) / (mass_bar + virial_mass_nfw)
    y_cm = (mass_bar * y_bar + virial_mass_nfw * y_nfw) / (mass_bar + virial_mass_nfw)
    distance = np.sqrt(np.square(x_sl - x_cm) + np.square(y_sl - y_cm))
    u = distance / theta_E

    return u
    
for i in stored_key:
    df = data[i][['mass_bar', 'x_bar', 'virial_mass_nfw', 'x_nfw', 'y_nfw', 'y_bar', 'x_sl', 'y_sl', 'theta_E']]
    u_result = u(*df.to_numpy().T)
    data[i]['u_value'] = u_result

def save_each(arrays, folder, prefix):
    os.makedirs(folder, exist_ok=True)          
    filenames = []                              

    for n, arr in enumerate(arrays, start=1):   
        name = f"{prefix}_{n:04d}.npy"           
        path = os.path.join(folder, name)        
        np.save(path, arr)                       
        filenames.append(name)                   
    return filenames

def filename():
    for i in stored_key:
        names = save_each(images[i], "results/images", i)
        data[i]['Filename'] = names
    return names
filename()

def class_ ():
    for i in stored_key:
        data[i]['Class'] = i
    return 
class_()

df1 = data['Ring'][['Filename', 'Class', 'u_value', 'theta_E']]
df2 = data['Arc_lens'][['Filename', 'Class', 'u_value', 'theta_E']]
df3 = data['Non_lens'][['Filename', 'Class', 'u_value', 'theta_E']]
df4 = data['Multi_lens'][['Filename', 'Class', 'u_value', 'theta_E']]
df3[['u_value','theta_E']] = np.nan
frames = [df1, df2, df3,df4]
result = pd.concat(frames, ignore_index=True)
os.makedirs("results/Cleaned_datasets", exist_ok=True)
result.to_csv("results/Cleaned_datasets/labels.csv", index=False)