import sys
sys.path.append('../..')
from analosis.run import Run
import random
import numpy as np

np.random.seed(165)
random.seed(165)

cosmology = {'id': 'planck18', 'H0': 67.4, 'Om': 0.315}

image_settings = {'generate_image': True,
                  'number_of_images': 100,
                  'image_name': 'Non_lens',
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

path = result.pathfinder()

from analosis.analysis.plots import Plots
p = Plots(cosmology, path)

p.image_plot(path, image_settings, number_of_columns=8, u_max=10, save=False, show=False)

