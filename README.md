# SPC5001: Using Computer Vision to Classify Strong Gravitational Lensing

**Author:** Fahim Ahmed (Queen Mary University of London)
**Supervisors:** Natalie Hogg, Ed Macaulay

This repository contains my SPC5001 project. The aim is to compare a convolutional neural network (CNN) and a vision transformer (ViT) on two tasks, using simulated images of strong gravitational lenses:

1. **Classification:** predict whether an image is a non-lens, an Einstein ring, arcs, or multiple images. Evaluated with precision, recall and F1 score.
2. **Parameter estimation:** predict physical lens parameters (such as the Einstein radius, lens ellipticity and source offset) directly from the image. Evaluated with RMSE against the true values.

Training time and computational cost will also be compared between the two models.

---

## Where this code comes from

The image simulation is built on **analosis**, written by **Natalie B. Hogg and Pierre Fleury**:

- Original repository: https://github.com/nataliehogg/analosis
- Paper: https://arxiv.org/abs/2210.07210

analosis is a wrapper for [lenstronomy](https://github.com/lenstronomy/lenstronomy). It simulates large numbers of strong lensing images and fits them with an MCMC parameter inference, which the original authors used to measure line-of-sight shear. This project uses only the **image simulation** side. The MCMC fitting isn't part of my pipeline.

The original licence is kept in `LICENSE`.

---

## Changes made to analosis

| File | Change | Why |
|---|---|---|
| `analosis/image/source.py` | Removed the cap that limited the source offset to the source's half-light radius | Without it, the source could never be placed far enough from the lens to produce arcs or multiple images |
| `analosis/image/source.py` | Source offset is now drawn uniformly in radius instead of uniformly in area | Gives an even spread of `u` values, so each lens type appears in similar numbers |
| `analosis/image/image_generator.py` | Added a `'source'` option to `image_settings` (`False` = no background galaxy) | Generates the non-lens class: the lens galaxy alone, with no lensed source |
| `analosis/image/image_generator.py` | Lens light is only passed in when `lens_light` is `True` | Setting `lens_light: False` previously crashed |
| `analosis/analysis/plots.py` | Replaced `matplotlib.cm.get_cmap` with `matplotlib.colormaps[...]`, and turned off `usetex` | Compatibility with current matplotlib, and no LaTeX install needed |

**Key quantity:** `u` = source offset ÷ Einstein radius. Roughly:
- `u` below ~0.15: Einstein ring
- ~0.2 to 0.4: arcs
- ~0.45 to 1: multiple images
- above ~1.2: no strong lensing

These boundaries are approximate and also depend on lens ellipticity and source size.

---

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Version notes:**
- **lenstronomy is pinned to 1.14.0.** Version 1.14.1 renamed `SimAPI`'s `numpix` argument to `num_pix`, which breaks analosis.
- **chainconsumer < 1.0 needs a patch** to work with current scipy. It's only needed for the MCMC plots, but it's imported on load. Run this once, after installing:

  ```powershell
  $f = ".venv\Lib\site-packages\chainconsumer\analysis.py"
  (Get-Content $f) -replace 'from scipy.integrate import simps', 'from scipy.integrate import simpson as simps' -replace 'from scipy.ndimage.filters import', 'from scipy.ndimage import' | Set-Content $f
  ```

After editing any `.py` file in `analosis`, **restart the notebook kernel**, because Python caches imported modules.

---

## Notebooks

- `analosis/notebooks/Experiment.ipynb`: exploring how `image_settings` affect the generated lenses
- `analosis/notebooks/Full_Einstein_ring.ipynb`: generating Einstein ring examples
- `analosis/notebooks/Non_lens.ipynb`: generating non-lens examples (`'source': False`)
- The other notebooks are from the original analosis repository.

---

## Status

- [x] Simulation pipeline running locally
- [x] Control over lens type via source offset (`u`)
- [x] Non-lens class generation
- [ ] Class labelling for rings / arcs / multiple images
- [ ] Balanced dataset generation and export
- [ ] CNN and ViT training and evaluation