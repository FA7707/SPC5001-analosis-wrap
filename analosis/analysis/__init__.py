"""
Compatibility fix for chainconsumer < 1.0 with current scipy.

chainconsumer < 1.0 imports two things that newer scipy versions removed:
- scipy.integrate.simps (renamed to simpson)
- scipy.ndimage.filters (merged into scipy.ndimage)

This runs before plots.py imports chainconsumer, and puts the old names back,
so nobody has to edit chainconsumer inside their venv.
"""
import sys
import scipy.integrate
import scipy.ndimage

if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

try:
    import scipy.ndimage.filters
except ImportError:
    sys.modules['scipy.ndimage.filters'] = scipy.ndimage