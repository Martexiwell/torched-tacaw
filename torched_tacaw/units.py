import scipy.constants as c
from scipy import constants as c

THz2meV = c.tera * c.h / (c.eV * c.milli)
meV2THz = 1 / THz2meV

def convert_THz2meV(frequency):
    energy = frequency * THz2meV
    return energy

def convert_meV2THz(energy):
    frequency = energy * meV2THz
    return frequency