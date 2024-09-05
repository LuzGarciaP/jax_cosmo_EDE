# This module defines a few default cosmologies
from functools import partial
from jax_cosmo.core import Cosmology_EDE

# To add new cosmologies, we just set the parameters to some default values using
# partial

# Planck 2015 paper XII Table 4 final column (best fit)
Planck15 = partial(
    Cosmology_EDE,
    Omega_c=0.2589,
    Omega_b=0.04860,
    Omega_k=0.0,
    h=0.6774,
    n_s=0.9667,
    sigma8=0.8159,
    eme=10.0,
    w0=-1.0,
    flag_ede=1,
    a_st=0.005,
)
