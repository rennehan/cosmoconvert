from .const import XH, PROTON_MASS, KBOLTZ, GAMMA


def convert_abundance_to_number_fraction(abundance):
    return 10.0**(abundance - 12.0)


def convert_u_to_temp(u, ne, h_frac = 0):
    if h_frac == 0:
        h_frac = XH
        
    # If ne = 0 gas is unionized
    mu = 4.0 / (3.0 * h_frac + 4.0 * h_frac * ne + 1.0)
    return u * (GAMMA - 1) * mu * (PROTON_MASS / KBOLTZ)


# From Torrey et al. (2014).
# rho is expected in cgs
def hot_cgm_line(rho):
    rho_param = 405.0  # protons/cm**3
    temp_param = 1e6  # Kelvin
    exponent = 0.25

    # Convert to cgs
    rho_param *= PROTON_MASS / XH

    return temp_param * (rho / rho_param)**exponent
