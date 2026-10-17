"""
physics.py — shared physics + simulation for binary black-hole orbital data.

Ported verbatim from `05 POST MINKOWSKIAN EXPANSION/Functions_All.ipynb`
(the consolidated source of truth for the bachelor project).

Cosmetic changes only (no physics changes):
  * `γ` renamed to `gamma` (ASCII-safe function name).
  * terminal event functions renamed from Danish to English:
      `dyk`        -> `plunge`
      `sprædning`  -> `scatter`
  * added `save_orbits` / `load_orbits` (numpy `.npz` + JSON sidecar) and a
    small shared `plot_orbit` helper used by both the preview and the SVG export.

Models (the `mode` argument of `run_model`):
  N       Newtonian
  N_rel   relativistic-kinetic Newtonian gravity
  EIH     Einstein-Infeld-Hoffmann (1PN EIH)
  1PN     1st order post-Newtonian
  1PM     1st order post-Minkowskian
  2PM     2nd order post-Minkowskian

Units: with `DL=True` (the default, and what all the pretty plots use) the
system is in geometrized units c = G = 1, so distances are in G*M/c^2 and
times in G*M/c^3.
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u
from scipy.integrate import solve_ivp

c = const.c
G = const.G
AU = (1 * u.au).to('m')
M_sun = const.M_sun
M_earth = const.M_earth
palette = np.array(['#420a68', '#932667', '#dd513a', '#fca50a'])


# --------------------------------------------------------------------------- #
# Basic utility functions
# --------------------------------------------------------------------------- #
def rel_len(x_1, x_2):  # relative distance between two vectors
    r_vec = x_1 - x_2
    r = np.linalg.norm(r_vec)
    return r


def findrCM(x_1, x_2, m_1, m_2):  # center of mass given two masses + positions
    M = m_1 + m_2
    r_cm = (m_1 * x_1 + m_2 * x_2) / M
    return r_cm


def gamma(v, DL=True):  # Lorentz factor; DL=True -> c=G=1
    c = const.c
    G = const.G
    if DL == True:
        c, G = 1, 1
    return 1 / np.sqrt(1 - np.dot(v, v) / c**2)


# --------------------------------------------------------------------------- #
# Acceleration equations (each returns the COM-frame state derivative)
# --------------------------------------------------------------------------- #
def accelerationN(t, s_cm, m_1, m_2, r_max, R_S=1, DL=True):
    c = const.c
    G = const.G
    if DL == True:
        c, G = 1, 1

    r_vec_x, r_vec_y, v_vec_x, v_vec_y = s_cm
    r_vec = np.array([r_vec_x, r_vec_y])
    v_vec = np.array([v_vec_x, v_vec_y])

    μ = m_1 * m_2 / (m_1 + m_2)
    v_1 = μ / m_1 * v_vec

    p_vec = m_1 * v_1

    r = np.linalg.norm(r_vec)
    r_vec_dot = (1 / m_1 + 1 / m_2) * p_vec

    a = -G * (m_1 + m_2) * r_vec / r**3
    a_x, a_y = a[0], a[1]
    return v_vec_x, v_vec_y, a_x, a_y


def accelerationN_rel(t, s_cm, m_1, m_2, r_max, R_S=1, DL=True):
    c = const.c
    G = const.G
    if DL == True:
        c, G = 1, 1

    r_vec_x, r_vec_y, v_vec_x, v_vec_y = s_cm
    r_vec = np.array([r_vec_x, r_vec_y])
    v_vec = np.array([v_vec_x, v_vec_y])

    r = np.linalg.norm(r_vec)

    μ = m_1 * m_2 / (m_1 + m_2)
    v_1 = μ / m_1 * v_vec

    p_vec = gamma(v_1, DL) * m_1 * v_1
    p = np.linalg.norm(p_vec)
    p_vec_dot = -G * m_1 * m_2 * r_vec / r**3

    E_1 = np.sqrt((m_1 * c**2)**2 + (p * c)**2)
    E_2 = np.sqrt((m_2 * c**2)**2 + (p * c)**2)

    dE_1dp = p_vec * c**2 / E_1
    dE_2dp = p_vec * c**2 / E_2
    E_1_dot = np.dot(dE_1dp, p_vec_dot)
    E_2_dot = np.dot(dE_2dp, p_vec_dot)
    dE_1dp_dot = p_vec_dot * c**2 / E_1 - p_vec * c**2 / E_1**2 * E_1_dot
    dE_2dp_dot = p_vec_dot * c**2 / E_2 - p_vec * c**2 / E_2**2 * E_2_dot

    a = dE_1dp_dot + dE_2dp_dot
    a_x, a_y = a[0], a[1]
    return v_vec_x, v_vec_y, a_x, a_y


def accelerationEIH(t, s_cm, m_1, m_2, r_max, R_S=1, DL=True):
    c = const.c
    G = const.G
    if DL == True:
        c, G = 1, 1

    r_vec_x, r_vec_y, v_vec_x, v_vec_y = s_cm
    r_vec = np.array([r_vec_x, r_vec_y])
    v_vec = np.array([v_vec_x, v_vec_y])

    μ = m_1 * m_2 / (m_1 + m_2)
    v_1 = μ / m_1 * v_vec

    p_vec = m_1 * v_1

    r = np.linalg.norm(r_vec)
    p = np.linalg.norm(p_vec)

    r_vec_dot = (1 / m_1 + 1 / m_2) * p_vec - 1 / 2 * (1 / m_1**3 + 1 / m_2**3) / c**2 * p**2 * p_vec - G / r * ((1 + 3 * (m_1 + m_2)**2 / (m_1 * m_2)) * p_vec / c**2 + np.dot(p_vec, r_vec) / (r * c)**2 * r_vec)
    r_dot = np.dot(r_vec, r_vec_dot) / r

    p_vec_dot = G**2 * m_1 * m_2 * (m_1 + m_2) / (c**2 * r**4) * r_vec - G / r**3 * ((m_1 * m_2 + (1 / 2 + 3 / 2 * (m_1 + m_2)**2 / (m_1 * m_2)) * p**2 / c**2 + 3 / 2 * np.dot(p_vec, r_vec)**2 / (c * r)**2) * r_vec - np.dot(p_vec, r_vec) / c**2 * p_vec)
    a = (1 / m_1 + 1 / m_2) * p_vec_dot - 3 / 2 * (1 / m_1**3 + 1 / m_2**3) / c**2 * p**2 * p_vec_dot - G / r * ( -r_dot / r * ((1 + 3 * (m_1 + m_2)**2 / (m_1 * m_2)) / c**2 * p_vec + np.dot(p_vec, r_vec) / (c * r)**2 * r_vec) + (1 + 3 * (m_1 + m_2)**2 / (m_1 * m_2)) / c**2 * p_vec_dot + ((np.dot(p_vec_dot, r_vec) + np.dot(p_vec, r_vec_dot)) * r_vec + np.dot(p_vec, r_vec) * r_vec_dot) / (c * r)**2 - 2 * np.dot(p_vec, r_vec) / (c**2 * r**3) * r_vec * r_dot)
    a_x, a_y = a[0], a[1]
    return v_vec_x, v_vec_y, a_x, a_y


def accelerationEIH_trunc(t, s_cm, m_1, m_2, r_max, R_S=1, DL=True):
    c = const.c
    G = const.G
    if DL == True:
        c, G = 1, 1

    r_vec_x, r_vec_y, v_vec_x, v_vec_y = s_cm
    r_vec = np.array([r_vec_x, r_vec_y])
    v_vec = np.array([v_vec_x, v_vec_y])

    μ = m_1 * m_2 / (m_1 + m_2)
    v_1 = μ / m_1 * v_vec

    p_vec = m_1 * v_1

    r = np.linalg.norm(r_vec)
    p = np.linalg.norm(p_vec)

    r_vec_dot = (1 / m_1 + 1 / m_2) * p_vec - 1 / 2 * (1 / m_1**3 + 1 / m_2**3) / c**2 * p**2 * p_vec - G / r * ((1 + 3 * (m_1 + m_2)**2 / (m_1 * m_2)) * p_vec / c**2 + np.dot(p_vec, r_vec) / (r * c)**2 * r_vec)
    r_dot = np.dot(r_vec, r_vec_dot) / r

    a = -G / r**3 * (m_1 + m_2) * r_vec + 1 / c**2 * (G**2 * (m_1 + m_2)**2 / r**4 * r_vec - G / r**3 * ((6 * (m_1 + m_2) / (m_1 * m_2) * p**2 + np.dot(p_vec, r_vec)**2 / r**2) * r_vec - np.dot(p_vec, r_vec) * p_vec) + G**2 / r**4 * (2 * m_1 * m_2 + 3 * (m_1 + m_2)**2) * r_vec + 3 * G / r**3 * (m_1 + m_2) / (m_1 * m_2) * np.dot(p_vec, r_vec) * ((m_1 + m_2)**2 / (m_1 * m_2) * p_vec + np.dot(p_vec, r_vec) * r_vec))
    a_x, a_y = a[0], a[1]
    return v_vec_x, v_vec_y, a_x, a_y


def acceleration1PN(t, s_cm, m_1, m_2, r_max, R_S=1, DL=True):
    c = const.c
    G = const.G
    if DL == True:
        c, G = 1, 1

    r_vec_x, r_vec_y, v_vec_x, v_vec_y = s_cm
    r_vec = np.array([r_vec_x, r_vec_y])
    v_vec = np.array([v_vec_x, v_vec_y])

    μ = m_1 * m_2 / (m_1 + m_2)
    v_1 = μ / m_1 * v_vec

    p_vec = m_1 * v_1
    p = np.linalg.norm(p_vec)

    r = np.linalg.norm(r_vec)

    a = -G * (m_1 + m_2) / r**3 * r_vec + 1 / c**2 * ((10 * G**2 * m_1 * m_2 / r**3 + 4 * G**2 * (m_1**2 + m_2**2) / r**3 + G / r**2 * (3 / (2 * r**2) * np.dot(r_vec, p_vec)**2 * μ - (m_2 / m_1**2 + m_1 / m_2**2) * p**2 - 6 * μ * p**2)) * r_vec / r + G / r**3 * 4 * (m_2 / m_1 + m_1 / m_2 + 3 / 2) * np.dot(r_vec, p_vec) * p_vec / μ)
    a_x, a_y = a[0], a[1]
    return v_vec_x, v_vec_y, a_x, a_y


def acceleration1PM(t, s_cm, m_1, m_2, r_max, R_S=1, DL=True):
    c = const.c
    G = const.G
    if DL == True:
        c, G = 1, 1

    r_vec_x, r_vec_y, v_vec_x, v_vec_y = s_cm
    r_vec = np.array([r_vec_x, r_vec_y])
    v_vec = np.array([v_vec_x, v_vec_y])

    μ = m_1 * m_2 / (m_1 + m_2)
    v_1 = μ / m_1 * v_vec

    p_vec = m_1 * v_1 * gamma(v_1, DL)

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    p = np.linalg.norm(p_vec)

    E_1 = np.sqrt((m_1 * c**2)**2 + (p * c)**2)
    E_2 = np.sqrt((m_2 * c**2)**2 + (p * c)**2)
    c_1 = (m_1 * m_2 * c**2)**2 - 2 * (E_1 * E_2 / c**2 + p**2)**2

    p_vec_dot = G * c_1 / (E_1 * E_2 * r**3) * r_vec

    dE_1dp = p_vec * c**2 / E_1
    dE_2dp = p_vec * c**2 / E_2
    E_1_dot = np.dot(dE_1dp, p_vec_dot)
    E_2_dot = np.dot(dE_2dp, p_vec_dot)
    dE_1dp_dot = p_vec_dot * c**2 / E_1 - p_vec * c**2 / E_1**2 * E_1_dot
    dE_2dp_dot = p_vec_dot * c**2 / E_2 - p_vec * c**2 / E_2**2 * E_2_dot

    dc_1dp = -4 * (E_1 * E_2 / c**2 + p**2) * ((dE_1dp * E_2 + E_1 * dE_2dp) / c**2 + 2 * p_vec)
    c_1_dot = -4 * (E_1 * E_2 / c**2 + p**2) * ((E_1_dot * E_2 + E_1 * E_2_dot) / c**2 + 2 * np.dot(p_vec, p_vec_dot))
    dc_1dp_dot = -4 * (((E_1_dot * E_2 + E_1 * E_2_dot) / c**2 + 2 * np.dot(p_vec, p_vec_dot)) * ((dE_1dp * E_2 + E_1 * dE_2dp) / c**2 + 2 * p_vec) + (E_1 * E_2 / c**2 + p**2) * ((dE_1dp_dot * E_2 + dE_1dp * E_2_dot + E_1_dot * dE_2dp + E_1 * dE_2dp_dot) / c**2 + 2 * p_vec_dot))

    r_vec_dot = dE_1dp + dE_2dp + G / (E_1**2 * E_2**2 * r) * (E_1 * E_2 * dc_1dp - c_1 * (dE_1dp * E_2 + E_1 * dE_2dp))
    r_dot = np.dot(r_vec, r_vec_dot) / r
    a = dE_1dp_dot + dE_2dp_dot + G / (E_1**2 * E_2**2 * r) * (-(r_dot / r + 2 * E_1_dot / E_1 + 2 * E_2_dot / E_2) * (E_1 * E_2 * dc_1dp - c_1 * (dE_1dp * E_2 + E_1 * dE_2dp)) + (E_1_dot * E_2 + E_1 * E_2_dot) * dc_1dp + E_1 * E_2 * dc_1dp_dot - c_1_dot * (dE_1dp * E_2 + E_1 * dE_2dp) - c_1 * (dE_1dp_dot * E_2 + dE_1dp * E_2_dot + E_1_dot * dE_2dp + E_1 * dE_2dp_dot))

    a_x, a_y = a[0], a[1]
    return v_vec_x, v_vec_y, a_x, a_y


def acceleration1PM_trunc(t, s_cm, m_1, m_2, r_max, R_S=1, DL=True):
    c = const.c
    G = const.G
    if DL == True:
        c, G = 1, 1

    r_vec_x, r_vec_y, v_vec_x, v_vec_y = s_cm
    r_vec = np.array([r_vec_x, r_vec_y])
    v_vec = np.array([v_vec_x, v_vec_y])

    μ = m_1 * m_2 / (m_1 + m_2)
    v_1 = μ / m_1 * v_vec

    p_vec = m_1 * v_1 * gamma(v_1, DL)

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    p = np.linalg.norm(p_vec)

    E_1 = np.sqrt((m_1 * c**2)**2 + (p * c)**2)
    E_2 = np.sqrt((m_2 * c**2)**2 + (p * c)**2)
    c_1 = (m_1 * m_2 * c**2)**2 - 2 * (E_1 * E_2 / c**2 + p**2)**2

    p_vec_dot = G * c_1 / (E_1 * E_2 * r**3) * r_vec

    dE_1dp = p_vec * c**2 / E_1
    dE_2dp = p_vec * c**2 / E_2
    E_1_dot = np.dot(dE_1dp, p_vec_dot)
    E_2_dot = np.dot(dE_2dp, p_vec_dot)
    dE_1dp_dot = p_vec_dot * c**2 / E_1 - p_vec * c**2 / E_1**2 * E_1_dot
    dE_2dp_dot = p_vec_dot * c**2 / E_2 - p_vec * c**2 / E_2**2 * E_2_dot

    dc_1dp = -4 * (E_1 * E_2 / c**2 + p**2) * ((dE_1dp * E_2 + E_1 * dE_2dp) / c**2 + 2 * p_vec)
    c_1_dot = -4 * (E_1 * E_2 / c**2 + p**2) * ((E_1_dot * E_2 + E_1 * E_2_dot) / c**2 + 2 * np.dot(p_vec, p_vec_dot))
    dc_1dp_dot = -4 * (((E_1_dot * E_2 + E_1 * E_2_dot) / c**2 + 2 * np.dot(p_vec, p_vec_dot)) * ((dE_1dp * E_2 + E_1 * dE_2dp) / c**2 + 2 * p_vec) + (E_1 * E_2 / c**2 + p**2) * ((dE_1dp_dot * E_2 + dE_1dp * E_2_dot + E_1_dot * dE_2dp + E_1 * dE_2dp_dot) / c**2 + 2 * p_vec_dot))

    r_vec_dot = dE_1dp + dE_2dp
    r_dot = np.dot(r_vec, r_vec_dot) / r
    a = dE_1dp_dot + dE_2dp_dot + G / (E_1**2 * E_2**2 * r) * (-(r_dot / r) * (E_1 * E_2 * dc_1dp - c_1 * (dE_1dp * E_2 + E_1 * dE_2dp)))

    a_x, a_y = a[0], a[1]
    return v_vec_x, v_vec_y, a_x, a_y


def acceleration2PM(t, s_cm, m_1, m_2, r_max, R_S=1, DL=True):
    c = const.c
    G = const.G
    if DL == True:
        c, G = 1, 1

    r_vec_x, r_vec_y, v_vec_x, v_vec_y = s_cm
    r_vec = np.array([r_vec_x, r_vec_y])
    v_vec = np.array([v_vec_x, v_vec_y])

    μ = m_1*m_2/(m_1 + m_2)
    v_1 = μ/m_1*v_vec

    p_vec = m_1*v_1*gamma(v_1, DL)

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    p = np.linalg.norm(p_vec)

    E_1 = np.sqrt((m_1*c**2)**2 + (p*c)**2)
    E_2 = np.sqrt((m_2*c**2)**2 + (p*c)**2)
    c_1 = (m_1*m_2*c**2)**2 - 2*(E_1*E_2/c**2 + p**2)**2
    c_r = 3*m_1**2*((m_1*m_2)**2*c**4 - 5*(E_1*E_2/c**2 + p**2)**2)
    c_l = (m_2/m_1)**2*c_r

    p_vec_dot = G*c_1/(E_1*E_2*r**3)*r_vec + G**2/r**4*2/(E_1*E_2)*r_vec*( 1/4*(c_r/m_1 + c_l/m_2)/c**2 + c_1*(E_1 + E_2)/(E_1*E_2) * (c_1/2*(1/(E_1 + E_2)**2 - 1/(E_1*E_2)) - 4*(E_1*E_2/c**2 + p**2)/c**2)  )
    p_dot = np.linalg.norm(p_vec_dot)

    dE_1dp = p_vec*c**2/E_1
    dE_2dp = p_vec*c**2/E_2
    E_1_dot = np.dot(dE_1dp, p_vec_dot)
    E_2_dot = np.dot(dE_2dp, p_vec_dot)
    dE_1dp_dot = p_vec_dot*c**2/E_1 - p_vec*c**2/E_1**2*E_1_dot
    dE_2dp_dot = p_vec_dot*c**2/E_2 - p_vec*c**2/E_2**2*E_2_dot

    green = E_1_dot*E_2 + E_1*E_2_dot
    purple = dE_1dp*E_2 + E_1*dE_2dp
    yellow = dE_1dp_dot*E_2 + dE_1dp*E_2_dot + E_1_dot*dE_2dp + E_1*dE_2dp_dot

    dc_1dp = -4*(E_1*E_2/c**2 + p**2)*((dE_1dp*E_2 + E_1*dE_2dp)/c**2 + 2*p_vec)
    c_1_dot = -4*(E_1*E_2/c**2 + p**2)*((E_1_dot*E_2 + E_1*E_2_dot)/c**2 + 2*np.dot(p_vec, p_vec_dot))
    dc_1dp_dot = -4*( ((E_1_dot*E_2 + E_1*E_2_dot)/c**2 + 2*np.dot(p_vec,p_vec_dot))*((dE_1dp*E_2 + E_1*dE_2dp)/c**2 + 2*p_vec) + (E_1*E_2/c**2 + p**2)*((dE_1dp_dot*E_2 + dE_1dp*E_2_dot + E_1_dot*dE_2dp + E_1*dE_2dp_dot)/c**2 + 2*p_vec_dot) )
    dc_rdp = -30*m_1**2*(E_1*E_2/c**2 + p**2)*(purple/c**2 + 2*p_vec)
    c_r_dot = -30*m_1**2*(E_1*E_2/c**2 + p**2)*(green/c**2 + 2*np.dot(p_vec, p_vec_dot))
    dc_rdp_dot = -30*m_1**2*( (green/c**2 + 2*np.dot(p_vec, p_vec_dot))*(purple/c**2 + 2*p_vec) + (E_1*E_2/c**2 + p**2)*(yellow/c**2 + 2*p_vec_dot) )

    r_vec_dot = dE_1dp + dE_2dp + G**2/(E_1*E_2*r**2)*( -purple/(E_1*E_2)*(1/4*(c_r/m_1 + c_l/m_2)/c**2 + c_1*(E_1 + E_2)/(E_1*E_2)*(c_1/2*(1/(E_1 + E_2)**2 - 1/(E_1*E_2)) - 4*(E_1*E_2/c**2 + p**2)/c**2 ) )  +  1/4*(1/m_1 + m_2/m_1**2)*1/c**2*dc_rdp + ( (E_1 + E_2)/(E_1*E_2)*dc_1dp + c_1/(E_1*E_2)*(dE_1dp + dE_2dp - (E_1 + E_2)/(E_1*E_2)*purple ) ) * (c_1/2*(1/(E_1 + E_2)**2 - 1/(E_1*E_2)) - 4*(E_1*E_2/c**2 + p**2)/c**2)  +  c_1*(E_1 + E_2)/(E_1*E_2) * ( 1/2*dc_1dp*( 1/(E_1 + E_2)**2 - 1/(E_1*E_2) ) + c_1/2*(-2*(dE_1dp + dE_2dp)/(E_1 + E_2)**3 + 1/(E_1*E_2)**2*purple) - 4/c**2*(purple/c**2 + 2*p_vec) ) )
    r_dot = np.dot(r_vec, r_vec_dot)/r

    a_kin = dE_1dp_dot + dE_2dp_dot
    a_1PM = G/(E_1**2*E_2**2*r)*(  -( r_dot/r + 2*E_1_dot/E_1 + 2*E_2_dot/E_2 )*( E_1*E_2*dc_1dp - c_1*(dE_1dp*E_2 + E_1*dE_2dp) ) + (E_1_dot*E_2 + E_1*E_2_dot)*dc_1dp + E_1*E_2*dc_1dp_dot - c_1_dot*(dE_1dp*E_2 + E_1*dE_2dp) - c_1*(dE_1dp_dot*E_2 + dE_1dp*E_2_dot + E_1_dot*dE_2dp + E_1*dE_2dp_dot) )
    a_2PM = - G**2/(r**2*E_1*E_2)*(2*r_dot/r + green/E_1*E_2)  *  ( -purple/(E_1*E_2)*(1/4*(c_r/m_1 + c_l/m_2)/c**2 + c_1*(E_1 + E_2)/(E_1*E_2)*(c_1/2*(1/(E_1 + E_2)**2 - 1/(E_1*E_2)) - 4*(E_1*E_2/c**2 + p**2)/c**2 ) )  +  1/4*(1/m_1 + m_2/m_1**2)*1/c**2*dc_rdp + ( (E_1 + E_2)/(E_1*E_2)*dc_1dp + c_1/(E_1*E_2)*(dE_1dp + dE_2dp - (E_1 + E_2)/(E_1*E_2)*purple ) ) * (c_1/2*(1/(E_1 + E_2)**2 - 1/(E_1*E_2)) - 4*(E_1*E_2/c**2 + p**2)/c**2)  +  c_1*(E_1 + E_2)/(E_1*E_2) * ( 1/2*dc_1dp*( 1/(E_1 + E_2)**2 - 1/(E_1*E_2) ) + c_1/2*(-2*(dE_1dp + dE_2dp)/(E_1 + E_2)**3 + 1/(E_1*E_2)**2*purple) - 4/c**2*(purple/c**2 + 2*p_vec) ) )  +  G**2/(r**2*E_1*E_2) * ( -( yellow/(E_1*E_2) - purple/(E_1*E_2)**2*green ) * ( 1/4*(c_r/m_1 + c_l/m_2)/c**2 + c_1*(E_1 + E_2)/(E_1*E_2)*(c_1/2*(1/(E_1 + E_2)**2 - 1/(E_1*E_2)) - 4*(E_1*E_2/c**2 + p**2)/c**2 ) )  -  purple/(E_1*E_2) * ( 1/4*(1/m_1 + m_2/m_1**2)*c_r_dot/c**2 + ( (c_1_dot*(E_1 + E_2) + c_1*(E_1_dot + E_2_dot))/(E_1*E_2) - c_1*(E_1 + E_2)/(E_1*E_2)**2*green) * ( c_1/2*(1/(E_1 + E_2)**2 - 1/(E_1*E_2)) - 4*(E_1*E_2/c**2 +  p**2)/c**2 )  +  c_1*(E_1 + E_2)/(E_1*E_2) * ( c_1_dot/2*(1/(E_1 + E_2)**2 - 1/(E_1*E_2)) + c_1/2*(-2*(E_1_dot + E_2_dot)/(E_1 + E_2)**3 + green/(E_1*E_2)**2) - 4/c**2*(green/c**2 + 2*np.dot(p_vec, p_vec_dot) ) ) ) + 1/4*(1/m_1 + m_2/m_1**2)/c**2*dc_rdp_dot  +  ( ( (E_1_dot + E_2_dot)/(E_1*E_2) - (E_1 + E_2)/(E_1*E_2)**2*green)*dc_1dp + (E_1 + E_2)/(E_1*E_2)*dc_1dp_dot + (c_1_dot/(E_1*E_2) - c_1/(E_1*E_2)**2*green )* ( dE_1dp + dE_2dp - (E_1 + E_2)/(E_1*E_2)*purple)  +  c_1/(E_1*E_2)* ( dE_1dp_dot + dE_2dp_dot - ( (E_1_dot + E_2_dot)/(E_1*E_2) - (E_1 + E_2)/(E_1*E_2)**2*green )*purple - (E_1 + E_2)/(E_1*E_2)*yellow ) ) * (c_1/2*(1/(E_1 + E_2)**2 - 1/(E_1*E_2)) - 4*(E_1*E_2/c**2 + p**2)/c**2)  +  ( (E_1 + E_2)/(E_1*E_2)*dc_1dp + c_1/(E_1*E_2)* ( dE_1dp + dE_2dp - (E_1 + E_2)/(E_1*E_2)*purple ) ) * (c_1_dot/2*( 1/(E_1 + E_2)**2 - 1/(E_1*E_2) ) + c_1/2*(-2*(E_1_dot + E_2_dot)/(E_1 + E_2)**3 + green/(E_1*E_2)**2) - 4/c**2*(green/c**2 + 2*np.dot(p_vec, p_vec_dot) ) )  +  ( ( c_1_dot*(E_1 + E_2) + c_1*(E_1_dot + E_2_dot) )/(E_1*E_2) - c_1*(E_1 + E_2)/(E_1*E_2)**2*green )*( 1/2*dc_1dp*(1/(E_1 + E_2)**2 - 1/(E_1*E_2)) + c_1/2*( -2*(dE_1dp + dE_2dp)/(E_1 + E_2)**3 + purple/(E_1*E_2)**2 ) - 4/c**2*( purple/c**2 + 2*p_vec ) ) + c_1*(E_1 + E_2)/(E_1*E_2) * ( 1/2*dc_1dp_dot * (1/(E_1 + E_2)**2 - 1/(E_1*E_2)) + 1/2*dc_1dp*( -2*(E_1_dot + E_2_dot)/(E_1 + E_2)**3 + green/(E_1*E_2)**2 ) + c_1_dot/2*( -2*(dE_1dp + dE_2dp)/(E_1 + E_2)**3 + purple/(E_1*E_2)**2 )  +  c_1/2*( -2/(E_1 + E_2)**3*( dE_1dp_dot + dE_2dp_dot + purple*green  ) + 6*(dE_1dp + dE_2dp)/(E_1 + E_2)**4*(E_1_dot + E_2_dot) + yellow/(E_1*E_2)**2)  - 4/c**2*(yellow/c**2 + 2*p_vec_dot)  ) )

    a = a_kin + a_1PM + a_2PM

    a_x, a_y = a[0], a[1]
    return v_vec_x, v_vec_y, a_x, a_y


# --------------------------------------------------------------------------- #
# Terminal events for solve_ivp
# --------------------------------------------------------------------------- #
def plunge(t, s_cm, m_1, m_2, r_max, R_S=1, DL=True):
    # stop when the separation reaches the (Schwarzschild) radius R_S
    r_vec_x, r_vec_y, v_vec_x, v_vec_y = s_cm
    r_vec = np.array([r_vec_x, r_vec_y])
    r = np.linalg.norm(r_vec)
    return r - R_S
plunge.terminal = True


def scatter(t, s_cm, m_1, m_2, r_max, R_S=1, DL=True):
    # stop when the separation reaches the outer bound r_max
    r_vec_x, r_vec_y, v_vec_x, v_vec_y = s_cm
    r_vec = np.array([r_vec_x, r_vec_y])
    r = np.linalg.norm(r_vec)
    return r_max - r
scatter.terminal = True


def superluminal(t, s_cm, m_1, m_2, r_max, R_S=1, DL=True):
    # stop if the relative speed reaches c
    if DL == True:
        G, c = 1, 1
    else:
        c = const.c
        G = const.G
    r_vec_x, r_vec_y, v_vec_x, v_vec_y = s_cm
    v_vec = np.array([v_vec_x, v_vec_y])
    v = np.linalg.norm(v_vec)
    return c - v
superluminal.terminal = True


# --------------------------------------------------------------------------- #
# Integration
# --------------------------------------------------------------------------- #
def sToCOM(s):
    x_1, x_2, v_1, v_2, m_1, m_2 = s
    r_vec = x_1 - x_2
    v_vec = v_1 - v_2
    s_cm = r_vec[0], r_vec[1], v_vec[0], v_vec[1]
    return s_cm


def run_model(s_cm, m_1, m_2, t_max=int(2e4), dt=2e-1, R_S=1, d=5, mode='N', DL=True):
    """Integrate the two-body EOM in the COM frame.

    Returns (t, positions, velocities, r, t_max, sol) where
      positions  : (3, 2, N) = [pos1, pos2, posCM]
      velocities : (2, 2, N) = [vel1, vel2]
      r          : (N,)      separation
      t_max      : actual (possibly event-truncated) end time
    """
    if DL == True:
        G, c = 1, 1
    elif DL == False:
        c = const.c
        G = const.G
    else:
        print("Error: DL not chosen.")
        return

    y0 = r_vec_x, r_vec_y, v_vec_x, v_vec_y = s_cm
    r_vec = np.array([r_vec_x, r_vec_y])
    r_max = d * np.linalg.norm(r_vec)

    t_span = [0, int(t_max)]
    t_eval = np.arange(t_span[0], t_span[1], dt)

    if mode == 'N':
        sol = solve_ivp(accelerationN, t_span=t_span, y0=y0, t_eval=t_eval, events=[scatter, plunge, superluminal], method='DOP853', args=(m_1, m_2, r_max, R_S, DL))
    elif mode == 'EIH':
        sol = solve_ivp(accelerationEIH, t_span=t_span, y0=y0, t_eval=t_eval, events=[scatter, plunge, superluminal], method='DOP853', args=(m_1, m_2, r_max, R_S, DL))
    elif mode == 'N_rel':
        sol = solve_ivp(accelerationN_rel, t_span=t_span, y0=y0, t_eval=t_eval, events=[scatter, plunge, superluminal], method='DOP853', args=(m_1, m_2, r_max, R_S, DL))
    elif mode == '1PN':
        sol = solve_ivp(acceleration1PN, t_span=t_span, y0=y0, t_eval=t_eval, events=[scatter, plunge, superluminal], method='DOP853', args=(m_1, m_2, r_max, R_S, DL))
    elif mode == '1PM':
        sol = solve_ivp(acceleration1PM, t_span=t_span, y0=y0, t_eval=t_eval, events=[scatter, plunge, superluminal], method='DOP853', args=(m_1, m_2, r_max, R_S, DL))
    elif mode == '2PM':
        sol = solve_ivp(acceleration2PM, t_span=t_span, y0=y0, t_eval=t_eval, events=[scatter, plunge, superluminal], method='DOP853', args=(m_1, m_2, r_max, R_S, DL))
    else:
        print("Error: Valid mode not chosen.")
        return

    t = sol.t
    pos_rel = np.array([sol.y[0], sol.y[1]])
    vel_rel = np.array([sol.y[2], sol.y[3]])
    t_events = sol.t_events

    pos1 = m_2 / (m_1 + m_2) * pos_rel
    pos2 = -m_1 / (m_1 + m_2) * pos_rel
    vel1 = m_2 / (m_1 + m_2) * vel_rel
    vel2 = -m_1 / (m_1 + m_2) * vel_rel

    r = rel_len(pos1, pos2)
    pos_CM = []
    for i in range(len(pos1)):
        pos_CM.append(findrCM(pos1[i], pos2[i], m_1, m_2))

    positions = np.array([pos1, pos2, pos_CM])
    velocities = np.array([vel1, vel2])

    if len(t_events[0]) > 0:
        t_max = t_events[0][0]
    elif len(t_events[1]) > 0:
        t_max = t_events[1][0]
    else:
        t_max = t_max
    return t, positions, velocities, r, t_max, sol


# --------------------------------------------------------------------------- #
# Data I/O
# --------------------------------------------------------------------------- #
def save_orbits(name, outdir, case, t, positions, velocities, r, t_max):
    """Save an integrated orbit to `<outdir>/<name>.npz` + a `.json` sidecar."""
    os.makedirs(outdir, exist_ok=True)
    npz_path = os.path.join(outdir, f"{name}.npz")
    json_path = os.path.join(outdir, f"{name}.json")

    np.savez(
        npz_path,
        t=t,
        positions=positions,
        velocities=velocities,
        r=r,
        t_max=np.array([t_max]),
    )
    sidecar = dict(case)
    sidecar["_meta"] = {
        "t_max_achieved": float(t_max),
        "n_points": int(len(t)),
        "npz": os.path.basename(npz_path),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)
    return npz_path, json_path


def load_orbits(name, indir):
    """Load `<indir>/<name>.npz` (+ sidecar if present)."""
    npz_path = os.path.join(indir, f"{name}.npz")
    json_path = os.path.join(indir, f"{name}.json")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"No data file found: {npz_path}")
    data = np.load(npz_path)
    case = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            case = json.load(f)
    return {
        "t": data["t"],
        "positions": data["positions"],
        "velocities": data["velocities"],
        "r": data["r"],
        "t_max": float(data["t_max"][0]),
        "case": case,
    }


# --------------------------------------------------------------------------- #
# Shared plotting (used by the preview and by the SVG export)
# --------------------------------------------------------------------------- #
def _plot_lims_two_body(pos1, pos2):
    x1max, x1min = pos1[0].max(), pos1[0].min()
    x2max, x2min = pos2[0].max(), pos2[0].min()
    y1max, y1min = pos1[1].max(), pos1[1].min()
    y2max, y2min = pos2[1].max(), pos2[1].min()
    xmax, xmin = max(x1max, x2max), min(x1min, x2min)
    ymax, ymin = max(y1max, y2max), min(y1min, y2min)
    start = max(abs(pos1[0][0]), abs(pos1[1][0]), abs(pos2[0][0]), abs(pos2[1][0]))
    end = max(abs(pos1[0][-1]), abs(pos1[1][-1]), abs(pos2[0][-1]), abs(pos2[1][-1]))
    pad = 0.1 * max(start, end)
    return (xmin - pad, xmax + pad), (ymin - pad, ymax + pad)


def plot_orbit(positions, ax=None, pretty=True, figsize=(11, 11), title="", show=False):
    """Draw the two-body orbital trajectories.

    positions : (3, 2, N) = [pos1, pos2, posCM]
    pretty    : True -> clean (no ticks/grid/axes), False -> grid + labels + legend
    Returns the matplotlib Figure.
    """
    x_1, x_2, x_cm = positions[0], positions[1], positions[2]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    xlim, ylim = _plot_lims_two_body(x_1, x_2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    fig.patch.set_facecolor("white")

    if pretty:
        ax.plot(0, 0, color="black", marker=".", markersize=3, alpha=0.5)
        ax.plot(x_1[0], x_1[1], "b", label="m_1")
        ax.plot(x_1[0][0], x_1[1][0], marker="o", markersize=7, markerfacecolor="white",
                markeredgecolor="blue", markeredgewidth=2, linestyle="none", label="m_1 start")
        ax.plot(x_1[0][-1], x_1[1][-1], "bo", markersize=10, label="m_1 stop")
        ax.plot(x_2[0], x_2[1], "r", label="m_2")
        ax.plot(x_2[0][0], x_2[1][0], marker="o", markersize=7, markerfacecolor="white",
                markeredgecolor="red", markeredgewidth=2, linestyle="none", label="m_2 start")
        ax.plot(x_2[0][-1], x_2[1][-1], "ro", markersize=10, label="m_2 stop")
        ax.tick_params(left=False, right=False, labelleft=False, bottom=False, labelbottom=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
    else:
        ax.plot(x_cm[0], x_cm[1], "g:", label="CM")
        ax.plot(x_cm[0][0], x_cm[1][0], "gx", markersize=7)
        ax.plot(x_cm[0][-1], x_cm[1][-1], "g.", markersize=7)
        ax.plot(x_1[0], x_1[1], "b")
        ax.plot(x_1[0][0], x_1[1][0], marker="o", markersize=7, markerfacecolor="white",
                markeredgecolor="blue", markeredgewidth=2, linestyle="none", label="m_1 start")
        ax.plot(x_1[0][-1], x_1[1][-1], "bo", markersize=10, label="m_1 stop")
        ax.plot(x_2[0], x_2[1], "r")
        ax.plot(x_2[0][0], x_2[1][0], marker="o", markersize=7, markerfacecolor="white",
                markeredgecolor="red", markeredgewidth=2, linestyle="none", label="m_2 start")
        ax.plot(x_2[0][-1], x_2[1][-1], "ro", markersize=10, label="m_2 stop")
        ax.set_xlabel(r"$[G M_\odot/c^2]$", fontsize=12)
        ax.set_ylabel(r"$[G M_\odot/c^2]$", fontsize=12)
        ax.grid(c="grey", alpha=0.2, ls="--")
        ax.legend(facecolor="grey", fontsize=12, bbox_to_anchor=(1.01, 1), loc="upper left")

    if title:
        ax.set_title(title, fontsize=18)

    fig.tight_layout()
    if show:
        plt.show()
    return fig
