#!/usr/bin/env python

# 1D model. No cooling term; heat only exits through the endpoint at x=L.
#
# 316L steel, cast, solid phase only. Melting point T0 = 1700 K.

from unpythonic.env import env

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

# melting point [K]
# This is also used as the reference temperature for thermal expansion.
T0 = 1700

# --------------------------------------------------------------------------------

# temperature [K]
T_for_ραck = np.array(range(3, 18)) * 1e2

# density [g/cm³] --> [kg/m³]
ρ_func = lambda T: (7.9841 - 2.6506e-4 * T - 1.1580e-7 * T**2) * 1e-3 / 0.01**3
ρ = ρ_func(T_for_ραck)

# Curiously, the values produced by the formula don't match this table, which looks like
# it's from the same source. The formula says we should be at 7894.16 @ 300 K.
# At least it's in the same ballpark (0.8% relative difference from the tabulated value 7954 @ 300 K).
# ρ = np.array([7.954, 7.910, 7.864, 7.818, 7.771, 7.723, 7.674,
#               7.624, 7.574, 7.523, 7.471, 7.419, 7.365, 7.311, 7.256]) * 1e-3 / 0.01**3

# coefficient of linear thermal expansion [1/K]
# not available for T = 300 K
α = np.array([np.nan, 1.890, 1.917, 1.944, 1.973, 2.002, 2.031,
              2.061, 2.092, 2.123, 2.156, 2.188, 2.222, 2.256, 2.291]) * 1e-5

# specific heat capacity [cal / (g K)] --> [J / (kg K)]
# 1 cal = 4.186 J
c_func = lambda T: (0.1097 + 3.174e-5 * T) * 4.186 / 1e-3
c = c_func(T_for_ραck)

# heat conductivity [W / (cm K)] --> [W / (m K)]
k = np.array([0.1396, 0.1553, 0.1710, 0.1868, 0.2025, 0.2182, 0.2339,
              0.2496, 0.2653, 0.2810, 0.2967, 0.3125, 0.3282, 0.3439, 0.3596]) * 1 / 0.01

# --------------------------------------------------------------------------------
# This apparently comes from a different source.

# temperature [K]
# The seemingly weird choices for temperature are because the original is in Fahrenheit.
T_for_E = np.array([-200, -129, -73, 21, 93, 149, 204, 260,
                    316, 371, 427, 482, 538, 593, 649]) + 273.15

# Young's modulus [MPsi] --> [Pa]
# 1 psi = 6894.7572931783 Pa
Mpsi2Pa = 6894.7572931783 * 1e6  # 1 Mpsi --> Pa
# E = np.array([209e9, 205e9, 201e9, 195e9, 190e9, 186e9, 183e9, 178e9,
#               25.3 * Mpsi2Pa, 24.8 * Mpsi2Pa, 24.1 * Mpsi2Pa, 23.5 * Mpsi2Pa,
#               22.8 * Mpsi2Pa, 22.1 * Mpsi2Pa, 21.2 * Mpsi2Pa])
E = np.array([30.3, 29.7, 29.1, 28.3, 27.6, 27.0, 26.5, 25.8,
              25.3, 24.8, 24.1, 23.5, 22.8, 22.1, 21.2]) * Mpsi2Pa

# --------------------------------------------------------------------------------
# This comes from yet another source.

# The only available data for damping is the /loss factor/ of harmonic vibrations at
# various frequencies, measured at an unknown temperature, and reported as a plotted figure.
# So here's a very rough estimate.

# frequency [Hz]
# These were measured, at a resolution of 1 mm, from a screenshot with
# the zoom set so that one grid spacing of 100 Hz = 5 cm on the monitor.
# The numbers are in cm of on-screen distance from the vertical axis.
f = np.array([3.2, 4.9, 9.2, 10.7, 13.0, 16.3, 19.5, 20.8, 23.5]) / 5.0 * 100
# --> [ 64.  98. 184. 214. 260. 326. 390. 416. 470.] [Hz]

# angular frequency [rad/s]
ω = 2 * np.pi * f
# --> [ 402.12385966  615.7521601  1156.10609652 1344.60165574 1633.62817987
#       2048.31841014 2450.4422698  2613.80508779 2953.09709437] [rad/s]

# loss factor [nondimensional]
# Same method. One grid spacing of 0.02 = 7.2 cm on the monitor.
# The numbers are in cm of on-screen distance from the horizontal axis.
LF = np.array([3.3, 11.4, 2.4, 2.4, 3.3, 2.0, 4.4, 1.2, 2.4]) / 7.2 * 0.02
# --> [0.00916667 0.03166667 0.00666667 0.00666667 0.00916667 0.00555556
#      0.01222222 0.00333333 0.00666667] [nondimensional]

# --> LF / ω ≈
# [2.27956299e-05 5.14276177e-05 5.76648345e-06 4.95809792e-06
#  5.61123197e-06 2.71225193e-06 4.98776175e-06 1.27527999e-06
#  2.25751692e-06]
# --> mean value 1.1310207939695e-05

# The loss factor is defined as
#   LF(ω) := G'' / G'
# where G'' is the loss modulus (representing viscous dissipation),
# and G' is the storage modulus (representing elastic response).
# Specifically for the Kelvin--Voigt model,
#   G'' = η ω
#   G'  = E
# where ω is the angular frequency [rad/s]. Therefore, for Kelvin--Voigt,
#   LF = η ω / E
# so the viscous modulus η [Pa s] can be expressed as
#   η = (LF(ω) / ω) E
#
# Averaging this over the different frequencies gives us a rough approximation of η.
#
# We have no data at different temperatures, so the best we can do with this data
# is to model η as a constant with respect to temperature.
#
# viscous modulus [Pa s]
# We average over the frequency, and then take the temperature dependence from E.
η = np.mean(LF / ω) * E

# We obtain
# [4066520.47528502 9174190.90528123 1028685.02141597  884476.84084364
#  1000989.65545477  483839.5806251   889768.58262647  227497.64896699
#   402719.24242668]
# The outlier (the second data point) looks suspicious, but we're keeping it for now.

# --------------------------------------------------------------------------------

def constant_model():
    """Approximate each parameter as a constant.

    Return a bunch with attributes ρ, α, c, k, E, η.
    Each value is a floating-point number.
    """
    material = env(ρ=np.mean(ρ),
                   α=np.mean(α[1:]),
                   c=np.mean(c),
                   k=np.mean(k),
                   E=np.mean(E),
                   η=np.mean(η))
    print("For model with constant coefficients:")
    print(f"ρ [kg/m³] ≈ {material.ρ:0.6g}")
    print(f"α [1/K] ≈ {material.α:0.6g}")
    print(f"c [J/(kg K)] ≈ {material.c:0.6g}")
    print(f"k [W/(m K)] ≈ {material.k:0.6g}")
    print(f"E [Pa] ≈ {material.E:0.6g}")
    print(f"η [Pa s] ≈ {material.η:0.6g}")
    return material

def linear_model():
    """Compute a linear regression for each parameter.

    Return a bunch with attributes ρ, α, c, k, E, η.

    Each value contains an bunch with two attributes:
    `a` and `fit`.

    `a` is a rank-1 array with the lowest-order coefficient
    first, so that `a[k]` is the coefficient of `T**k`, where
    `T` is the absolute temperature in Kelvin. This format is
    compatible with `np.poly`.

    `fit` contains the output structure from `scipy.stats.linregress`
    as-is (which is useful to examine the goodness of fit via the
    `rvalue` and `pvalue` attributes).
    """
    print("For model with linear coefficients:")

    fit = scipy.stats.linregress(T_for_ραck, ρ)
    s = "+" if np.sign(fit.slope) >= 0 else "-"
    print(f"ρ [kg/m³] ≈ {fit.intercept:0.6g} {s} {np.abs(fit.slope):0.6g} * T  [r = {fit.rvalue:0.6g}, p = {fit.pvalue:0.6g}]")
    ρ_linear = env(a=np.array([fit.intercept, fit.slope]),
                   fit=fit)

    fit = scipy.stats.linregress(T_for_ραck[1:], α[1:])
    s = "+" if np.sign(fit.slope) >= 0 else "-"
    print(f"α [1/K] ≈ {fit.intercept:0.6g} {s} {np.abs(fit.slope):0.6g} * T  [r = {fit.rvalue:0.6g}, p = {fit.pvalue:0.6g}]")
    α_linear = env(a=np.array([fit.intercept, fit.slope]),
                   fit=fit)

    fit = scipy.stats.linregress(T_for_ραck, c)
    s = "+" if np.sign(fit.slope) >= 0 else "-"
    print(f"c [J/(kg K)] ≈ {fit.intercept:0.6g} {s} {np.abs(fit.slope):0.6g} * T  [r = {fit.rvalue:0.6g}, p = {fit.pvalue:0.6g}]")
    c_linear = env(a=np.array([fit.intercept, fit.slope]),
                   fit=fit)

    fit = scipy.stats.linregress(T_for_ραck, k)
    s = "+" if np.sign(fit.slope) >= 0 else "-"
    print(f"k [W/(m K)] ≈ {fit.intercept:0.6g} {s} {np.abs(fit.slope):0.6g} * T  [r = {fit.rvalue:0.6g}, p = {fit.pvalue:0.6g}]")
    k_linear = env(a=np.array([fit.intercept, fit.slope]),
                   fit=fit)

    fit = scipy.stats.linregress(T_for_E, E)
    s = "+" if np.sign(fit.slope) >= 0 else "-"
    print(f"E [Pa] ≈ {fit.intercept:0.6g} {s} {np.abs(fit.slope):0.6g} * T  [r = {fit.rvalue:0.6g}, p = {fit.pvalue:0.6g}]")
    E_linear = env(a=np.array([fit.intercept, fit.slope]),
                   fit=fit)

    fit = scipy.stats.linregress(T_for_E, η)
    s = "+" if np.sign(fit.slope) >= 0 else "-"
    print(f"η [Pa s] ≈ {fit.intercept:0.6g} {s} {np.abs(fit.slope):0.6g} * T  [r = {fit.rvalue:0.6g}, p = {fit.pvalue:0.6g}]")
    η_linear = env(a=np.array([fit.intercept, fit.slope]),
                   fit=fit)

    material = env(ρ=ρ_linear,
                   α=α_linear,
                   c=c_linear,
                   k=k_linear,
                   E=E_linear,
                   η=η_linear)
    return material


def plot_constant_model(material):
    L = 1e-2  # length of Eulerian domain being modeled [m]
    v = 1e-1  # printing laser velocity [m/s]
    b = 0.0  # external force density [N/kg] = [m/s²]
    b0 = 9.81  # characteristic external force density [N/kg] = [m/s²]

    bbar = b / b0  # nondimensional external force density

    # material.α = 0  # DEBUG
    # material.η = material.E / 100
    # print(material.E / material.η)
    #
    # - Third mechanical boundary condition: depsilon/dx = 0 at midpoint
    # - Heat sink throughout domain (model a rod in free space)   s (T - Text)
    # - Heat flux boundary condition for heat equation (at x = 1)

    # boundary conditions
    T1 = T0  # temperature at left endpoint [K]
    T2 = 1650  # 0.95 * T0  # temperature at right endpoint [K], the value is a wild guess
    u1 = 0  # displacement at left endpoint [m]
    ε1 = 0  # strain at left endpoint [nondimensional]
    # u2: displacement at right endpoint [m]; essentially, the constant contribution to the strain.
    u2 = material.α * (T2 - T0) * L
    # u2 = 1e-3 * L
    # u2 = 0

    # nondimensional values for boundary conditions
    T1bar = T1 / T0
    T2bar = T2 / T0
    u1bar = u1 / L
    ε1bar = ε1
    u2bar = u2 / L

    # nondimensional parameters
    a1 = material.ρ * v**2 / material.E  # axial motion vs. elasticity
    a2 = material.E * L / (material.η * v)  # elasticity vs. viscosity in moving material
    a3 = L * b0 / v**2   # external force vs. axial motion
    a4 = material.k / (material.c * material.η)  # thermal response vs. viscosity
    a5 = material.α * T0  # strength of thermal expansion
    # # fourth power of thermoviscous vs. elastic wave velocities, only present in transient analysis
    # a6 = material.ρ**2 * material.E**-2 * material.η**-1 * material.k * material.c * material.α**-2
    # a6 = (material.k * material.c / (material.η * material.α**2)) * np.sqrt(material.E / material.ρ)**-4

    # parameters of the analytical solution
    λ = a2 * a4**-1 * a1
    # P = a2 - a4 * λ
    P = a2 * (1 - a1)  # equivalent
    γ = a5 * (a2 - λ) / (P - λ)
    β = a1 * a2 * a3

    # parameters determined from boundary conditions
    Q0 = λ * (T2bar - T1bar) / (np.exp(λ) - 1)  # nondimensional ∂T/∂x|x=0
    θ0 = T1bar  # nondimensional T|x=0
    w0 = ((ε1bar - (u2bar - u1bar) + λ**-2 * γ * Q0 * (-λ + np.exp(λ) - 1) + (1 / 2) * β * bbar * P**-1) *
          (P**-2 * (-P - np.exp(-P) + 1))**-1)
    ε0 = ((u2bar - u1bar) - λ**-2 * γ * Q0 * (np.exp(λ) - 1) + (1 / 2) * β * bbar * P**-1 -
          P**-2 * (np.exp(-P) - 1) * w0)
    u0 = u1bar - λ**-2 * γ * Q0 - P**-2 * w0

    # DEBUG
    print("=" * 80)
    print("DEBUG:")
    print("  Nondimensional parameters:")
    print(f"    a1 = {a1:0.6g}")
    print(f"    a2 = {a2:0.6g}")
    print(f"    a3 = {a3:0.6g}")
    print(f"    a4 = {a4:0.6g}")
    print(f"    a5 = {a5:0.6g}")
    print("  Solution parameters:")
    print(f"    λ = {λ:0.6g}")
    print(f"    P = {P:0.6g}")
    print(f"    γ = {γ:0.6g}")
    print(f"    β = {β:0.6g}")
    print("  Integration constants:")
    print(f"    Q0 = {Q0:0.6g}")
    print(f"    w0 = {w0:0.6g}")
    print(f"    ε0 = {ε0:0.6g}")
    print(f"    u0 = {u0:0.6g}")
    print("=" * 80)

    # nondimensional analytical solution of 1D model with constant coefficients (and constant external force density)
    xx = np.linspace(0, 1, 100001)
    xx_millimetres = (1e3 * L) * xx
    fig, ax = plt.subplots(2, 2, constrained_layout=True)

    # nondimensional u(x)
    uubar = (u0 + ε0 * xx + P**-2 * w0 * np.exp(-P * xx) -
             β / 2 * bbar * P**-1 * xx**2 +
             λ**-2 * γ * Q0 * np.exp(λ * xx))
    uu = L * uubar  # [m]
    ax[0, 0].plot(xx_millimetres, 1e3 * uu)  # [mm, mm]
    ax[0, 0].grid(b=True, which="both")
    ax[0, 0].axis("tight")
    ax[0, 0].set_ylabel(r"$u$ [mm]")

    # ε = ∂u/∂x
    εεbar = (ε0 - P**-1 * w0 * np.exp(-P * xx) -
             β * bbar * P**-1 * xx +
             λ**-1 * γ * Q0 * np.exp(λ * xx))
    εε = εεbar  # nondimensional
    ax[0, 1].plot(xx_millimetres, 100 * εε)  # [mm, %]
    ax[0, 1].hlines(0, 0, 1, transform=ax[0, 1].get_yaxis_transform(), color="#808080", linestyle="dashed")
    ax[0, 1].grid(b=True, which="both")
    ax[0, 1].axis("tight")
    ax[0, 1].set_ylabel(r"$\varepsilon$ [%]")

    # axially moving thermoviscoelastic Kelvin-Voigt, constant coefficients
    σσ = (material.E * (ε0 - material.α * T0 * (θ0 - Q0 / λ - 1)) -
          β * bbar * P**-1 * (material.E * xx + material.η * v / L) -
          w0 * (material.E * P**-1 - material.η * v / L) * np.exp(-P * xx) +
          (material.E * λ**-1 + material.η * v / L) * (γ - material.α * T0) * Q0 * np.exp(λ * xx))  # [Pa]
    ax[1, 0].plot(xx_millimetres, 1e-6 * σσ)  # [mm, MPa]
    ax[1, 0].grid(b=True, which="both")
    ax[1, 0].axis("tight")
    ax[1, 0].set_ylabel(r"$\sigma$ [MPa]")

    # nondimensional T(x)
    TTbar = (θ0 - Q0 / λ) + (Q0 / λ) * np.exp(λ * xx)
    TT = T0 * TTbar  # [K]
    ax[1, 1].plot(xx_millimetres, TT)  # [mm, K]
    ax[1, 1].grid(b=True, which="both")
    ax[1, 1].axis("tight")
    ax[1, 1].set_ylabel(r"$T$ [K]")

    # # Time rate of temperature [K/s] at the material parcel currently at x
    # #
    # # This is the material derivative in steady state:
    # #   dT/dt = v ∂T/∂x
    # # We obtain the value by reorganizing the heat equation for a moving material,
    # # and inserting our solution T(x). The expression given here holds when there are
    # # no external heat sources inside the domain.
    # d2Tbardxbar2 = Q0 * λ * np.exp(λ * xx)  # nondimensional
    # dTbardxbar_times_v = material.k / (material.ρ * material.c * L) * d2Tbardxbar2  # [m/s]
    # dTdt = (T0 / L) * dTbardxbar_times_v  # [K/s]
    # ax[1, 1].plot(xx_millimetres, dTdt)  # [mm, K/s]
    # ax[1, 1].grid(b=True, which="both")
    # ax[1, 1].axis("tight")
    # ax[1, 1].set_ylabel(r"$\mathrm{d} T / \mathrm{d} t$ [K/s]")

    ax[1, 0].set_xlabel(r"$x$ [mm]")
    ax[1, 1].set_xlabel(r"$x$ [mm]")
    plt.show()


def main():
    print("Model settings:")
    print(f"T0 = {T0} K")
    print()
    const_material = constant_model()
    print()
    linear_model()

    plot_constant_model(const_material)

if __name__ == '__main__':
    main()
