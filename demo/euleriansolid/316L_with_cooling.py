#!/usr/bin/env python

# 1D model. With cooling term; heat exits both at x=L and through
# the exposed part of the surface inside the 1D domain.
#
# 316L steel, cast, solid phase only. Melting point T0 = 1700 K.

from contextlib import contextmanager

from unpythonic.env import env

import mpmath
import numpy as np
import scipy.stats
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

# melting point [K]
# This is also used as the reference temperature for thermal expansion.
T0 = 1700

# --------------------------------------------------------------------------------
# This comes from:
#   Choong S. Kim. 1975. Thermophysical properties of stainless steels.
#   Report ANL-75-55. Argonne National Laboratory, Argonne, Illinois.

# Temperatures at which the the data are tabulated [K]
T_for_ραck = np.array(range(3, 18)) * 1e2

# Density [g/cm³] --> [kg/m³]
ρ_func = lambda T: (7.9841 - 2.6506e-4 * T - 1.1580e-7 * T**2) * 1e-3 / 0.01**3
ρ = ρ_func(T_for_ραck)

# Curiously, the values produced by the formula don't match this table,
# which is from the same reference. The formula says we should be at 7894.16 @ 300 K.
# At least it's in the same ballpark (0.8% relative difference from the tabulated value 7954 @ 300 K).
# ρ = np.array([7.954, 7.910, 7.864, 7.818, 7.771, 7.723, 7.674,
#               7.624, 7.574, 7.523, 7.471, 7.419, 7.365, 7.311, 7.256]) * 1e-3 / 0.01**3

# Coefficient of linear thermal expansion [1/K]
# Not available for T = 300 K; seems to be the reference temperature.
α = np.array([np.nan, 1.890, 1.917, 1.944, 1.973, 2.002, 2.031,
              2.061, 2.092, 2.123, 2.156, 2.188, 2.222, 2.256, 2.291]) * 1e-5

# Specific heat capacity [cal / (g K)] --> [J / (kg K)]
# 1 cal = 4.186 J
c_func = lambda T: (0.1097 + 3.174e-5 * T) * 4.186 / 1e-3
c = c_func(T_for_ραck)

# Heat conductivity [W / (cm K)] --> [W / (m K)]
k = np.array([0.1396, 0.1553, 0.1710, 0.1868, 0.2025, 0.2182, 0.2339,
              0.2496, 0.2653, 0.2810, 0.2967, 0.3125, 0.3282, 0.3439, 0.3596]) * 1 / 0.01

# --------------------------------------------------------------------------------
# This comes from Engineering Toolbox:
#   https://www.engineeringtoolbox.com/young-modulus-d_773.html

# # temperature [K]
# # The seemingly weird choices for temperature are because the original is in Fahrenheit.
# T_for_E = np.array([-200, -129, -73, 21, 93, 149, 204, 260,
#                     316, 371, 427, 482, 538, 593, 649]) + 273.15
#
# # Young's modulus [MPsi] --> [Pa]
# # 1 psi = 6894.7572931783 Pa
# Mpsi2Pa = 6894.7572931783 * 1e6  # 1 Mpsi --> Pa
# # E = np.array([209e9, 205e9, 201e9, 195e9, 190e9, 186e9, 183e9, 178e9,
# #               25.3 * Mpsi2Pa, 24.8 * Mpsi2Pa, 24.1 * Mpsi2Pa, 23.5 * Mpsi2Pa,
# #               22.8 * Mpsi2Pa, 22.1 * Mpsi2Pa, 21.2 * Mpsi2Pa])
# E = np.array([30.3, 29.7, 29.1, 28.3, 27.6, 27.0, 26.5, 25.8,
#               25.3, 24.8, 24.1, 23.5, 22.8, 22.1, 21.2]) * Mpsi2Pa

# --------------------------------------------------------------------------------
# This comes from:
#   Nickel institute. 2020. High Temperature Characteristics of Stainless Steel.
#   /A Designers' Handbook Series/ N${}^{\mathrm{o}}$ 9004. Republication of the handbook published
#   in 1979 by the Committee of Stainless Steel Producers, American Iron and Steel Institute.
#
# NOTE: The data for the heat conductivity $k$ provided in Nickel institute (2020)
# is given correctly only in the imperial unit "Btu/hr/sq ft/ft/°F", which seems
# to actually mean Btu / hr / ft / °F, judging by converting the values to
# metric and comparing to metric values in other references.
#
# According to this online unit converter:
#     https://www.gordonengland.co.uk/conversion/thermcon.htm
# the conversion factor is
#     1 Btu / hr / ft / °F = 1.730735 W / (m K)
#     1 W / (m K) =  0.5777893 Btu / hr / ft / °F
# which yields (p. 43, type 316 S31600)
#      9.4 Btu / hr / ft / °F → 16.27 W/(m K)  @ 100°C
#     12.4 Btu / hr / ft / °F → 21.46 W/(m K)  @ 500°C
# This is well within the ballpark of other references.
#
# However, we need only the Young modulus table, p. 43. This is for type 316 steel.

T_for_E = np.array([27, 93, 149, 204, 260,
                    316, 371, 427, 482, 538,
                    593, 649, 704, 760, 816]) + 273.15  # K
E = np.array([193, 194, 190, 185, 181,
              177, 172, 167, 162, 157,
              153, 148, 143, 138, 132]) * 1e9  # Pa

# --------------------------------------------------------------------------------
# This comes from:
#   E. Sarlin, Y. Liu, M. Vippola, M. Zogg, P. Ermanni, J. Vuorinen, and T.Lepistö. 2012.
#   Vibration damping properties of steel/rubber/composite hybrid structures.
#   /Composite structures/ *94* (11), 3327--3335. doi:10.1016/j.compstruct.2012.04.035

# The only available data for damping in solid steel is the /loss factor/ of harmonic vibrations
# at various frequencies. Here's a very rough estimate based on the plotted figure from the reference.

# frequency [Hz]
# These were measured, at a resolution of 1 mm, from a screenshot with
# the zoom set so that one grid spacing of 100 Hz = 5 cm on the monitor.
# The numbers are in cm of on-screen distance from the vertical axis.
# f = np.array([3.2, 4.9, 9.2, 10.7, 13.0, 16.3, 19.5, 20.8, 23.5]) / 5.0 * 100
# --> [ 64.  98. 184. 214. 260. 326. 390. 416. 470.] [Hz]

# angular frequency [rad/s]
# ω = 2 * np.pi * f
# --> [ 402.12385966  615.7521601  1156.10609652 1344.60165574 1633.62817987
#       2048.31841014 2450.4422698  2613.80508779 2953.09709437] [rad/s]

# loss factor [nondimensional]
# Same method. One grid spacing of 0.02 = 7.2 cm on the monitor.
# The numbers are in cm of on-screen distance from the horizontal axis.
# LF = np.array([3.3, 11.4, 2.4, 2.4, 3.3, 2.0, 4.4, 1.2, 2.4]) / 7.2 * 0.02
# --> [0.00916667 0.03166667 0.00666667 0.00666667 0.00916667 0.00555556
#      0.01222222 0.00333333 0.00666667] [nondimensional]

# --> LF / ω ≈
# [2.27956299e-05 5.14276177e-05 5.76648345e-06 4.95809792e-06
#  5.61123197e-06 2.71225193e-06 4.98776175e-06 1.27527999e-06
#  2.25751692e-06]
# --> mean value 1.1310207939695e-05

# The same reference says the maximum value the authors obtained for the loss factor,
# as well as the maximum value reported in three different pieces of literature, is 0.01.
# So let's use that.

LF = np.array([0.01])
# The lowest frequency in the tabulated data was 64 Hz, so using 50 Hz
# should give us ballpark estimate of the highest possible retardation time LF/ω.
ω = np.array([2 * np.pi * 50.0])  # ω = 2 π f
# --> LF / ω = 3.1831e-05

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
# Note that the Kelvin--Voigt retardation time is defined as
#   t_{ret} ≡ η / E
# so actually for Kelvin--Voigt,
#   LF(ω) / ω = t_{ret}
#
# Taking a suitable aggregate (perhaps the most suitable is the maximum)
# over the different frequencies gives us a rough approximation of η.
#
# We have no data at different temperatures, so the best we can do with this data
# is to model the loss factor as a constant with respect to temperature.
#
# viscous modulus [Pa s]
# We average over the frequency, and then take the temperature dependence from E.
η = np.max(LF / ω) * E

# With the tabulated data above, we obtain
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
    USE_MPMATH = True  # arbitrary precision software floating point
    NPOINTS = 1001  # grid points for plotting

    L = 2.0  # length of Eulerian domain being modeled [m]
    v = 50e-3  # printing laser velocity [m/s]
    b = 0.0  # external force density [N/kg] = [m/s²]
    b0 = 9.81  # characteristic external force density [N/kg] = [m/s²]

    # material.η = material.E  # → t_{ret} = 1 s, fictitious viscous material
    # material.η = 1e-8

    # material.α = 0  # DEBUG

    bbar = b / b0  # nondimensional external force density

    # Cooling via free edge
    #
    T_air = 22 + 273.15  # [K]

    # # Convective model, Newton's law of cooling. Empirical formula from
    # #     https://www.engineeringtoolbox.com/convective-heat-transfer-d_430.html
    # v_air = 2  # [m/s]
    # Γ = 12.12 - 1.16 * v_air + 11.6 * v_air**(1 / 2)  # air, [W/m²K], for v_air ∈ [2, 20] [m/s]

    # This one comes from:
    #   Philip Kosky, Robert Balmer, William Keat, and George Wise. 2020.
    #   /Exploring Engineering: An Introduction to Engineering and Design/. 5th edition.
    #   Academic Press. ISBN\nbsp{}978-0-12-815073-3. doi:10.1016/C2017-0-01871-2
    #
    # Table 14.2:
    Γ = 10.0  # air, [W/m²K]; a high value for free convection, simultaneously, a low value for forced convection.

    # Then we convert to [W/(kg K)], taking into account the shape of the cross-section:
    #
    # # Rectangular, dimensions:
    # W = 50e-6  # width of line being printed, [m]
    # H = 50e-6  # height of line being printed, [m]
    #
    # # Choose which surfaces of the cross-section are exposed to air:
    #
    # # All four surfaces
    # dAdm = (2 * (W + H)) / (material.ρ * W * H)  # [m²/kg]
    # # Top and sides (bottom insulated)
    # dAdm = (W + 2 * H) / (material.ρ * W * H)
    # # Top only (sides and bottom insulated)
    # dAdm = (1 / material.ρ * H)

    # # Half-circle, round part exposed to air:
    R = 25e-6  # radius of the circle, [m]
    dAdm = 2 / (material.ρ * R)  # [m²/kg]

    r = dAdm * Γ  # heat transfer per unit mass, [W/(kg K)]

    # boundary conditions
    T1 = T0  # temperature at left endpoint [K]
    T2 = None  # None = use thermal equilibrium boundary condition at right endpoint
    # T2 = 1650  # 0.95 * T0  # temperature at right endpoint [K], the value is a wild guess

    u1 = 0  # u, displacement at left endpoint [m]
    ε1 = 0  # ε = ∂u/∂x, strain at left endpoint [nondimensional]
    w1 = 0  # w = ∂²u/∂x² at left endpoint [1/m]

    # nondimensional values for boundary conditions
    T1bar = T1 / T0
    T2bar = (T2 / T0) if T2 else None
    Tbar_air = T_air / T0
    u1bar = u1 / L
    ε1bar = ε1
    w1bar = w1 * L

    # independent nondimensional parameters (via Buckingham's π theorem)
    a1 = material.ρ * v**2 / material.E  # inertial modulus vs. elastic modulus
    a2 = material.E * L / (material.η * v)  # laser travel time vs. Kelvin-Voigt retardation time
    a3 = L * b0 / v**2   # laser travel time vs. characteristic time of external acceleration
    a4 = material.k / (material.c * material.η)  # thermal response vs. viscosity
    a5 = material.α * T0  # thermal expansion factor per unit of nondimensional temperature
    # # fourth power of thermoviscous vs. elastic wave velocities, only present in transient analysis
    # a6 = material.ρ**2 * material.E**-2 * material.η**-1 * material.k * material.c * material.α**-2
    # Flipping it the other way around, and taking the fourth root:
    # thermoviscous vs. elastic wave velocity
    a6 = (material.k * material.c / (material.η * material.α**2))**(-1 / 4) * np.sqrt(material.E / material.ρ)
    # laser travel time vs. characteristic cooling time
    a7 = L * r / (v * material.c)

    # nondimensional solution parameters (these appear in the analytical solution)
    λ = a2 * a4**-1 * a1
    P = a2 * (1 - a1)
    β = a1 * a2 * a3
    # κ = a7 * λ  # when v ≠ 0
    Δ = λ * (λ + 4 * a7)  # discriminant of the characteristic polynomial of the thermal problem
    assert Δ > 0  # for our solution, μ1 and μ2 are distinct and nonzero.
    μ1 = (1 / 2) * (λ + Δ**(1 / 2))
    μ2 = (1 / 2) * (λ - Δ**(1 / 2))
    γ1 = a5 * μ1 * (a2 + μ1) / (P + μ1)
    γ2 = a5 * μ2 * (a2 + μ2) / (P + μ2)

    if USE_MPMATH:
        exp = np.vectorize(lambda x: mpmath.exp(x))
        μ1m = mpmath.mpf(μ1)  # need extra precision when computing Qj
        μ2m = mpmath.mpf(μ2)
    else:
        exp = np.exp
        μ1m = μ1
        μ2m = μ2

    # Constants of integration for the thermal problem
    if T2:  # Tbar|x=0 = T1bar,  Tbar|x=1 = T2bar  (known endpoint temperature)
        assert T2bar is not None
        Q1 = float(((T2bar - Tbar_air) - exp(μ2) * (T1bar - Tbar_air)) / (exp(μ1m) - exp(μ2m)))
        Q2 = float(((T2bar - Tbar_air) - exp(μ1) * (T1bar - Tbar_air)) / (exp(μ2m) - exp(μ1m)))
    else:  # Tbar|x=0 = T1bar,  ∂Tbar/∂xbar|x=1 = 0  (thermal equilibrium at right endpoint)
        assert T2bar is None
        Q1 = float((T1bar - Tbar_air) / (1 - (μ1 * exp(μ1m)) / (μ2 * exp(μ2m))))
        Q2 = float((T1bar - Tbar_air) / (1 - (μ2 * exp(μ2m)) / (μ1 * exp(μ1m))))

    # Constants of integration for the mechanical part of the full thermomechanical problem
    # from boundary conditions  ubar|x=0 = u1bar,  εbar|x=0 = ε1bar,  w|x=0 = w1bar
    u0 = u1bar - Q1 * (μ1**-2 - P**-2) * γ1 - Q2 * (μ2**-2 - P**-2) * γ2 - w1bar / P**2 - β * b / P**3
    ε0 = ε1bar - Q1 * (μ1**-1 + P**-1) * γ1 - Q2 * (μ2**-1 + P**-1) * γ2 + w1bar / P + β * b / P**2
    w0 = w1bar + β * b / P - Q1 * γ1 - Q2 * γ2

    print("=" * 80)
    print("Parameter values:")
    print("  Laser scan parameters:")
    print(f"    L = {1000*L:0.6g} mm (domain length)")
    print(f"    v = {1000*v:0.6g} mm/s (laser scan speed)")
    print(f"    L/v = {L/v:0.6g} s (laser travel time)")
    print("  Viscosity to elasticity ratio (Kelvin-Voigt retardation time)")
    print(f"    η/E = {material.η / material.E:0.6g} s")
    print("  Cooling via exposed surface:")
    print(f"    Heat transfer into surrounding air {Γ:0.6g} W/(m² K)")
    print(f"    Area of exposed surface per unit mass {dAdm:0.6g} m²/kg")
    print(f"    Heat transfer per unit mass {r:0.6g} W/(kg K)")
    print(f"    Characteristic cooling time c/r = {material.c/r:0.6g} s")
    print("  Nondimensional parameters:")
    print(f"    a1 = ρv²/E                 = {a1:0.6g} (inertial modulus vs. elastic modulus)")
    print(f"    a2 = EL/(ηv)               = {a2:0.6g} (laser travel time vs. KV retardation time)")
    print(f"    a3 = Lb₀/v²                = {a3:0.6g} (laser travel time vs. characteristic time of ext. accel.)")
    print(f"    a4 = k/(cη)                = {a4:0.6g} (thermal vs. mechanical viscosity)")
    print(f"    a5 = αT₀                   = {a5:0.6g} (thermal expansion per unit of nondim. temperature)")
    print(f"    a6 = √⁴(kc/(ηα²)) / √(E/ρ) = {a6:0.6g} (thermoviscous vs. elastic wave velocity; unused in steady-state analysis)")
    print(f"    a7 = Lr/(vc)               = {a7:0.6g} (laser travel time vs. characteristic cooling time)")
    print("  Solution parameters:")
    print(f"    λ  = a₂ a₄⁻¹ a₁                  = {λ:0.6g}")
    print(f"    P  = a₂ (1 - a₁)                 = {P:0.6g}")
    print(f"    μ1 = (1/2) (λ + √(λ [λ + 4 a₇])) = {μ1:0.6g}")
    print(f"    μ2 = (1/2) (λ - √(λ [λ + 4 a₇])) = {μ2:0.6g}")
    print(f"    γ1 = a₅ μ₁ (a₂ + μ₁) / (P + μ₁)  = {γ1:0.6g}")
    print(f"    γ2 = a₅ μ₂ (a₂ + μ₂) / (P + μ₂)  = {γ2:0.6g}")
    print(f"    β  = a₁ a₂ a₃                    = {β:0.6g}")
    print("  Integration constants (determined by boundary conditions):")
    print(f"    Q1 = {Q1:0.6g}")
    print(f"    Q2 = {Q2:0.6g}")
    print(f"    w0 = {w0:0.6g}")
    print(f"    ε0 = {ε0:0.6g}")
    print(f"    u0 = {u0:0.6g}")
    print("=" * 80)

    # Nondimensional analytical solution of 1D model with constant coefficients
    # (and constant external force density).
    #
    # The argument `x` is actually nondimensional x ("xbar"). Can be `float` or `mpmath.mpf`.
    # The argument `xx` is a rank-1 np.array of nondimensional x.
    def ubar(x):
        return (u0 + ε0 * x + P**-2 * w0 * exp(-P * x) -
                β / 2 * bbar * P**-1 * x**2 +
                Q1 * μ1**-2 * γ1 * exp(μ1 * x) +
                Q2 * μ2**-2 * γ2 * exp(μ2 * x))
    def ε(x):  # ε = ∂ubar/∂xbar = ∂u/∂x (the conversion factors cancel)
        return (ε0 - P**-1 * w0 * exp(-P * x) -
                β * bbar * P**-1 * x +
                Q1 * μ1**-1 * γ1 * exp(μ1 * x) +
                Q2 * μ2**-1 * γ2 * exp(μ2 * x))
    def dεdxbar(x):  # ∂ε/∂x, nondimensional. This is the auxiliary variable w.
        return (w0 * exp(-P * x) +
                Q1 * γ1 * exp(μ1 * x) +
                Q2 * γ2 * exp(μ2 * x))
    def Tbar(x):
        return Tbar_air + Q1 * exp(μ1 * x) + Q2 * exp(μ2 * x)
    def dTbardxbar(x):
        return Q1 * μ1 * exp(μ1 * x) + Q2 * μ2 * exp(μ2 * x)
    def εth(x):
        return material.α * T0 * (Tbar(x) - 1)
    def dεthdxbar(x):
        return material.α * T0 * dTbardxbar(x)
    def εve(x):
        return ε(x) - εth(x)
    def dεvedxbar(x):
        return dεdxbar(x) - dεthdxbar(x)
    def σ(x):  # dimensional, [Pa]
        # stress, axially moving thermoviscoelastic Kelvin-Voigt, constant coefficients
        dεdx_ve = dεvedxbar(x) / L  # ∂ε_{ve}/∂x, [1/m]
        # [Pa] [1] + [Pa s] [m/s] [1/m] = [Pa] + [Pa], ok!
        return material.E * εve(x) + material.η * v * dεdx_ve  # σ, [Pa]
        # # old debug stuff
        # print("  dε/dx...")
        # print("  dT/dx...")
        # print("  dε_{th}/dx...")
        # print("  dε_{ve}/dx...")
        # print(f"max (abs.) viscoelastic strain {np.max(np.abs(np.array(εε_ve, dtype=np.float64))):0.6g}")
        # print(f"max (abs.) derivative of viscoelastic strain {np.max(np.abs(np.array(dεdx_ve, dtype=np.float64))):0.6g} 1/m")
        # Analytical expression for stress (same output, so it is correct)
        # σσ = (material.E * (ε0 - material.α * T0 * (Tbar_air - 1)) -
        #       β * bbar * P**-1 * (material.E * x + material.η * v / L) -
        #       w0 * (material.E * P**-1 - material.η * v / L) * exp(-P * x) +
        #       (material.E + (material.η * v / L) * μ1) * Q1 * ((γ1 / μ1) - material.α * T0) * exp(μ1 * x) +
        #       (material.E + (material.η * v / L) * μ2) * Q2 * ((γ2 / μ2) - material.α * T0) * exp(μ2 * x))  # [Pa]

    # Plotting routines
    #
    def make_main_figure(xx):  # xx: rank-1 np.array of nondimensional x
        xx_millimetres = (1e3 * L) * xx
        if USE_MPMATH:
            xx = np.array(xx, dtype=mpmath.mpf)

        fig, ax = plt.subplots(2, 2, constrained_layout=True)

        # nondimensional u(x)
        uubar = ubar(xx)
        uu = L * uubar  # [m]
        ax[0, 0].plot(xx_millimetres, 1e3 * uu)  # [mm, mm]
        ax[0, 0].grid(b=True, which="both")
        ax[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax[0, 0].axis("tight")
        ax[0, 0].set_ylabel(r"$u$ [mm]")

        # ε = ∂u/∂x
        εεbar = ε(xx)
        εε = εεbar  # for the strain, conversion factors cancel

        # split the strain into viscoelastic and thermal contributions:
        TTbar = Tbar(xx)
        TT = T0 * TTbar  # T(x), [K]
        εε_th = εth(xx)
        εε_ve = εve(xx)

        ax[0, 1].plot(xx_millimetres, 100 * εε, label=r"$\varepsilon$")  # [mm, %]
        ax[0, 1].plot(xx_millimetres, 100 * εε_th, label=r"$\varepsilon_{\mathrm{th}}$")  # [mm, %]
        ax[0, 1].plot(xx_millimetres, 100 * εε_ve, label=r"$\varepsilon_{\mathrm{ve}}$")  # [mm, %]
        ax[0, 1].axis("tight")
        ax[0, 1].hlines(0, 0, 1, transform=ax[0, 1].get_yaxis_transform(), color="#808080", linestyle="dashed")
        ax[0, 1].yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax[0, 1].grid(b=True, which="both")
        ax[0, 1].set_ylabel(r"$\varepsilon$ [%]")
        ax[0, 1].legend(loc="best")

        σσ = σ(xx)
        ax[1, 0].plot(xx_millimetres, σσ)  # [mm, Pa]
        ax[1, 0].grid(b=True, which="both")
        ax[1, 0].yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax[1, 0].axis("tight")
        ax[1, 0].set_ylabel(r"$\sigma$ [Pa]")
        print(f"max (abs.) stress for x ∈ [{xx_millimetres[0]:0.6g}, {xx_millimetres[-1]:0.6g}] mm is {np.max(np.abs(np.array(σσ, dtype=np.float64))):0.6g} Pa")

        # temperature
        TT_celsius = TT - 273.15  # [°C]
        T_air_celsius = T_air - 273.15  # [°C]
        T0_celsius = T0 - 273.15
        ax[1, 1].plot(xx_millimetres, TT_celsius)  # [mm, °C]
        ax[1, 1].axis("tight")
        ax[1, 1].hlines([T_air_celsius, T0_celsius], xx_millimetres[0], xx_millimetres[-1],
                        color="#808080", linestyle="dashed")
        ax[1, 1].grid(b=True, which="both")
        ax[1, 1].yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax[1, 1].set_ylabel(r"$T$ [°C]")

        ax[1, 0].set_xlabel(r"$x$ [mm]")
        ax[1, 1].set_xlabel(r"$x$ [mm]")

        return fig, ax

    # Separate figure of viscoelastic strain (small, saturates very quickly at the beginning of the domain)
    # xx: initial grid, will be refined to find the saturation point.
    def make_viscoelastic_figure(xx):
        if USE_MPMATH:
            xx = np.array(xx, dtype=mpmath.mpf)

        @contextmanager
        def mpl_largefont():
            oldsize = mpl.rcParams["font.size"]
            try:
                mpl.rcParams["font.size"] = oldsize * 2
                yield
            finally:
                mpl.rcParams["font.size"] = oldsize

        # Get `matplotlib`'s default color sequence.
        # https://matplotlib.org/stable/gallery/color/named_colors.html
        # https://matplotlib.org/stable/tutorials/intermediate/color_cycle.html
        colors = [item["color"] for item in mpl.rcParams["axes.prop_cycle"]]
        mycolor = colors[2]  # index of εε_ve in the previous plot, to make it the same color here.

        xx_detail = xx
        εε_ve_detail = εve(xx)
        for _ in range(2):  # refine once to allow for a coarse initial grid.
            big_ve_strain = εε_ve_detail > (np.max(εε_ve_detail) * 0.999)
            first_idx = np.argmax(big_ve_strain)
            xx_detail = np.linspace(0, xx_detail[first_idx], NPOINTS)
            εε_ve_detail = εve(xx_detail)
        xx_detail_millimetres = (1e3 * L) * xx_detail
        with mpl_largefont():
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.plot(xx_detail_millimetres, 100 * εε_ve_detail,
                     color=mycolor, label=r"$\varepsilon_{\mathrm{ve}}$")  # [mm, %]
            ax.axis("tight")
            ax.hlines(0, 0, 1, transform=ax.get_yaxis_transform(), color="#808080", linestyle="dashed")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))  # same as for the overview figure
            ax.grid(b=True, which="both")
            ax.set_ylabel(r"$\varepsilon_{\mathrm{ve}}$ [%]")
            ax.set_xlabel(r"$x$ [mm]")

        return fig, ax

    print(f"Evaluating solution...{' (mpmath enabled)' if USE_MPMATH else ''}")
    print(f"  Using {NPOINTS} points.")

    # Logarithmic spacing helps with detection of the saturation of
    # viscoelastic strain, for the detail plot.
    #
    # xx = np.linspace(0, 1, NPOINTS)
    xx = (np.logspace(0, np.log10(11), NPOINTS) - 1) / 10
    fig1, ax1 = make_main_figure(xx)
    fig2, ax2 = make_viscoelastic_figure(xx)

    xx = np.linspace(0, 500e-3 / L, NPOINTS)  # up to 500mm
    fig3, ax3 = make_main_figure(xx)

    print("  All done. Displaying.")
    plt.show()


def main():
    print("Model settings:")
    print(f"T₀ = {T0} K")
    print()
    const_material = constant_model()
    print()
    linear_model()

    plot_constant_model(const_material)

if __name__ == '__main__':
    main()
