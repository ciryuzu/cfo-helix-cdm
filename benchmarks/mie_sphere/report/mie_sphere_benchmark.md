# Homogeneous-sphere Mie benchmark

This run compares an exact Mie-series calculation with electric-dipole
polarizability approximations for a homogeneous isotropic sphere. It is a
numerical code benchmark, not evidence about any particular material.

## Input and settings

- Input: Bundled synthetic teaching input; not experimental or CFO material data.
- Input SHA-256: `e0d22f915a71c927b248f085083eead4a5b79364ea3880033521f7d189ed6396`
- Wavelength range: 200–800 nm
- Primary radius: 4 nm; scan radii: 2, 3, 4, 5, 6, 8, 10 nm
- Medium refractive index: 1.33
- Convention: exp(-i omega t), epsilon = eps1 + i eps2 with eps2 >= 0
- Cross-section unit: nm^2

## Primary-radius summary

- Maximum size parameter: 0.16713
- Peak extinction: 8.912e+00 nm^2 at 280.0 nm
- Peak absorption: 8.906e+00 nm^2 at 280.0 nm
- Maximum scattering/extinction ratio in the signal region: 0.0096733
- Maximum relative extinction error versus the exact result: Rayleigh 2.128e-02; Kuwata 3.339e-03; electric-dipole Mie 7.131e-03

The signal region is defined as extinction at least 1% of the maximum for each
radius. The synthetic default table has positive loss throughout its tabulated
range; near-zero cross sections can still show floating-point roundoff.

## Numerical checks

- Overall validation: **PASS**
- Pass/fail checks: input_refractive_index_consistency=True, coefficient_sum_matches_library_total_in_signal_region=True, energy_identity=True, passive_cross_sections_within_numerical_floor=True, multipole_lmax_3_converged_for_scan=True, existing_cdm_mie_matches_exact_E1_in_signal_region=True
- Maximum relative energy-balance residual: 2.202e-16
- Maximum truncation error at multipole order l <= 3: extinction 2.821e-07; scattering 4.569e-14
- Electric-dipole Mie polarizability versus exact E1 extinction error: 1.208e-10

This calculation assumes a homogeneous isotropic sphere in a uniform,
non-absorbing medium. It does not include particle interactions, substrates,
anisotropy, magneto-optical response, or a real material data set.
