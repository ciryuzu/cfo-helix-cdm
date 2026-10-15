# Evidence and limitations

This document defines how claims in the public repository should be read. A
successful script run means that a stated numerical task was executed and
checked. It does not automatically promote a reduced model, an inverse fit, or
an idealized input into a material measurement.

## Evidence labels

| Label | Meaning | What it permits | What it does not permit |
|---|---|---|---|
| **Validated** | A calculation ran with recorded inputs and passed the checks defined for that calculation. | A claim about the output of that model and test. | A claim that the model is the unique physical explanation or that it reproduces an experiment independently. |
| **Diagnostic** | A toy model, inverse probe, coarse-grained response, or sensitivity calculation was run to assess scale or plausibility. | A statement about a possible range, trend, or missing-response demand. | Calling a fitted parameter a material constant or calling a proxy a microscopic observable. |
| **Conditional** | An output depends on fixed geometry, normalization, regularization, material interpolation, or another explicit convention. | Reproduction of the conditional result when the same convention is used. | Treating the result as a unique decomposition, concentration, or absolute calibration. |
| **Pending** | A test is planned, incomplete, or blocked by an environment or missing input. | A documented next step and acceptance criterion. | Describing the planned outcome as evidence. |
| **Superseded** | An earlier result was affected by an identified input or interpretation issue and is retained only for history. | A reproducibility note explaining the correction. | Using the old result as current evidence. |

## Claim register

The CFO-specific rows below summarize earlier project runs. Their CFO inputs
and experimental targets are intentionally absent from this public seed
package, so the listed results cannot be rerun from the bundled files. The Mie
benchmark shipped here uses synthetic input and supports only the code checks
reported in its own output.

| Claim | Status | Evidence boundary |
|---|---|---|
| The Python 3N electric coupled-dipole baseline is runnable and reproduces the intended mirror-sign behavior. | Validated | Supports the implementation and the stated idealized geometry. It does not validate CFO material realism or an experimental magnitude. |
| A scalar CFO sphere was predominantly electric-dipole in an earlier project Mie benchmark. | Historical validation; input excluded | This public package does not reproduce that CFO-specific run. Its bundled synthetic-sphere benchmark says nothing about `epsilon_xy`, magnetic domains, or a magnetized anisotropic particle. |
| A pure electric helix can produce structural CD and an R/S sign reversal. | Validated in the stated model | The public synthetic sanity check confirms mirror-sign reversal for its input. Earlier CFO calculations also reported the sign change, with a modeled scale below the experimental target; those material-specific inputs and outputs are not included here. |
| Changing idealized geometry or fixed incidence alone closes the scale gap. | Not supported | Those variations alter shape and amplitude but did not supply the missing scale in the tested cases. A fixed-incidence calculation is not a multilayer film calculation. |
| A supplied `+M/-M` tensor can create a magnetization-odd response in tensor CDM. | Diagnostic | The result is a reduced Rayleigh-tensor calculation using supplied tensor data. It requires a tensor single-particle full-wave benchmark and a complete parity decomposition before mechanism language is used. |
| The helix has a collective magnetic-like response. | Diagnostic | A moment or bianisotropy proxy can be useful for mode interpretation. Origin dependence and omitted higher multipoles prevent it from being an intrinsic single-particle magnetic or magnetoelectric constant. |
| An inverse coupling scale is a CFO `alpha_EH` or `alpha_HE`. | Not supported | The inverse answers how large an effective channel would need to be under the chosen model. It is not derived from an independently validated local response. |
| A CD-only inverse curve is the unique CFO dielectric function. | Not supported | One CD observable per wavelength cannot by itself determine both complex dielectric components and sample-scale factors. Separate fits can absorb missing physics. |
| The corrected publication-oriented spectral decomposition is an independent composition measurement. | Not supported | It is a constrained, regularized attribution using fixed geometry and reference material spectra. Its coefficients are not mass, volume, coverage, or polymer fractions. |
| The PS-b-PAA effective input is a directly measured optical function of the specific sample. | Not supported | It is constructed from literature constituent inputs and a stated mixing approximation; short-wave extrapolation and source mismatch remain part of its uncertainty. |
| The mixed PS-b-PAA input is fully Kramers–Kronig validated. | Not supported | Finite-band checks of selected source curves are diagnostics, not a complete causality proof of the stitched effective function. |
| A full-wave NGSolve calculation has already established the magnetic mechanism. | Pending | The planned route starts with a scalar benchmark, then tensor particle, dimer, and small helix. Environment setup and convergence evidence must accompany any future claim. |

## Layered limitations

### Material layer

Scalar CFO inputs may come from reconstructed or digitized spectra and are
therefore sensitive to finite spectral windows, baseline handling, interpolation,
and tail assumptions. The `+M/-M` tensor input represents an idealized bulk
material response; raw electronic-structure files and nanoparticle-specific
surface, defect, strain, and domain information are not part of the public
model by default. A nonzero off-diagonal dielectric term first modifies the
electric response tensor; it is not automatically an electric–magnetic
cross-polarizability.

### Finite-particle layer

Rayleigh/Kuwata and electric-dipole Mie polarizabilities are reduced models.
They do not replace a finite-particle Maxwell calculation when shape, tensor
orientation, internal fields, or higher multipoles matter. The bundled Mie
benchmark constrains only its synthetic scalar-sphere case. It does not
reproduce the earlier CFO-specific benchmark summarized in the claim register.

### Cluster and propagation layer

The baseline uses a specified Green tensor, geometry, and incidence convention.
Substrate reflection, multilayer propagation, film morphology, particle-size
and shape distributions, and orientation averaging are not automatically
included. A collective moment built from electric dipoles is not the same object
as an intrinsic magnetic dipole inside each particle.

### Observable layer

The model must distinguish extinction, absorption, and scattering. The
experimental instrument may measure a quantity affected by collection angle,
depolarization, film thickness, optical path, or normalization. A plotted CD
or `g_CD` comparison must state which modeled observable and which scale
convention were used.

### Inverse-problem layer

Regularization, non-negativity, smoothness, column scaling, a mirror relation,
and a fixed overall scale can select a stable numerical representative. They
do not create information that was absent from the data. A small training
residual is compatible with non-unique physical explanations.

The direct-CD inverse and the PS-b-PAA attribution must therefore be described
as conditional inverse analyses. The latter is especially sensitive to the
effective organic input, the short-wave extrapolation, and the assumption that
two uniform reference materials stand in for a structured composite.

## Required symmetry checks for a mechanism claim

An eventual mechanism study should report the four combinations of geometry
handedness and magnetization, not infer a mechanism from one spectrum:

```text
(R,+M), (R,-M), (S,+M), (S,-M)
```

The structural contribution should be even under magnetization reversal and
odd under mirror reversal; MCD should be odd under magnetization reversal; a
mixed structural–magnetic term should change sign under either operation. The
exact observable and normalization must be stated before those parities are
interpreted.

## Public-claim rules

* Keep the words “validated,” “diagnostic,” “conditional,” and “pending” next
  to the relevant figure, table, or result file.
* Never call an empirical `eta`, `beta`, branch-scale factor, or regularized
  coefficient a measured material constant.
* Never infer that scalar Mie dominance proves magnetic effects are negligible.
* Never infer that a successful CD fit proves a unique dielectric function or
  a composition fraction.
* Treat old outputs affected by the interpolation correction as historical
  comparisons only; regenerate current figures from the corrected input chain.
* State whether a result uses public-source data, a supplied tensor, a
  synthetic example, or restricted experimental data.
* Treat plot-panel exports and coordinate-rendered geometry snapshots as
  presentation derivatives. They do not upgrade the evidence level of the
  underlying CDM or inverse calculation.

## Open acceptance tests

Before upgrading any magnetic-mechanism statement, the repository should have
machine-readable evidence for:

1. scalar full-wave versus Mie cross sections and energy balance;
2. tensor single-particle parity and passivity checks;
3. mesh, exterior-domain, and convergence studies;
4. dimer and small-helix agreement between full-wave and reduced models;
5. R/S × `+M/-M` decomposition with a declared observable;
6. a clear treatment of substrate, disorder, orientation, and detector effects
   whenever they are needed to compare with the experiment.
