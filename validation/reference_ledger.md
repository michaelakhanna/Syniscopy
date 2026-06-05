# Validation Reference Ledger

This ledger is the source-of-truth for what each validation check is allowed to
claim. A check that does not call an external reference implementation must be
anchored to an explicit equation, named model, or documented analytic limit.

## Status Labels

- `external_reference`: compared with an independently implemented package.
- `analytic_literature`: compared with a literature equation or closed-form
  physical limit.
- `internal_contract`: validates schema, serialization, determinism, or an
  internal invariant; not a physics reference claim.
- `fixture_required`: a credible external reference exists, but a pinned fixture
  is not yet part of the repository.

## Physics / Model Checks

| Surface | Validator | Status | Equation or reference used | Tolerance / claim boundary |
|---|---|---|---|---|
| Mie angular amplitudes | `validation/mie_crosscheck.py`, `run_validation_suite.py --profile release` | `external_reference` | `miepython.S1_S2(..., norm="wiscombe")`, which documents Mie S1/S2 normalization choices; Mie theory as in Bohren-Huffman / Wiscombe. | Angular dependence must match up to the documented normalization; this validates S1/S2, not the full PSF renderer. |
| Thin-film reflection | `validation/thinfilm_electron_validation.py`, `run_validation_suite.py --profile release` | `external_reference` + `analytic_literature` | Normal-incidence Fresnel coefficient `r=(n0-ns)/(n0+ns)` and `tmm.coh_tmm` coherent transfer-matrix reference. | No-layer complex Fresnel must match directly; layered reflectance `|r|^2` must match `tmm`. Complex amplitude phase convention is recorded separately. |
| Translational Brownian motion | `validation/trajectory_validation.py`, `run_validation_suite.py --profile fast` | `analytic_literature` | Stokes-Einstein-Sutherland `D=kBT/(6*pi*eta*r)=kBT/(3*pi*eta*d)` and Brownian MSD `E[Delta x^2]=2D dt` per axis. | Direct step standard deviation and seeded determinism pass; long-lag MSD is reported with stochastic tolerance. |
| Rotational diffusion | `run_validation_suite.py --profile fast` | `analytic_literature` | Stokes-Einstein-Debye rotational diffusion for a sphere: `D_r=kBT/(8*pi*eta*r^3)`, step scale `sqrt(2D_r dt)`. | Per-particle step scale must match the formula; generated matrices must remain SO(3) and deterministic under seed. |
| Camera shot noise | `validation/noise_validation.py`, `run_validation_suite.py --profile fast` | `analytic_literature` | Poisson counting identity: variance equals mean electron/photon count; counts-domain gain conversion. | Closed-form shot-noise std and sampled variance/mean slopes must pass. |
| Lateral Fisher / CRLB | `validation/fisher_validation.py`, `run_validation_suite.py --profile fast` | `analytic_literature` | Fisher information `F=sum((dC/dtheta_i)(dC/dtheta_j)/variance)` and closed-form Gaussian localization scaling, consistent with Thompson-Larson-Webb and Ober-Ram-Ward localization-limit literature. | Validates estimator algebra on analytic images, not that a rendered contrast image is physically correct. |
| Electron wavelength / Scherzer | `validation/thinfilm_electron_validation.py`, `run_validation_suite.py --profile fast` | `analytic_literature` | Relativistic electron wavelength published tables; Scherzer defocus/resolution relations from TEM CTF theory and Kirkland-style electron microscopy references. | Wavelength table residuals and Scherzer formula residuals must pass. |
| Scalar PSF Airy limit | `validation/psf_airy_validation.py`, `run_validation_suite.py --profile fast` | `analytic_literature` | Fourier-optics Airy first dark ring `r=0.61 lambda/NA` in the low-NA unaberrated limit. | Validates the scalar pupil-to-image transform in the Airy-like limit; not a full high-NA/vectorial claim. |
| Vectorial Debye PSF | `validation/vectorial_debye_validation.py`, `run_validation_suite.py --profile fast` | `analytic_literature` | Richards-Wolf / Debye-Wolf vectorial focusing: polarization rotation covariance, zero cross/longitudinal field at the optical axis for x-polarized focus by symmetry, and low-NA reduction to the scalar Airy limit. | Validates structural vectorial-Debye consequences. A high-NA numeric match to PyFocus or another Richards-Wolf implementation would be a stronger optional fixture. |
| iSCAT / interferometric contrast | `validation/iscat_contrast_validation.py`, `run_validation_suite.py --profile fast` | `analytic_literature` | Interference identity `|E_r+E_s|^2-|E_r|^2 = 2 Re(conj(E_r) E_s)+|E_s|^2`, used by iSCAT literature. | Validates contrast equation only, not a full rendered image. |
| Coherent bright-field / COBRI | `validation/flagship_validation.py`, `run_validation_suite.py --profile fast` | `analytic_literature` | Same coherent interference identity with the incident/background field as reference. | Validates bright-field contrast algebra. |
| DPC | `validation/flagship_validation.py`, `run_validation_suite.py --profile fast` | `analytic_literature` | Differential phase contrast from asymmetric illumination; Tian-Waller LED-array DPC and half-plane DPC operator conventions. | Validates the phase-gradient and sign/operator behavior used by the simplified model. |
| Dark-field | `validation/modality_equation_validation.py`, `run_validation_suite.py --profile fast` | `analytic_literature` | Reference-free dark-field intensity proportional to `|E_sca|^2` under the selected gain. | Validates equation and reference independence. |
| QPI | `validation/modality_equation_validation.py`, `run_validation_suite.py --profile fast` | `analytic_literature` | Quantitative phase `phi=arg(1+E_s/E_ref)` and weak-object small-signal `Im(E_s/E_ref)`. | Validates phase output convention. |
| RICM | `validation/modality_equation_validation.py`, `run_validation_suite.py --profile fast` | `analytic_literature` | Two-reflection interference using Fresnel substrate/reference coefficient and particle reflection coefficient. | Validates the implemented interference identity and Fresnel source. |
| Off-axis holography | `validation/modality_equation_validation.py`, `run_validation_suite.py --profile fast` | `analytic_literature` | Holographic tilted-reference intensity `|E_ref exp(iK.r)+E_s|^2-|E_ref|^2`. | Validates contrast identity and the configured carrier period on the model canvas. |
| Fluorescence widefield | `validation/modality_equation_validation.py`, `run_validation_suite.py --profile fast` | `analytic_literature` | Incoherent Gaussian emission PSF proxy and linear superposition of emitter density. | Validates the parametric fluorescence backend, not the vectorial photophysics backend. |
| TIRF | `validation/modality_equation_validation.py`, `run_validation_suite.py --profile fast` | `analytic_literature` | Evanescent penetration depth `d=lambda/(4*pi*sqrt((n1 sin theta)^2-n2^2))`, Axelrod-style TIRF. | Validates the angle-derived penetration-depth branch and below-critical rejection. |
| Zernike phase contrast | `validation/modality_equation_validation.py`, `run_validation_suite.py --profile fast` | `analytic_literature` | Zernike phase-ring concept: pi/2 ring converts weak phase variation into intensity modulation. | Validates qualitative/limit behavior, not a full quantitative Zernike microscope. |
| TEM CTF proxy | `validation/modality_equation_validation.py`, `run_validation_suite.py --profile fast` | `analytic_literature` | Weak-phase CTF convention used by the backend, `ctf=2 sin(chi)` with `objective_transfer=exp(-i chi)`, and Scherzer first-zero scale. | Validates the internal sign convention and Scherzer scale. |
| TEM physical multislice | `validation/tem_multislice_validation.py`, `run_validation_suite.py --profile release --require-external` | `analytic_literature` + `external_reference_fixture` | Cowley-Moodie / Kirkland multislice anchors: free-space unit intensity, slab phase `sigma V t`, weak-phase CTF limit; generated abTEM/ASE fixture using the same potential slices, voltage, sampling, slice thickness, defocus, and Cs. | Internal physics anchors pass. The generated abTEM fixture comparison passes with tolerance-based exit-wave and image residuals. |
| SEM physical transport | `validation/sem_transport_validation.py`, `run_validation_suite.py --profile sem` | `analytic_literature` + `fixture_required` | Screened Rutherford / Joy-Luo stopping, Kanaya-Okayama range, Tabata/Reuter-style backscatter coefficient anchors, Browning Mott surrogate option. | Fast smoke checks monotonic finite behavior. Full SEM reference sweep is long and should be run separately; backend remains `diagnostic_only` unless that sweep passes. |
| Köhler partial coherence | `validation/modality_equation_validation.py`, `run_validation_suite.py --profile fast` | `analytic_literature` | Abbe/Hopkins sum of coherent systems; in the single on-axis source limit, objective-bandlimited fields reduce to coherent bright-field composition. | Validates the documented limit only. Arbitrary unfiltered fields are not expected to match COBRI. |

## Internal / Non-Physics Checks

| Surface | Validator | Status | Claim boundary |
|---|---|---|---|
| Material single source | `run_validation_suite.py --profile fast` | `internal_contract` | Top-level `materials.py` is absent; core imports resolve material optical/electron constants through `material_optical_catalog.py`; catalog lookups are finite over sampled wavelengths. |
| Manifests and packets | `run_validation_suite.py --profile fast` | `internal_contract` | Video/dataset manifests serialize with `allow_nan=False`; counterfactual packets save/load/validate without key-map fallback. |
| Runtime imports/defaults | `run_validation_suite.py --profile fast` | `internal_contract` | Core modules import and current real-physics defaults are set. |

## References / Links Used

- Richards and Wolf, "Electromagnetic diffraction in optical systems, II" DOI `10.1098/rspa.1959.0200`; bibliographic record: https://colab.ws/articles/10.1098/rspa.1959.0200
- `miepython` S1/S2 normalization documentation: https://miepython.readthedocs.io/en/2.5.5/api/miepython.miepython.mie_S1_S2.html
- `tmm` coherent transfer-matrix package overview: https://deepwiki.com/sbyrnes321/tmm/1-overview
- Stokes-Einstein-Sutherland equation summary: https://chem.libretexts.org/Courses/University_of_Wisconsin_Oshkosh/Chem_371%3A_P-Chem_2_to_Folow_Combined_Biophysical_and_P-Chem_1_%28Gutow%29/05%3A_Moving_Molecules_and_Chemical_Kinetics/5.05%3A_Stokes-Einstein-Sutherland_Equation
- Poisson shot-noise detector discussion: https://camera.hamamatsu.com/us/en/learn/technical_information/thechnical_guide/photon_shot_noise.html
- iSCAT interference contrast review: https://pmc.ncbi.nlm.nih.gov/articles/PMC6750867/
- Tian and Waller DPC paper: https://opg.optica.org/oe/fulltext.cfm?uri=oe-23-9-11394
- Zernike phase-contrast weak-phase/ring discussion: https://pmc.ncbi.nlm.nih.gov/articles/PMC3085486/
- Kanaya-Okayama / electron-range context: https://www.globalsino.com/EM/page4967.html
- NIST SEM dimensional-metrology review including Kanaya-Okayama expression context: https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=910884
- abTEM PlaneWave API: https://abtem.readthedocs.io/en/latest/reference/api/_autosummary/abtem.waves.PlaneWave.html
- abTEM PotentialArray API: https://abtem.readthedocs.io/en/main/reference/api/_autosummary/abtem.potentials.iam.PotentialArray.html
