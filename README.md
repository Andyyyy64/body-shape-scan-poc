# Body Shape Scan PoC

A research project testing whether a fixed laptop RGB camera and a short self-rotation capture can measure longitudinal changes in external body shape. The local operator screen records video, runs SAM 3D Body, generates a canonical mesh, and compares saved scans. Measurement accuracy is not established.

**Primary question:** can independent captures detect a real 1 cm change in a defined abdominal or hip circumference, while keeping false change reports acceptably low?

日本語の詳細な実験範囲は [PoC scope](docs/poc-scope.ja.md) を参照してください。

## Run on an Apple Silicon Mac

The operator screen now supports recording or importing a video, running local SAM inference, viewing a canonical 3D body, and comparing two saved reconstructions.

```bash
uv sync --locked --extra vision --python 3.11
# One-time installation on a fresh machine; requires approved Hugging Face model access and CLI login.
uv run body-scan setup-mac --out "$HOME/body-scan-private/runtime"
# If the runtime is already installed, run only this command:
uv run body-scan serve --data-root "$HOME/body-scan-private/scans" \
  --runtime-config "$HOME/body-scan-private/runtime/config.json"
```

Open the loopback URL printed by `serve`. All videos, meshes, profiles, comparisons, weights and runtime logs stay outside Git. The inference subprocess runs with macOS network access denied. There is no cloud inference or automatic device fallback.

Reconstruction is the **shared-shape silhouette fit** (`fixed_pose_silhouette_fit.v1`): sample frames, select the primary person (small secondary detections are ignored, comparable ones reject the frame), estimate pose and shape per frame with SAM 3D Body, exclude frames that violate the capture protocol (head-to-pelvis inside the image, arms hanging below the elbows), initialise one shape from the per-frame median, then optimise that shared shape and per-frame camera translation against every frame's observed silhouette with the per-frame poses fixed. A capture with fewer than 75% usable frames is refused with the reasons. The first scan establishes shared skeleton scales and fixed torso measurement locations; later scans estimate their own shape. Turning speed and path are free; abdominal tension, arm position and framing are the protocol. Full joint pose/camera fitting remains a separate research gate.

The screen reports **model-space** geometry differences, not proven biological changes. Dates never force differences to zero. Reusing the same video is flagged as a duplicate input, not counted as independent repeatability evidence. [Detailed usage and limitations](docs/runbook.ja.md).

Numerical tools remain available without model weights:

```bash
uv run body-scan demo --out "$HOME/body-scan-private/numeric-01"
uv run body-scan render-demo --out "$HOME/body-scan-private/render-01"
uv run python -m unittest discover -v
```

## Selected scope

- **Primary device:** a laptop's built-in RGB camera. A fixed smartphone is a diagnostic comparator, not a substitute for laptop success.
- **Primary framing:** head, shoulders and pelvis visible. Feet are not required; unobserved legs are not measured. Abdomen-only crops are outside the initial capture protocol.
- **Primary output:** change in external abdominal circumference at a fixed, versioned anatomical plane. Hip circumference is a co-primary output only when its entire measurement region is observable; otherwise report it as unavailable and do not claim hip validation.
- **Secondary outputs:** neighboring cross-section profiles and local surface displacement. Chest measurements and mesh visualization are exploratory.
- **Initial capture comparison:** continuous rotation versus a turn-and-pause protocol; compare laps separately before evaluating averaging.
- **Evidence boundary:** geometry tests, rigid-object experiments, human repeatability, and longitudinal human accuracy are separate gates.

The primary 1 cm target is a design requirement, not a measured result. A predeclared 2 cm tier is reported separately and never presented as evidence for the 1 cm target.

## Candidate methods

| Method | Purpose |
| --- | --- |
| Two-axis ellipse and a separately versioned incumbent | Simple baselines; quantify whether additional complexity helps |
| Multi-angle width integration / perspective-aware torso cross-sections | Test whether full-body reconstruction is necessary |
| Independent SAM 3D Body predictions, then a declared aggregation rule | Baseline for the value of joint fitting |
| SAM initialization + one shared MHR shape per capture | Main 3D challenger; fit to observed silhouettes and keypoints |
| MHR + bounded local shape corrections | Conditional experiment only if standard MHR leaves systematic residuals |
| Constant-shape predictor | Negative control: repeatability without sensitivity must fail |

No candidate automatically replaces another when it fails. Preserve method identity, failure reason, and missing output. Do not treat generated occluded pixels as observed evidence.

## Stage gates

| Gate | Deliverable | Required evidence |
| --- | --- | --- |
| G0 | Reproducible inputs, calibration and private-data boundary | Public-file audit; metric-scale and coordinate round trips; frozen measurement definitions |
| G1 | CPU geometry and statistical validation | Known local changes, null changes, scale/angle errors, independent shape families, power calculation |
| G2 | Laptop capture and segmentation feasibility | Continuous vs paused rotation; manual-mask control; independent repeat scans; no hidden missing regions |
| G3 | SAM/MHR initialization and reconstruction | Pinned model versions; full parameter/mesh round trip; measured runtime and memory |
| G4 | Multi-view fitting comparison | Held-out angular blocks; multiple initializations; common observations and calibration for candidates |
| G5 | Real-world repeatability, sensitivity and longitudinal validation | Blinded repeated references; subject/session separation; failure denominator; uncertainty and detection power |
| G6 | Integration decision and offline output contract | Adopt, narrow scope, reject, or insufficient evidence, supported per device and body part |

GitHub Issues are the TODO and evidence source of truth. Individual issue closure cannot substitute for the final human evidence gate.

## Public and private materials

Publish only source code, public-source citations, generic documentation, and explicitly synthetic fixtures/results. **Do not commit or attach participant-derived data to this repository, Issues, pull requests, releases, Actions artifacts, or logs.**

Keep photographs, video, audio, masks, silhouettes, meshes, keypoints, body measurements, capture dates, device identifiers, consent records, private paths, and identity mappings outside the checkout. A blurred face or a pseudonym does not make body geometry anonymous. Public demonstration fixtures must be synthetic or separately cleared for publication. Aggregate human results require a separate disclosure review; small cohorts can remain identifiable.

Dependencies' code, model weights, and datasets have separate terms. This repository grants no rights to third-party assets. No cloud uploads, paid compute, or additional participant recruitment are implicit in this plan.

## Current evidence

The numerical, privacy, local API and synthetic-image tests are executable. On the development Mac, official SAM inference and the MHR full-parameter round trip were executed. The browser-to-video-to-SAM-to-canonical-mesh-to-comparison path was exercised with a public-reference-image test video. Repeated processing of that identical video produced identical geometry and was explicitly flagged as duplicate input.

A private single-participant short-interval repeatability evaluation was run on the development Mac with held-out captures; its results stay outside this repository pending disclosure review. This is **not** evidence of physical accuracy or of a longitudinal body-change trial, and sensitivity to a known real change is untested. Full multi-view silhouette fitting and confirmatory human validation remain open. GitHub Issues track progress and blockers.

## Primary sources

- [Rotation-based smartphone scan repeatability, 2024](https://pubmed.ncbi.nlm.nih.gov/38454153/): evidence for a class of dedicated systems, not a performance claim for this implementation.
- [Non-rigid smartphone avatar reconstruction, 2024](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2024.1485450/full): capture and pose-standardization context; body-composition prediction is separate from geometry.
- [SAM 3D Body paper](https://arxiv.org/html/2602.15989v1) and [official implementation](https://github.com/facebookresearch/sam-3d-body).
- [MHR official implementation](https://github.com/facebookresearch/MHR).
- [SAM-Body4D paper](https://arxiv.org/html/2512.08406v1): temporal stability does not itself establish multi-angle anthropometric accuracy.
- [OpenCV camera calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html).
- [nvdiffrast](https://nvlabs.github.io/nvdiffrast/) and [trimesh cross-sections](https://trimesh.org/section.html): implementation candidates, not accuracy evidence.
