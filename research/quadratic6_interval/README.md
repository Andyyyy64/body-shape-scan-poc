# Archived quadratic6 + interval likelihood prototype

This is a byte-for-byte archive of the existing 76-module shape-profile CLI
prototype. It preserves the quadratic six-control shape subspace and pixel
interval likelihood, with its original adapters and dependencies. See
[source hashes](SOURCE_SHA256.md). No new estimation method is introduced.

The preserved best configuration is selected in that CLI with
`--scale-camera --shape-profile quadratic --width-likelihood pixel-interval`.
The `quadratic` branch delegates to `quadratic_shape_profile.fit_curve` with
`restricted=True`: six Bernstein controls map into the original ten shape
coefficients. The interval branch uses the existing observation normalizers
through `shape_profile_polish.pixel_interval` and `pixel_interval_fit`.
These are archived settings, not authorization to run another synthetic trial.

The synthetic results are historical reference evidence. The previous synthetic
gain, signed 10 mm and 375-comparison gates are not Issue 52 adoption criteria.
This archive is not selected for the new real-video comparison: that comparison
uses default v1 and the October 3, 48-frame shared-skeleton candidate.

The prototype remains outside the installed `body_scan` package. To inspect its
existing CLI explicitly, from the repository root run:

```bash
PYTHONPATH=research/quadratic6_interval python -P -m body_scan.capture_measurement --help
```

Keep participant videos, observations, meshes, measurements and runtime/model
configuration outside the checkout. This archive is not an accuracy claim or a
default behavior change. Do not resume synthetic candidate exploration.
