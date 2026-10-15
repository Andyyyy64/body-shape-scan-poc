# Soma #52: frozen real-video review

This workflow produces one adoption decision for [Soma #52](https://github.com/Andyyyy64/soma/issues/52).
It does not search for candidates or attempt to make a failing method pass.
The ordinary capture store and default v1 numerical processing remain unchanged.

The two declared methods are `fixed_pose_silhouette_fit.v1` with its existing
16-frame runtime and the October 3 selected 48-frame candidate: abdomen weight 8,
shape weight 0.01, common skeleton, normalized observations, anchor weight 1,
and the original boundary-detail/rotation/mask-support QC. The latter is labelled
**診断用** and requires `--research-options ... --diagnostic-real` to run on real video.
Only the synthetic renderer's camera intrinsics are removed from that preset;
the existing real-video camera inference remains in use. Quadratic6 with interval
likelihood is archived under `research/` and is not a third acceptance method.

## Acceptance criteria, verbatim

- 感度: 変化ペアの差が既知変化と同じ向きで、無変化ペアの最大差より明確に大きい（無変化最大差の3倍以上を目安）。固定形状対照は不合格。
- 距離: 距離間の差が1%以内、または1%を超える距離を拒否/補正する規則が根拠付きで決まる。
- 失敗・除外・撮り直しを隠さず記録する。
- 判定: 両方合格 → #RM-1401 と #RM-1402 を進める。感度不合格 → 採用見送りを記録して停止。

The old synthetic ±10, gain 0.8–1.2 and all-375-pair gates are references only.
They are not used for the real-video decision.

## Prepare and record

Keep all originals, model outputs, QC and logs outside Git, under
`~/body-scan-private/analysis/`. Use a fresh review directory:

```sh
python -m body_scan.issue52 prepare \
  --runtime-config "$HOME/body-scan-private/wsl-cuda-runtime/config.json" \
  --out "$HOME/body-scan-private/analysis/issue52-real-review" \
  --budget-seconds 12600
python -m body_scan serve \
  --data-root "$HOME/body-scan-private/analysis/issue52-real-review/captures" \
  --runtime-config "$HOME/body-scan-private/analysis/issue52-real-review/v1.json" \
  --port 60755
```

The budget must be the remainder of the user's four-hour work allowance after
preparation, at most 14,400 seconds. The example reserves 30 minutes for preparation.
The frozen plan caps subsequent processing to this value; individual workers use
the remaining deadline and are terminated when it expires.

Open the loopback URL in the Windows browser. This prepared store is capture-only:
the ordinary UI's reconstruction and synthetic-demo actions are unavailable here.
The browser requests **1920×1080**. Actual negotiated dimensions are shown and
saved, so a camera that provides a smaller image is not misreported as full HD.
If a worker exceeds **8192 MiB RSS**, retain the original, record the reason and
request **1280×720** on the next camera preparation. Low system-available memory
alone does not trigger that resolution change. Reopen the camera to apply it;
there is no hidden recompression, processing retry or second decision.

Select the recording slot before each independent recording:

| Slots | Recordings | Conditions |
|---|---:|---|
| `normal_1` through `normal_3` | 3 | Approximately 2 m, no known change |
| `changed_1` through `changed_3` | 3 | Same distance, known abdomen circumference increase |
| `distance_1.5_1`, `distance_1.5_2` | 2 | Approximately 1.5 m, no wrap |
| `distance_2.0_1`, `distance_2.0_2` | 2 | Approximately 2.0 m, no wrap |
| `distance_2.5_1`, `distance_2.5_2` | 2 | Approximately 2.5 m, no wrap |

For every recording, relax with arms down and keep the head through waist in
frame. Rotate through a full turn; direction and speed are free. Keep clothing,
camera settings, posture and breathing as consistent as practical; document
departures rather than deleting them. Record the sensitivity six within one hour.
Mark the three physical camera distances before recording and document how the
distances were established. Begin with the normal group; both methods establish
their separate, shared skeleton profiles from the same first original.

Document failed camera or upload attempts that produced no saved video in the
operator notes as well; the automatic inventory can only enumerate saved files.

The known change should be approximately +20 mm. The user supplies its direction,
estimated magnitude and how it was established. A tape measure, string measured
against A4 paper, or thickness `t` with the approximation `2*pi*t` is acceptable
for this stated experiment. It is not a calibration of model units. Uploaded
files additionally require confirmation of their actual recording times: upload
timestamps alone do not establish the one-hour condition.

## Freeze inputs, then process once

```sh
python -m body_scan.issue52 inventory \
  --review "$HOME/body-scan-private/analysis/issue52-real-review"
python -m body_scan.issue52 bind \
  --review "$HOME/body-scan-private/analysis/issue52-real-review" \
  --known-change-mm 20 \
  --notes 'Measured increase approximately 20 mm; six sensitivity recordings within one hour; distances marked; no retakes.'
python -m body_scan.issue52 process \
  --review "$HOME/body-scan-private/analysis/issue52-real-review"
```

Replace the example change and notes with the actual observations. Duplicate
slots require explicit `--select normal_1=SESSION_ID` selections before processing.
Every capture, including unselected retakes and unlabelled uploads, is preserved
in the ledger. Missing slots, reused IDs, changed originals or a changed capture
inventory are refused. Source and configurations are hashed at preparation;
input selection is immutable after binding. Identical uploaded videos cannot fill
different independent recording slots. Processing has an exclusive, durable
one-shot marker, including after a crash. Do not remove that marker or silently
start a new review after results.

The same originals go to two separate method stores, with no profile leakage.
All 66 pairs are recorded using `model_abdomen_torso32.v1`. Before any results,
the conservative numerical interpretation is fixed as follows:

- Null variation: all six pairs within the normal and within the changed groups.
- Sensitivity: all nine normal-to-changed differences have the supplied direction,
  are strictly positive in that direction, and each is at least three times the
  largest null difference. Strict positivity makes a constant mesh fail even
  when null variation is zero. The fixed-shape control uses the first real
  measurement for every slot and is scored by the same rule.
- Distance: all twelve cross-distance pairs use the symmetric percentage
  `200*abs(a-b)/(a+b)` and must be at most 1%. Within-distance and other pairs are
  retained as references. If this fails, the report proposes a physical 2 m
  restriction only if the 2 m repeats themselves meet 1%; otherwise it proposes
  rejection of all tested distances. This is a proposal, not an implemented or
  validated capture guard. Adoption stays deferred until the operational rule
  is explicitly decided; no failing pairs are discarded or rescored.

Failures, frame exclusions and retakes are retained. A failed worker or timeout
stops processing and reports an incomplete decision. Completed methods are
compared once, with no fitting changes, gain rescaling or synthetic exploration.
`review.md` gives the one-page decision; `issue52-summary-draft.md` is the same
summary for the user to post. Detailed inputs and mesh remain private. Neither
passing the diagnostic candidate nor a proposed distance rule changes the v1
default or authorizes Soma integration, a push, or an external post.

Preparation stops when recording is needed. After the one-time result, stop and
report adoption, deferment or options; improvement failure is a valid stop reason.
