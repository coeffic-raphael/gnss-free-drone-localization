# Real-Time Algorithm — Step by Step

This document walks through `scripts/run_satellite_first_hybrid.sh`, the retained real-time pipeline, one frame at a time. It is a visual companion to `docs/final_report.md` section 3 (which has the full rationale and ablations) — here the goal is just to make the control flow obvious.

Everything below runs **once per incoming frame**, in strict temporal order, using only:
- the current frame's image,
- this script's own last position estimate (`state_lat`, `state_lon`),
- a precomputed reference-pool descriptor cache and the satellite tile map.

The query flight's true GPS is **never** read inside this loop. It is only read afterward, once per frame, purely to compute the error metric for evaluation.

---

## High-level flow

```
                              ┌─────────────────────────┐
                              │   New frame arrives       │
                              └────────────┬─────────────┘
                                           │
                                           ▼
                          ┌───────────────────────────────┐
                          │ have_fix == True  AND          │
                          │ altitude >= MIN_ALT_M ?         │
                          └───────────────┬─────────────────┘
                         NO  (bootstrap or too low)         YES
                          │                                  │
                          ▼                                  ▼
              ┌───────────────────────┐         ┌─────────────────────────────┐
              │  skip satellite stage  │         │   STAGE 1 — Satellite match   │
              │  sat_status =          │         │   (see detail below)          │
              │  "no_position_yet"     │         └───────────────┬───────────────┘
              │  or "low_alt"          │                         │
              └───────────┬────────────┘                         ▼
                          │                          ┌───────────────────────┐
                          │                          │ sat_status == "ok" ?    │
                          │                          └───────────┬─────────────┘
                          │                       YES             │  NO
                          │                        │              │
                          │                        ▼              │
                          │              ┌───────────────────┐    │
                          │              │ source = "SAT"      │    │
                          │              │ raw fix = sat point │    │
                          │              └─────────┬───────────┘    │
                          │                        │                │
                          └────────────────────────┼────────────────┘
                                                    │
                                     (no SAT fix obtained)
                                                    │
                                                    ▼
                                  ┌─────────────────────────────────┐
                                  │   STAGE 2 — VPR fallback           │
                                  │   (see detail below)                │
                                  └────────────────┬─────────────────────┘
                                                    │
                                                    ▼
                                   ┌───────────────────────────────┐
                                   │ inliers >= 100 AND ratio >= 0.7? │
                                   └───────────────┬─────────────────┘
                                  YES                              NO
                                   │                                │
                                   ▼                                ▼
                       ┌───────────────────────┐        ┌───────────────────┐
                       │ source = "VPR_FALLBACK" │        │ source = "NO_FIX"  │
                       │ raw fix = matched        │        │ raw fix = None     │
                       │ reference frame's GT      │        └─────────┬───────────┘
                       │ ground point               │                  │
                       └────────────┬─────────────┘                  │
                                   │                                  │
                                   └───────────────┬──────────────────┘
                                                    │
                                                    ▼
                              ┌─────────────────────────────────────┐
                              │   STAGE 3 — Causal gap-fill            │
                              │   + causal Gaussian smoothing           │
                              │   (see detail below)                     │
                              └───────────────────┬───────────────────────┘
                                                    │
                                                    ▼
                              ┌─────────────────────────────────────┐
                              │  state_lat / state_lon updated         │
                              │  → centres NEXT frame's satellite       │
                              │     search                               │
                              │  → written to CSV (this frame's output) │
                              └───────────────────────────────────────────┘
```

---

## Stage 0 — Bootstrap (no satellite search possible yet)

```
have_fix == False
        │
        ▼
sat_status = "no_position_yet"   ──►  jump straight to Stage 2 (VPR fallback)
                                       searched against the WHOLE reference pool
                                       (no geographic restriction — there is no
                                       position yet to restrict around)
```

Before the very first accepted fix, there is nothing causal to centre a satellite tile search on, so satellite matching is skipped entirely. Every frame goes through VPR retrieval (Stage 2) until one of them finally clears the acceptance gate. This is the only place in the pipeline where a fix is found "from nothing."

---

## Stage 1 — Satellite tile matching (detail)

Only attempted once `have_fix == True` and altitude ≥ 20 m.

The "heading" used by the warp below is the one disclosed exception to "no query GNSS": neither drone records gimbal yaw, so heading is estimated from the query flight's own past GPS positions (backward-only window, never a future frame — see `docs/final_report.md` §3 and §"Limitations"). Altitude (`alt`) comes from the barometer, not GPS, so it isn't an exception.

```
frame ──► IPM warp (bird's-eye view using altitude + heading)
                │
                ▼
   build 3×3 satellite mosaic CENTRED ON state_lat/state_lon
   (the pipeline's own last estimate — never the true GPS)
                │
                ▼
   SuperPoint features (warped frame)  ──┐
   SuperPoint features (satellite tile) ──┤──► LightGlue match
                                            │
                                            ▼
                              RANSAC homography (≥8 inliers required)
                                            │
                         ┌──────────────────┼───────────────────┐
                         ▼                  ▼                   ▼
                  no_tile / few_matches   ransac_fail        ok
                  (not enough overlap     (homography        │
                   or matches)             rejected)          ▼
                         │                  │          project IPM centre
                         └──────────────────┘          through homography
                                │                       → satellite pixel
                                ▼                       → lat/lon via tile
                          sat_status != "ok"              georeferencing
                          → fall through to Stage 2       sat_status = "ok"
```

Cost: ~0.2–0.4 s/frame. Tried first because it's cheap and needs no descriptor database search.

---

## Stage 2 — VPR fallback (detail)

Triggered whenever Stage 1 didn't produce `sat_status == "ok"` (including during bootstrap).

```
frame ──► DINOv2 descriptor, computed LIVE right now
          (not a precomputed batch — this is the honest streaming cost)
                │
                ▼
   cosine similarity vs. cached reference-pool descriptors
   (v11 / v12 / v14, or whichever REFERENCES were passed)
                │
                ▼
   take top-K candidates (default K=3)
                │
                ▼
   for each candidate: SuperPoint + LightGlue match
                        ──► RANSAC homography ──► inlier count, inlier ratio
                │
                ▼
   keep the candidate with the MOST inliers
                │
                ▼
   inliers >= 100  AND  ratio >= 0.70 ?
        │YES                              │NO
        ▼                                  ▼
  source = "VPR_FALLBACK"           source = "NO_FIX"
  raw fix = that reference           raw fix = None
  frame's own ground-truth
  ground point (precomputed
  during reference preprocessing)
```

Cost: ~0.8–1.5 s/frame (dominated by the live DINOv2 extraction + several LightGlue calls).

---

## Stage 3 — Causal gap-fill + causal smoothing (detail)

Runs every frame, regardless of which stage produced (or failed to produce) a raw fix.

```
raw fix this frame?
   │YES                                    │NO
   ▼                                        ▼
filled = raw fix                  has any fix ever been
                                   produced before (have_fix)?
                                        │YES            │NO
                                        ▼                ▼
                              filled = LAST          filled = None
                              filled position          (nothing to
                              (carried forward,         carry forward —
                              never a future             frame stays
                              frame's value)             unresolved)
   │                                    │                │
   └──────────────┬─────────────────────┘                │
                  ▼                                       │
       append `filled` to history                         │
                  │                                       │
                  ▼                                       │
       Gaussian-weighted average of the                   │
       current entry + up to SMOOTH_HALF_WINDOW*2          │
       PAST entries only (no look-ahead)                  │
                  │                                       │
                  ▼                                       │
       smoothed_lat, smoothed_lon                          │
       have_fix = True                                     │
       state_lat, state_lon = smoothed position             │
       (used to centre NEXT frame's satellite search)        │
                  │                                       │
                  └─────────────────┬─────────────────────┘
                                     ▼
                     write this frame's row to CSV:
                     sat_status, source, raw fix, raw error,
                     smoothed fix, smoothed error, timing
```

`SMOOTH_HALF_WINDOW` is a fixed parameter chosen ahead of time from offline tuning (not swept against the run being evaluated) — see `docs/final_report.md` sections 3.1/3.2 for the chosen values per video.

---

## What never happens (by design)

- The query flight's true GPS *position* is never read inside Stages 0-3 — only `ground_latitude`/`ground_longitude` is read once, after the frame's decision is already final, purely to compute `raw_error_m` / `smoothed_error_m` for the CSV.
- The satellite search is never centred on the true GPS — only on `state_lat`/`state_lon`, this script's own latest causal output.
- Gap-filling never uses a future frame's fix — only the last *past* filled value.
- Smoothing never uses a future frame's fix — only the current and past entries in the window.
- Nothing in Stages 0-3 ever reads a *future* frame's data — including `heading_deg` (Stage 1), which is estimated from a backward-only window over the query's own GPS trajectory.

**One disclosed exception:** `heading_deg`, used by Stage 1's IPM warp, is not from the query flight's true GPS *position* directly, but it is derived from the query flight's own *past* GPS trajectory (no gimbal yaw sensor exists on either drone). It is causal (backward-only, fixed during a code audit that found an earlier centered/look-ahead version), but it is still query GNSS in a narrow sense. See `docs/final_report.md` §3 and "Limitations" for the full discussion, including how an earlier version of the script violated causality more severely (centred the satellite search on the true GPS) and how an earlier heading estimator looked ahead.
