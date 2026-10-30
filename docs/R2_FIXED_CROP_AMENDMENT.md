# R2 fixed-crop protocol amendment for review

**Approved by the user on 30 October 2026; implementation and official checks recorded separately below. Supervisor sign-off is unverified.**

## Proposed decision

Study classification of uniformly encoded fixed 256x256 centre patches from the existing real/SD1.5 acquisition, rather than classification after the original variable-size centre-square/downsample pipeline. The measured candidate below gives a concrete basis for review; it is not a declaration that the original acquisition is unbiased.

After reviewing the candidate and this amendment, the user explicitly replied "aproved" to the proposed R2 approval step. This authorizes R2 implementation and the revised Phase 2 audit/splits. It is not represented as independent supervisor sign-off or approval of downstream training/evaluation.

## Evidence available for the decision

The original metadata audit remains blocked at 0.8308/0.9998. Full-source metadata showed insufficient real 512x512 images for R1 exact native matching. One subsequent exploratory treatment used the same 5,000 audited images, fixed 256x256 crop without resampling, and identical quality-96 JPEG encoding for both classes. Its two metadata probe means were both 0.5814. See [actual methods and results](../reports/phase_2_fixed_crop_diagnostic.md).

## Exact treatment if adopted

1. Decode **original acquisition bytes**, verify checksum, apply EXIF orientation and convert to RGB.
2. Reject either dimension smaller than 256. Take the centered 256x256 patch, using floor division for odd offsets. Do not resize, upscale, stretch or crop differently by label.
3. Encode that patch once to JPEG at the fixed recorded quality 96, 4:4:4, non-progressive, no optimization, then decode it.
4. Convert RGB to greyscale L, divide by 255 to finite float64 `[0,1]`. Frequency and pixel baselines consume this same array. The prescribed pixel-baseline reduction to 32x32 remains separate.

This replaces the current crop/downsample treatment and introduces one common JPEG round trip inside the shared preprocessing contract. Give it a new explicit version/config hash. Quality, crop size and all encoder parameters are fixed before creating research splits; they cannot later be tuned against validation/test results.

## Audit change requiring explicit approval

Maintain the original-file audit as immutable evidence of failure. Add a distinct treated-input audit, with all six existing metadata columns and the same two balanced seeded five-fold probes and <=0.60 limit. Both candidate means meet this numerical bar, but original-file dimensions remain predictive.

**R2 changes the gate's domain from the original acquisition geometry to the exact image representation supplied to the detector.** That is a change to the PRD's validity criterion, even though the numeric threshold is unchanged. The team/supervisor must explicitly accept or reject it. Do not describe it as the original preplanned JPEG remedy or erase the raw audit.

Because geometry/format/quality are fixed by the new treatment, their lack of predictiveness is expected by construction. The remaining metadata probe is effectively a check of encoded-size differences. It cannot certify equal original compression history, image content or object scale. Passing this narrower audit supports only a constrained patch experiment; claims about generator-specific artifacts or general authenticity must remain qualified.

The amendment and the history of candidate selection must be included in deviations and the final report. The treatment was proposed after seeing metadata diagnostics. No research test was opened, but that does not make the treatment selection preregistered from the original study's start.

## Implementation contract after approval

- Use a new experiment root and versioned configuration; keep failed original audit and all R1/R2 diagnostic evidence. No reset of prior research records.
- Keep `preprocess_image` as the one detector entry point for extraction and the app. A shared internal canonicalization helper may produce the exact JPEG patch for measurement and preprocessing. This is not a second class-specific/app-specific pipeline.
- Official extraction and serving must both accept original bytes and perform the JPEG treatment **exactly once**. Do not feed pre-encoded candidate patches into a preprocessing function that would encode them again.
- The active manifest must separately name and hash the original source bytes and the treated representation. File integrity and extraction paths bind to originals; treated metadata must never be mislabeled as original-file metadata. Audit/extraction provenance checks need explicit R2 support rather than replacement of an old status field.
- Compute duplicate groups over both original and treated hashes/perceptual hashes before immutable split creation. Investigate mixed-label groups and retain exact class balance/group separation.
- Run and save the official R2 audit after implementation, confirm deterministic parity with the candidate, and conduct team review. Then create new 70/15/15 splits once. Never use candidate crops as an already approved split.
- Re-run synthetic tests and train-only visual checks. Test EXIF/odd offsets, border crops, no-upscaling, deterministic JPEG bytes/quality, metadata provenance, raw-vs-treated hashes, exactly-once treatment and app/extraction parity. Retain all six model comparisons, sealed-test protections, practical-target agreement and the single held-out evaluation rules.
- The app/model card must identify fixed centre-patch scope and the actual versioned treatment. Its original authenticity/single-generator limitations remain applicable.

No new images, paid resources or GPU work are proposed for this treatment. Existing JPEG history cannot be reversed; uniform recompression is not proof of equivalent histories. The centre patch also discards border content and may sample different object scales across sources.

## Decision form

- Adopt the fixed-patch research question and exactly-once preprocessing treatment: **user approved, 30 October 2026**.
- Accept the explicitly narrower treated-input audit domain while retaining raw failure: **user approved, 30 October 2026**.
- Accept residual content/scale/compression limitations and exploratory amendment disclosure: **user approved, 30 October 2026**.
- Team reviewer(s), supervisor decision/date/rationale: **pending**.
- Authorize versioned R2 implementation and official re-audit: **user approved, 30 October 2026**.

If these changes are unacceptable, the current study remains blocked and a different acquisition/source proposal is required. A numerically below-threshold candidate alone cannot settle this decision.
