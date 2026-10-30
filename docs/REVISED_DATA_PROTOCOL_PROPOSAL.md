# Revised data protocol proposal

**Draft R1 — 30 October 2026. Prepared for team/supervisor review. Not approved or implemented.**

**Feasibility update:** The user-supplied full GenImage metadata CSV was subsequently inspected. It contains only 51 real 512×512 entries across all source folders, including just 7 in the SD1.5 folder. The investigated GenImage exact-matching candidate therefore cannot reach R1's proposed 1,000/class minimum. This draft is retained as the reviewed investigation plan, not a ready-to-run acquisition recommendation. See [the measured feasibility result](../reports/genimage_metadata_feasibility_2026-10-30.md). A different source or an explicit R2 amendment needs a new decision.

Project: Detecting AI-Generated Images Using Frequency-Domain Features.

## Decision requested

Approve a metadata-first feasibility review of a native-resolution SD1.5/ImageNet cohort, followed by a separately approved, bounded acquisition if feasible. Keep the current Tiny-GenImage run as a documented failed acquisition candidate. Preserve the scientific question, six models, shared preprocessing and 0.60 shortcut limit.

The user's authorization is to **prepare this proposal**. It does not authorize downloads, dataset replacement, preprocessing changes, split creation or downstream execution. Supervisor agreement is not recorded.

## Why a revision is necessary

The live Colab audit used 2,500 real and 2,500 SD1.5 images after eligibility and balancing. Identical JPEG encoding at estimated quality 96 left metadata-only accuracies of 0.8308 (logistic regression) and 0.9998 (depth-3 tree). Both exceed 0.60. Width alone gave tree accuracy 0.9992. Exact dimension matching retained no pairs. Approximate matching retained at most 50 per class, and every adequately sized candidate failed its probe. These are metadata probe scores, not detector results.

Evidence: [live verification report](../reports/live_colab_verification_2026-10-30.md), Drive audit and diagnostic reports listed there. No research split or held-out evaluation has been created. The present failure concerns this collected cohort; it does not establish that every SD1.5 source is unsuitable.

Merely resizing all stored files to 256×256 would make their measured dimensions identical while potentially retaining class-dependent resampling history in the frequency spectrum. A metadata pass on those rewritten dimensions would not establish removal of the original shortcut. Repeatedly changing bins, quality or sources until a probe passes would also make the audit an adaptive selection exercise. Neither is the proposed remedy.

## Options considered

| Option | Consequence | Disposition |
| --- | --- | --- |
| Native-resolution source with geometry-matched acquisition | New source/provenance review and potential transfer; preserves the main task and preprocessing if feasible | Recommended staged investigation; no guaranteed pass |
| Resize/crop current acquisition before the audit | Changes the experimental treatment; can hide geometry metadata without removing spectral or content confounding | Excluded from R1; would require a different explicitly justified protocol |
| Switch to SD1.4 | Changes the single-generator scope, source mapping and app/report claims; may still have size/encoding bias | Separate fallback proposal only; not automatic |
| Retain Tiny-GenImage and publish the failed audit as the outcome | Valid methodological finding, but cannot supply the planned six-model research comparison | Honest fallback if an eligible free source is infeasible |

## R1: proposed acquisition and audit procedure

### 1. Preserve the failed experiment

Retain original bytes, acquisition manifests, failed audit, diagnostic candidates and notebook outputs. Record their hashes and the existing config hash. Never overwrite `data/main`, `data/audit` or their evidence. Use a separate prospective root, for example `aidetect/r1_native_sd15/`, after approval. Do not reuse a passing flag or mix old/new manifests.

### 2. Review source feasibility before downloading image archives

Start with the official GenImage SD1.5 branch and any metadata offered by the corresponding bias-analysis project. These are investigation leads, not approved mirrors or verified subsets. Do not automatically fetch the full collection.

Prepare a source feasibility record containing:

- Exact source URL, revision or immutable release, native-file/derivative status, SD1.5 identity evidence, and real-image provenance.
- Applicable image and metadata terms, required attribution, and any access restrictions. Do not assume a repository's code license covers its images, or carry Tiny-GenImage's stated license onto another source.
- Available original width/height, encoding/quality metadata and, if available, content-category identifiers. Distinguish measured values from source claims. Missing metadata must be measured later and may invalidate feasibility.
- Counts in exact native geometry strata, missing/corrupt entries, potential duplication, and whether enough real/AI images share geometry.
- The minimum archive/range access required to obtain selected files; estimated transfer, extracted storage, runtime and available free Drive/Colab capacity. Set explicit team-approved transfer/storage limits **before** acquisition. Stop if selective access is unavailable or the limit would be exceeded.

No paid service, API key, generation job, GPU or full-dataset download is assumed. Even metadata access that entails a large transfer requires its cost/capacity review. Existing failed-cohort images may only be reused if their original identity and native provenance can be verified against the selected source; no image reuse is presumed feasible.

### 3. Lock the cohort-selection rules before acquisition and probes

Proposed rules requiring approval:

1. AI label 1 means SD1.5 only; real label 0 means the approved real source. No generator mixing or silent SD1.4 substitution.
2. Both native sides must be at least 256; no upscaling. Require evidence of native dimensions rather than trusting resized preview metadata.
3. Match classes exactly on **ordered native `(width, height)`**. Within each stratum, retain equal real/AI counts. This proposal deliberately does not reuse the unsuccessful approximate-bin search.
4. Proposed target: **2,000 per class**. Proposed minimum after exclusions/duplicate handling: **1,000 per class**. These are review values, not a power guarantee. Below the minimum, stop and request a new decision; do not silently shrink the experiment.
5. Use seed 42 and stable IDs. Deterministically randomize rows within strata and equal-count real/AI pairs across strata, then retain up to the target. Save the candidate universe, ordered selection and all exclusions. Select using metadata/provenance only, never FFT/pixel scores or detector performance.
6. If content-category identifiers are trustworthy and available for both classes, match within `(category, width, height)` instead. Declare which rule is feasible and lock it before selecting images. If unavailable, record residual content confounding and retain the restriction against per-category accuracy claims.
7. Do not keep testing alternative cohorts after failure under the same R1 record. A failed gate closes this candidate. Any further candidate needs a dated amendment that records all prior failures.

Exact geometry matching reduces one measured confound; it does not establish equal object scale, JPEG history or freedom from other shortcuts. If the source cannot supply enough shared native geometry, R1 is infeasible rather than a reason to relax the matching rule during execution.

At the nominal 15% test allocation, 2,000 per class yields about 600 test images total; 1,000 per class yields about 300. For planning only, the usual binomial approximation at accuracy 0.5 gives 95% half-widths near 4.0 and 5.7 percentage points, respectively. Actual intervals must use the prescribed exact method and actual frozen counts. These estimates do not establish power for paired frequency-versus-pixel comparisons; the team must review the precision and practical target before accepting the minimum.

### 4. Acquire original bytes and independently verify metadata

Implement an acquisition adapter only after the source and bounded transfer are approved. Preserve raw files. Record stable source IDs, generator mapping, revision, original SHA-256, dimensions, encoding and selection policy/version. Independently decode and verify the acquired files; a disagreement with source metadata stops the candidate rather than silently substituting other images.

Exclude and record corrupt/undersized files before the cohort is finalized. Restore exact geometry/class balance deterministically from the locked candidate order; do not use replacement with duplicates. Deduplicate original bytes and perceptual hashes before final sampling. Mixed-label duplicates require explicit investigation/exclusion. Preserve original-plus-normalized union groups and the existing group-separation rules for the final audit/split pipeline. Confirm the final cohort still meets the approved minimum and geometry balance.

### 5. Apply the original audit gate to the revised cohort

Keep all six metadata columns, balanced seeded five-fold probes, logistic regression and depth-3 tree. Audit original files first. Retain the existing conditional remedy: if either mean exceeds 0.60, encode **both** classes identically at the real cohort's median estimated JPEG quality with 4:4:4 subsampling, then repeat the same probes once. Missing usable real JPEG quality blocks the remedy.

Both final means must be <=0.60. Retain before/after distributions, all fold scores, exclusions, source/cohort hashes and final group/geometry counts. Do not remove byte size, alter the threshold, or redefine the measured files to obtain a pass. A pass only means these preregistered probes did not detect a shortcut at this limit; residual encoding history and content bias remain limitations.

Failure stops R1. The team reviews the evidence before another amendment. No frequency features or detector scores are used to choose a replacement acquisition.

### 6. Resume the existing research pipeline after review

After a passing audit and team review, create the immutable class-balanced group-separated 70/15/15 split once in the new root. Confirm feasibility of group allocation and exact class balance; retain exclusions and final counts. Do not regenerate it to obtain nicer outcomes.

Keep `preprocess_image` unchanged: EXIF orientation, RGB-to-L, centre square crop, conditional LANCZOS downsampling to 256, no upscaling. Geometry matching should make the applied resizing geometry comparable across labels; this is still not proof of equivalent source history.

Then complete Phase 3/4 train-only visual review, Phase 5 deterministic extraction/sealing, Phase 6 six matched train/validation pipelines and model selection, agreed practical target, and Phase 7 freeze/one evaluation in their established order. Update full config/provenance hashes and feature-contract compatibility deliberately; old batches/models must fail reuse checks. All teams' review gates remain in force. Phase 8 serving and Phase 9 reporting must identify the actual approved source and cohort.

## Required implementation after approval

| Area | Work and validation |
| --- | --- |
| Configuration | Version R1 source/selection settings and new roots; reject accidental old/new mixing |
| Acquisition | Adapter, deterministic matched sampler, original-byte manifests, bounded transfers/resume checks |
| Audit/splits | Geometry/count/provenance checks; retain original shortcut gate and immutable grouping |
| Tests | Tiny fixtures for matching, seed determinism, metadata disagreement, interrupted resume, source hashes, duplicates, insufficient strata, threshold refusal and split separation |
| Documentation | Approved decision, deviations, source/license review, phase report and explicit Colab commands |
| Colab environment | Separate pinned research environment if required to resolve the pandas/Colab host conflict; verify imports, `pip check` and all tests there without changing research pins just to silence the host warning |

This proposal changes no implementation. Software tests passing remain separate from successful real-data gates. No approved practical target, research model or evaluation result is being supplied here.

## Approval record — complete before implementation

- Decision: **pending** (approve R1 / revise / retain blocked outcome).
- Approved source/revision and acceptable native provenance: **pending**.
- Category availability and exact matching key: **pending**.
- Sample target/minimum and statistical precision assessment: **pending**.
- Free transfer/storage limits and access/license review: **pending**.
- Team reviewer(s), supervisor decision, date and recorded rationale: **pending**.
- Separate authorization to implement/acquire: **pending**.

Approving the draft text does not imply a passing audit or authorize a single held-out evaluation. If feasibility fails, record it and return for a decision.

## Primary sources consulted

Grommelt et al., *Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets*, arXiv:2403.17608v2 (2024), especially sections 3.2 and 4.2. The paper discusses geometry/resizing confounds and a size-constrained mitigation. Its experimental setting and treatment differ from this FFT study; its results do not validate R1. [Full text](https://arxiv.org/html/2403.17608v2).

The authors' repository describes metadata-assisted selection and compression alignment, and advertises a complete collection of roughly 500 GB. That makes a bounded-access feasibility check necessary; no such full transfer is proposed. [Author repository](https://github.com/gendetection/UnbiasedGenImage).

The official GenImage repository lists separate SD1.4 and SD1.5 subsets and published download locations. Availability of individual native files, suitable matched counts and permitted access still require verification. [Official repository](https://github.com/GenImage-Dataset/GenImage).

Sources accessed on 30 October 2026. Literature performance numbers are not used as expected project results.
