# GenImage metadata feasibility — 30 October 2026

**Conclusion: draft R1 exact native-geometry matching is infeasible for this metadata source at the proposed minimum of 1,000 images per class. Do not download the image archives for R1.**

## Input and integrity

Read-only input: `C:/Users/ADMIN/Downloads/genimage_metadata.csv`, supplied by the user after downloading the separately listed Harvard Dataverse metadata file. The source CSV was not edited or copied into Git.

- Size: 223,850,689 bytes.
- SHA-256: `afd9b5f8b1911997b5e859c8fe94380b1857bcd3916ac4f4c9734411bbb89ac2`.
- MD5: `2242b70fede792b9a08165f6023ca196`, consistent with the previously displayed abbreviated `224...196` checksum. The complete publisher checksum has not been independently retrieved; abbreviated agreement alone is not full publisher-integrity verification.
- Columns: `path`, `generator`, `mode`, `width`, `height`, `compression_rate`, `class`.
- Rows: **2,681,150**. No width/height parse failures in the streaming inspection.

Dimensions and quality are publisher metadata claims, not a fresh decode or quantization-table measurement of candidate image files. `class` represents the source content category; it is not the binary real/AI label. `mode` is an upstream source partition, not this study's sealed test split.

## Exact 512×512 counts

| Population | Train | Source val | Total |
| --- | ---: | ---: | ---: |
| SD1.5 generated rows | 166,000 | 8,000 | 174,000 |
| Real rows across all generator source folders | 49 | 2 | 51 |
| Real rows in the SD1.5 source folder | 7 | 0 | 7 |

All 174,000 SD1.5 generated rows have dimensions 512×512 in this metadata. Therefore exact ordered geometry matching is limited by the real-image population: at most 51 metadata rows per class even if real images are pooled from every source folder, or 7 per class when restricted to the SD1.5 source folder.

For the SD1.5 train source folder, matching on content class as well as 512×512 geometry also retains only 7 per class. None of the 51 real 512×512 entries has recorded compression rate 96. That additional quality constraint would reduce this exact-geometry population to zero; it is a diagnostic count, not a proposed change to the PRD's median-quality remedy.

As a check against repeated real-file listings, the 51 real 512×512 rows were keyed by `(mode, filename)`, ignoring the enclosing generator folder. This yielded 49 train and 2 val logical identities, with no conflicting dimension/category/quality metadata among repeated logical keys. This is not image-byte or perceptual deduplication; such checks could only reduce, not increase, the usable exact-matched population.

## Broader size interval: diagnostic only

Real entries with both dimensions in [450, 550] number **47,704 train** and **4,323 source val**. This demonstrates that near-size natural images are more common than exact 512×512 images. It does **not** satisfy R1 exact geometry matching or establish a passing original-metadata probe. No cohort was selected from this interval, and no new probe, resizing or preprocessing change was run.

Adopting a wider interval and a fixed crop before resizing would be a different protocol, with a different treatment and a need to address the original-metadata gate explicitly. Do not silently widen R1 or claim those counts unblock the existing code. An alternative source of verified native 512×512 real images is another separately reviewed possibility; its content/encoding provenance and sufficient counts are not established here.

## Verification performed

1. Standard-library streaming `csv.DictReader` pass counted rows by generator/source partition, dimension and folder, and examined exact-geometry logical identities and content matches.
2. Independent pandas pass in 100,000-row chunks reproduced total rows, SD1.5 rows, exact SD1.5 geometry, all real exact-geometry rows and SD1.5-folder real exact-geometry rows. Explicit equality checks passed for `(2,681,150; 174,000; 174,000; 51; 7)`.
3. Computed complete SHA-256 and MD5 from the original downloaded bytes.

Used the bundled Python runtime at `C:/Users/ADMIN/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe` for the two counting passes. No image dataset, model, sealed features or research predictions were loaded. No metadata-only audit pass is being claimed from row counts.

## Next decision

Close the investigated GenImage exact-matching candidate as infeasible under draft R1. Review either a separately justified source of native 512×512 real images or an explicit R2 treatment/audit amendment with the supervisor. Preserve the failed Tiny-GenImage audit and this failed R1 feasibility finding. Do not acquire image archives, create splits or train under the assumption that this CSV fixes the shortcut.
