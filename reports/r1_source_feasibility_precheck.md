# R1 source feasibility precheck — 30 October 2026

Status: **existing cohort infeasible for exact geometry matching; replacement-source feasibility incomplete**. This is a read-only investigation, not approval of R1 acquisition or a passing audit.

## Work completed in live Colab

The corrected backup cell had already executed successfully when Chrome was opened. Added and executed an independent checksum/geometry verification cell. It confirmed that all **13** available backup files (audit JSON, active manifest, exclusions, report/tables and three figures) exist and match their sources. The missing split artifacts were not fabricated.

Read the existing `audit/original_audit.csv` only. All 2,500 AI rows are 512×512. The most frequent real dimensions include:

| Width | Height | Real count |
| ---: | ---: | ---: |
| 500 | 375 | 1,026 |
| 500 | 333 | 344 |
| 375 | 500 | 292 |
| 333 | 500 | 120 |
| 500 | 334 | 80 |
| 500 | 500 | 49 |

Across the full existing eligible population, exact ordered-dimension matching retains **zero per class**. Thus resampling this same acquired population cannot satisfy draft R1's exact-matching rule. These are measured stored-file dimensions; native-source history has not thereby been established.

Saved the actual Colab findings to:

`/content/drive/MyDrive/aidetect/r1_feasibility_20261030T132523Z/`

- `backup_integrity.json`
- `existing_original_dimensions.csv`
- `feasibility.json`

No acquisition, source replacement, image rewriting, splits, feature extraction, training, freeze or evaluation was executed. Existing audit records remain blocked.

## Replacement-source investigation

| Lead | Evidence inspected | Remaining limitation |
| --- | --- | --- |
| Official GenImage | Repository lists a separate SD1.5 subset and official download locations. Its dataset terms identify CC BY-NC-SA 4.0 with additional noncommercial terms. | Suitable native geometry counts and bounded access to selected files remain unverified. Official Drive folder was inaccessible through the web tool. |
| Authors' UnbiasedGenImage metadata | Author repository points to Harvard Dataverse DOI `10.7910/DVN/AKDIHF`. The indexed Dataverse V2 record lists `genimage_metadata.csv` at 213.5 MB, including geometry/quality/category information. | Direct Dataverse page requested browser robot verification; the public metadata API used in the authors' script was inaccessible through the web tool. The CSV was not downloaded or counted. Indexed values are source-reported, not an independently verified current file checksum/size. |
| `nebula/GenImage-arrow` mirror | Maintainer card states embedded bytes with source MD5/width/height for train rows, SD1.5-specific partitions, and validation/test aliases. File listing reports a 655 GB full repository and a roughly 344 kB manifest. | Third-party source/native-byte claims not independently validated. Detailed manifest/partition endpoints were inaccessible through the web tool. No data was downloaded or source adopted. |

The authors' download script iterates all listed data files. **Do not run it unchanged** for this bounded feasibility task. A separate metadata file, if accessible and approved, is the next acquisition candidate; full archive downloads are not the next step. A mirror's validation/test aliases are also not independent research splits.

## Concrete next decision

The next bounded step is to obtain and inspect **only** the separately listed GenImage metadata CSV (source-reported 213.5 MB), with a proposed maximum transfer of **250 MB**, then count shared native 512×512 real/SD1.5 geometry and trustworthy category strata. These limits and source terms require review before that transfer. Approval would cover metadata inspection only, not image downloads or adoption of a new dataset.

Stop if metadata is larger than the approved bound, requires expanded credentials, or cannot establish source revision/schema/provenance. Do not solve or bypass a browser verification challenge automatically. After metadata inspection, report exact usable counts and the smallest plausible image transfer; approve source/image budget separately before implementing the R1 adapter.

R1's proposed 2,000/class target and 1,000/class minimum remain review values. There is presently no evidence that exact native matching reaches either value. Do not broaden bins or change preprocessing merely to make this feasibility check pass.

## Sources inspected

- [Official GenImage repository](https://github.com/GenImage-Dataset/GenImage)
- [Official dataset terms](https://github.com/GenImage-Dataset/GenImage/blob/main/License)
- [Authors' metadata and mitigation repository](https://github.com/gendetection/UnbiasedGenImage)
- [Authors' downloader source, inspected but not executed](https://github.com/gendetection/UnbiasedGenImage/blob/master/download_genimage.py)
- [Indexed Harvard Dataverse record](https://doi.org/10.7910/DVN/AKDIHF)
- [Mirror maintainer card](https://huggingface.co/datasets/nebula/GenImage-arrow/blob/main/README.md)
- [Mirror file listing](https://huggingface.co/datasets/nebula/GenImage-arrow/tree/main)

Accessed 30 October 2026. A listed source is an investigation lead, not validation of its bytes, license compatibility or suitability for this study.
