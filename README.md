# BreXpert-MM

BreXpert-MM is a license-aware reconstruction pipeline for a multimodal breast imaging dataset spanning mammography (MG), MRI (MR), and ultrasound (US). It harmonises eleven upstream sources into a common exam schema with ACR BI-RADS v2025 assessments and multi-finding lesion descriptions, then generates patient-disjoint splits and templated visual question answering (VQA) dialogues.

This repository does not redistribute upstream images, metadata, processed arrays, aggregate CSVs, or VQA files. It is a workflow that rebuilds those artifacts locally from sources you obtain and are authorised to use. The result is intended for research on multimodal breast imaging representation learning, report generation, and VQA — not for clinical diagnosis, triage, or any deployment without separate regulatory, safety, and domain-expert review.

## Requirements

- Python 3.12 (pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/) for dependency management
- Independently obtained copies of the source datasets you intend to use (see [Stage the source data](#2-stage-the-source-data))
- Substantial local storage for the raw and processed datasets

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Stage the source data

Each source must be obtained directly from its original provider — see the licensing note below before downloading anything. Stage each source under `../data/raw/<source-folder>` (i.e. one directory above this repository), unless you update the corresponding adapter path in `preprocessing/`.

| Adapter slug | Upstream source and access page | Expected local staging path |
| --- | --- | --- |
| `advanced-mri-lesions` | [Advanced-MRI-Breast-Lesions, TCIA](https://www.cancerimagingarchive.net/collection/advanced-mri-breast-lesions/) | `../data/raw/advanced-mri-breast-lesions/` with DICOM files, `Advanced-MRI-Breast-Lesions-DA-Clinical-Jan112024.xlsx`, and `metadata.csv` |
| `breast-lesions-usg` | [Breast-Lesions-USG, TCIA](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/) | `../data/raw/breast-lesions-usg/BrEaST-Lesions_USG-images_and_masks/` and `BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx` |
| `breast-micro-calc` | [Breast Micro-Calcifications Dataset, Zenodo](https://zenodo.org/records/5036062) | `../data/raw/breast-micro-calc/imgs/` and `Description.xlsx` |
| `bus-cot` | [BUS-CoT, Figshare](https://doi.org/10.6084/m9.figshare.30838715) | `../data/raw/BUSCoT/BUS-Expert/` and `../data/raw/BUSCoT/DatasetFiles/BUS-Expert_dataset.json` |
| `cbis-ddsm` | [CBIS-DDSM, TCIA](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) | `../data/raw/cbis-ddsm/jpeg/` plus calcification and mass case-description CSVs for train/test |
| `embed` | [EMBED Open Data](https://github.com/Emory-HITI/EMBED_Open_Data) ([Research Use Agreement](https://github.com/Emory-HITI/EMBED_Open_Data/blob/main/EMBED_license.md)) | `../data/raw/embed/imgs/`, `EMBED_OpenData_clinical.csv`, `EMBED_OpenData_metadata_reduced.csv`, `AWS_Open_Data_Clinical_Legend.csv`, and `image_sizes.csv` |
| `inbreast` | [INbreast Kaggle](https://www.kaggle.com/datasets/martholi/inbreast) | `../data/raw/inbreast/AllDICOMs/`, `../data/raw/inbreast/AllXML/`, and `INbreast.xls` |
| `la-breast` | [LA-Breast DCE-MRI Dataset, Mendeley Data](https://data.mendeley.com/datasets/8rzyn3ng9c/1) | `../data/raw/la-breast/imgs/`, `train.csv`, `test.csv`, and `val.csv` |
| `mama-mia` | [MAMA-MIA GitHub/Synapse instructions](https://github.com/LidiaGarrucho/MAMA-MIA) | `../data/raw/mama-mia/images/`, `../data/raw/mama-mia/segmentations/expert/`, and `clinical_and_imaging_info.xlsx` |
| `oasbud` | [OASBUD, Zenodo](https://zenodo.org/records/545928) | `../data/raw/oasbud/OASBUD.mat` |
| `tompei-cmmd` | [TOMPEI-CMMD analysis result, TCIA](https://www.cancerimagingarchive.net/analysis-result/tompei-cmmd/) (plus the required CMMD image collection) | `../data/raw/tompei-cmmd/cmmd/` and `TOMPEI-CMMD_clinical_data_v01_20250121.xlsx` |

`mama-mia` aggregates the DUKE, ISPY1, ISPY2, and NACT components under mixed licenses; keep their provenance separate rather than treating them as one combined permission.

### 3. Run preprocessing

Each source is processed independently under `../data/processed/`. The adapter writes CSV and Parquet metadata, `uint16` NumPy image arrays, available masks, and a build report with dropped and unmapped values:

```bash
mkdir -p logs
python -m scripts.preprocess --dataset <adapter-slug>
```

Or process every staged source via the Makefile:

```bash
make processed-datasets
```

### 4. Build the patient-disjoint splits

```bash
make report_generation_dataset
```

This loads the processed Parquet files, falling back to CSV, and collapses BI-RADS into three action-oriented groups: `healthy (routine)`, `probably benign (follow up)`, and `suspicious/malignancy-likely (biopsy)`. It creates patient-disjoint train/val/test splits and saves the patient assignments in `patient_splits.csv` so later rebuilds keep existing patients in the same split. Output lands in `../data/report_generation_split/`, both combined (`all-rg-*`) and per modality (`mg-rg-*`, `mr-rg-*`, `us-rg-*`).

### 5. Generate VQA dialogues

```bash
make create_vqa_datasets_robust
# Optional: separate VQA files per modality
make create_vqa_datasets_robust_single_modalities
```

This produces JSONL dialogue files under `../data/robust_vqa_split/`. Training dialogues use context dropout, shuffled questions, inapplicable-question negatives, and some multiple-choice questions. Validation and test dialogues keep the full context and a fixed question order, with no negative examples. Dialogues are capped at 16 assistant turns by default.

### 6. Record what you built

After a build, update [`data_manifest.yaml`](data_manifest.yaml) with the commit you ran and the exact source versions you downloaded. Always pair reported results with this file: results from different manifests are not directly comparable.

## Using the dataset

Each processed CSV row follows the `ExamInformation` schema in [dtos/breast_cancer_dataset.py](dtos/breast_cancer_dataset.py):

| Field | Meaning |
| --- | --- |
| `id` | Exam/image-row identifier generated by the preprocessing adapter. |
| `patient` | Source-local patient identifier, usually source-prefixed or otherwise pseudonymous. |
| `dataset` | Source identifier. |
| `modality` | `mammography`, `mri`, or `ultrasound`. |
| `birads` | Harmonized BI-RADS assessment string. |
| `race` | Source-provided race/ethnicity value when available, otherwise `unknown`. |
| `machine` | Source-provided scanner/device information when available. |
| `exam` | Processed image path. |
| `segmentation` | Optional list of processed segmentation paths. |
| `context` | JSON report context string with patient/exam metadata. |
| `findings` | JSON report findings string with assessment and lesion descriptors. |
| `exam_id` | Identifier shared by rows from the same source exam. |
| `breast_id` | Optional breast-level identifier. |
| `assessment_scope` | `grounded`, `exam_level`, or `derived`. |
| `n_segmentations` | Number of segmentation paths attached to the row. |

Images are resized without changing their aspect ratio and padded to a modality-specific canvas (width × height): MG `784 × 1120`, US `672 × 560`, and MR `560 × 560`. Masks use the same geometry and remain binary. The `findings` JSON records whether lesion status is known, whether annotations are complete or index-only, and all described lesions.

Where things end up:

| Stage | Output location |
| --- | --- |
| Preprocessing | Per-source CSV/Parquet metadata, build reports, images, and masks under `../data/processed/` |
| Splits | CSV/Parquet metadata and per-modality image/mask binaries under `../data/report_generation_split/` |
| VQA dialogues | `../data/robust_vqa_split/vqa_{train,val,test}*.jsonl` |

## Validating a build

Use `notebooks/preprocessing_audit.ipynb` to check the image pipeline, masks, label mapping, and report JSON against the raw sources. Then run `notebooks/aggregated_data_checks.ipynb` for schema and split checks, and `notebooks/vqa_data_checks.ipynb` for the generated dialogues.

## Licensing and source terms

Code and documentation in this repository are MIT-licensed (see [LICENSE](LICENSE)). That license covers the pipeline only — it does not relicense, override, or extend to any upstream images, metadata, annotations, or files you generate from those sources. Each source keeps its own license and access terms (recorded per source in [data_manifest.yaml](data_manifest.yaml)); verify current terms before downloading, since they can change. Keep sources with incompatible terms in separate reconstructions, and do not publish derived artifacts unless every included source permits that exact release.

## Known limitations

- Demographic metadata (race, ethnicity, exact age) coverage is uneven across sources; age is stored in coarse bins rather than exact values.
- BI-RADS and finding labels are harmonized from heterogeneous upstream annotations; source-specific meaning and acquisition protocols may not be equivalent.

## Contributing

- Install the pre-commit hooks before your first commit: `pre-commit install`. They run `black`, `flake8`, and basic YAML/TOML/JSON/whitespace checks.
- To add a new source, create an adapter in `preprocessing/` implementing `BreastCancerDataset` (see [dtos/breast_cancer_dataset.py](dtos/breast_cancer_dataset.py)), register it in `scripts/preprocess.py`'s `dataset_mapping`, and add it to the staging table above.
- Keep changes scoped and update `README.md`/`data_manifest.yaml` when behavior or source versions change.
