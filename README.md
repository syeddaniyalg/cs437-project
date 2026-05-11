# ARIA: Autonomous Radiology Intelligence Agent

> **CS 437 — Deep Learning Research Project**
> Lahore University of Management Sciences (LUMS)
---

## Abstract

ARIA is a neuro-symbolic pipeline for automated radiology report generation from brain MRI scans. The system addresses the core limitation of end-to-end Vision-Language Model (VLM) approaches, namely, the conflation of visual perception with clinical reasoning, by interposing a deterministic semantic layer between the segmentation model and the language model. Given a multi-parametric MRI scan from the BraTS 2021 dataset, ARIA segments tumor sub-regions using a pre-trained SegResNet model, extracts over 80 quantitative radiomic features, encodes them into a Neo4j Knowledge Graph enriched with WHO CNS5, RTOG, and RANO clinical guidelines, and finally constrains a generative LLM to synthesize reports exclusively from this structured context. Evaluated using the GREEN framework, the best-performing configuration (DeepSeek-R1) achieves a GREEN Score of **0.718** and a Hallucination Score of **0.192**, representing a **+105.1%** improvement in finding-level accuracy and a **−66.2%** reduction in hallucination rate over the strongest VLM baseline (LLaVA-13B).

---

## Table of Contents

1. [Motivation](#motivation)
2. [Repository Structure](#repository-structure)
3. [System Architecture](#system-architecture)
4. [Dataset](#dataset)
5. [Components](#components)
   - [Segmentation Model](#1-segmentation-model)
   - [Radiomic Feature Extractor](#2-radiomic-feature-extractor)
   - [Knowledge Graph](#3-knowledge-graph)
   - [Generative LLM Module](#4-generative-llm-module)
   - [Baseline Pipeline](#5-baseline-pipeline)
   - [Evaluation Framework](#6-evaluation-framework)
6. [Results](#results)
7. [Installation and Setup](#installation-and-setup)
8. [Usage](#usage)
9. [Dependencies](#dependencies)
10. [Authors](#authors)
11. [References](#references)

---

## Motivation

The preparation of structured radiology reports from multi-parametric MRI (mpMRI) is a time-intensive process subject to significant inter-rater variability, compounded by a growing global shortage of radiologists. While large Vision-Language Models (VLMs) such as LLaVA, MedGemma, and Qwen-VL can generate fluent clinical text from medical images in a single forward pass, this end-to-end paradigm produces outputs that are linguistically coherent but factually unreliable. In high-stakes domains such as neuro-oncology — where an erroneous grade classification or missed ependymal contact can materially alter a treatment plan — such hallucinations are clinically unacceptable.

ARIA addresses this problem through an **architecture** that decouples visual understanding from language generation, replacing probabilistic pixel-to-text mapping with a deterministic, auditable synthesis pipeline grounded in quantifiable imaging measurements.

---

## Repository Structure

```
cs437-project/
│
├── atlas_space/                    # LPBA40 brain atlas files for lobe localization
│
├── datasets/
│   └── segmentation/               # BraTS 2021 dataset (NIfTI format)
│
├── model/
│   └── segmentation/               # Pre-trained SegResNet weights
│
├── outputs/                        # Generated radiology reports and evaluation artefacts
│
├── baseline_pipeline.ipynb         # End-to-end VLM baseline (SegResNet → VLM → Report)
├── proposed_pipeline.ipynb         # Full ARIA pipeline (Segmentation → Features → KG → LLM)
├── evaluation.ipynb                # GREEN and NLI-DeBERTa-V3 evaluation framework
└── server.ipynb                # Server setup for inference through kaggle/colab
```

> **Note:** BraTS dataset volumes is excluded from version control due to size constraints. Refer to: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1?select=BraTS2021_Training_Data.tar 
and extract the dataset inside datasets/segmentation folder

---

## System Architecture

The ARIA pipeline consists of four sequential, modular stages:

```
Raw mpMRI Scan (FLAIR, T1, T1ce, T2)
         │
         ▼
┌─────────────────────┐
│   SegResNet (MONAI) │  ──▶  Segmentation Masks (ET, TC, WT)
└─────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Custom Feature Extractor│  ──▶  JSON Payload (80+ radiomic features)
│  (Topographic, Morpho-   │
│   logical, Intensity,    │
│   Textural features)     │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Neo4j Knowledge Graph   │  ──▶  Rule-augmented graph context
│  (WHO CNS5 / RTOG / RANO │       (9 deterministic clinical rules)
│   clinical rule engine)  │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Constrained Generative  │  ──▶  Structured Clinical Report
│  LLM (DeepSeek-R1 /      │
│  LLaMA-3 / MeditronV3)   │
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  GREEN Evaluation +      │  ──▶  GREEN Score / Hallucination Score
│  NLI-DeBERTa-V3          │       / NLI Consistency Score
└──────────────────────────┘
```

The parallel **Baseline Pipeline** routes segmentation mask visualizations directly to VLMs (MedGemma-27B, LLaVA-13B, Qwen-VL-7B) without the intermediate feature and knowledge graph stages.

---

## Dataset

This project uses the **Brain Tumor Segmentation (BraTS) 2021** challenge dataset, a large-scale, multi-institutional collection of clinically acquired mpMRI scans of glioma patients.

| Property | Detail |
|---|---|
| Modalities | T1, T1ce (post-contrast), T2, T2-FLAIR |
| Format | NIfTI (`.nii.gz`) |
| Resolution | Isotropic 1 mm³ (pre-resampled) |
| Pre-processing | Co-registration, skull-stripping |

**Annotated sub-regions:**

| Label | Sub-Region | Abbreviation |
|---|---|---|
| Enhancing Tumor | Actively enhancing on T1ce | ET |
| Tumor Core | Necrotic core + ET | TC |
| Whole Tumor | NCR + ED + ET | WT |

The dataset is publicly available through the [Synapse BraTS 2021 challenge portal](https://www.synapse.org/#!Synapse:syn27046444/wiki/616571) upon registration.

---

## Components

### 1. Segmentation Model

**Notebook:** `proposed_pipeline.ipynb`, `baseline_pipeline.ipynb`

A pre-trained **SegResNet** (3D residual encoder-decoder) from the MONAI framework is used for volumetric tumor segmentation.

| Parameter | Value |
|---|---|
| Input channels | 4 (FLAIR, T1, T1ce, T2) |
| Output channels | 3 (ET, TC, WT) |
| Initial filters | 16 |
| Encoder depth blocks | [1, 2, 2, 4] |
| Decoder depth blocks | [1, 1, 1] |
| Dropout probability | 0.2 |
| Inference strategy | Sliding window (224×224×144, overlap=0.5) |
| Activation | Sigmoid (threshold = 0.5) |
| Evaluation metric | Dice Similarity Coefficient (DSC) per sub-region |

---

### 2. Radiomic Feature Extractor

**Notebook:** `proposed_pipeline.ipynb`

A deterministic feature extraction pipeline parses raw MRI tensors and their corresponding segmentation masks into a structured JSON payload. Two additional sub-regions are derived: the **Necrotic Core** (NCR = TC \ ET) and **Peritumoral Edema** (ED = WT \ TC).

Features are organized across four domains:

#### a) Topographical and Relational Features
- **Lobe Localization:** Voxel overlap fractions against the LPBA40 atlas (Frontal, Temporal, Parietal, Occipital, Deep)
- **Hemispheric Dominance:** Left/right voxel fraction; midline crossing detection
- **Focality:** Connected component analysis → Unifocal / Multifocal / Multicentric classification
- **Anatomical Proximity:** Ependymal contact via CSF mask dilation; cortical involvement via brain mask erosion
- **Centroid Coordinates:** MNI millimeter space (x, y, z)

#### b) Morphological and Geometric Features
- **Volumetrics:** Absolute volumes (mL) for ET, TC, WT, NCR, ED; Edema-to-TC ratio; necrosis proportion
- **Shape Regularity:** Sphericity (Ψ) via marching cubes surface mesh
- **Dimensionality:** PCA-derived elongation and flatness; maximum 3D physical diameter (convex hull)
- **Bounding Box:** Longest orthogonal diameters (axial, coronal, sagittal)

#### c) First-Order Intensity Statistics
Computed across 8 modality-mask pairs (e.g., T1ce-ET, FLAIR-ED):
- Distribution: mean, median, SD, variance, IQR, MAD
- Shape: skewness, kurtosis
- Energy: signal energy, RMS, Shannon entropy
- Semantic bins: enhancement level (Absent/Low/Moderate/High), tumor heterogeneity category

#### d) Texture and Composite Clinical Indices
- **3D GLCM:** 13-direction, 32-bin; Haralick features (Contrast, Entropy, Energy, IDM, Correlation, Cluster Prominence, Cluster Shade)
- **NGTDM Coarseness:** Spatial intensity change quantification
- **Infiltration Index:** FLAIR intensity ratio — peritumoral edema vs. contralateral healthy tissue
- **Enhancement Pattern:** Solid / Ring-Enhancing / Heterogeneous classification (NCR/ET ratio + margin thickness)
- **Heterogeneity Index:** Weighted composite of T1ce entropy, T1ce SD, and sphericity

---

### 3. Knowledge Graph

**Notebook:** `proposed_pipeline.ipynb`

The JSON feature payload is ingested into a **stateless Neo4j property graph** (wiped and re-instantiated per patient to prevent cross-patient data leakage).

**Graph topology:**
- Central node: `PatientCase` (continuous biomarkers as node properties)
- Categorical ontology nodes: `AnatomicLocation`, `ShapeCategory`, `EnhancementPattern`, `InfiltrationLevel`, `VolumeCategory`, `Hemisphere`
- Directed relationship edges: `SHOWS_ENHANCEMENT`, `EXHIBITS_INFILTRATION`, etc.

**Deterministic Clinical Rule Engine (9 rules, evaluated via Cypher queries):**

| Rule | Trigger Condition |
|---|---|
| 1: WHO CNS5 Grade Forcing | Necrosis proportion > 0 |
| 2: RTOG CTV Expansion | Ependymal contact ≤ 10mm |
| 3: RANO Pseudo-response Risk | Edema-to-TC ratio > 1.0 and FLAIR ratio > 2.0 |
| 4: IDH-Mutant Radiogenomic Prior | Strictly frontal, unilateral, no midline crossing |
| 5: VASARI Differential Narrowing | Heterogeneous, unifocal, no satellites |
| 6: RTOG SRS Exclusion | Max 3D diameter > 30mm | 
| 7: IDH-Wildtype Composite Prediction | Heterogeneity index > 0.6, T1ce GLCM entropy > 7.0, extensive infiltration |
| 8: Eloquent Cortex Surgical Risk | Left hemisphere localization | 
| 9: RTOG Cortical Truncation | Cortical involvement detected |

---

### 4. Generative LLM Module

**Notebook:** `proposed_pipeline.ipynb`

The knowledge graph state is serialized into a **tripartite context package** and injected into a rigid system prompt that constrains the LLM to act as a deterministic synthesis engine rather than a probabilistic prognosticator.

**Context package components:**
1. Raw extracted metrics (quantitative volumetrics, morphological parameters, textural features)
2. Categorical radiomic profile (phenotypic classifications)
3. Triggered clinical guidelines (rule engine outputs: WHO CNS5, RTOG, RANO)

**Evaluated LLM configurations (with and without Knowledge Graph):**

| Model | Parameters | Type |
|---|---|---|
| DeepSeek-R1 | 13B | General reasoning LLM |
| LLaMA-3 | 8B | General-purpose LLM |
| MeditronV3 | 7B | Medical domain-adapted LLM |

---

### 5. Baseline Pipeline

**Notebooks:** `baseline_pipeline.ipynb`

The baseline follows a two-stage segmentation-then-captioning approach:
1. SegResNet produces color-coded 2D slice visualizations (ET=red, TC=green, WT=blue on FLAIR background)
2. Annotated slices are passed directly to VLMs prompted to act as a senior neuroradiologist

**Baseline VLMs evaluated:**

| Model | Parameters | Type |
|---|---|---|
| MedGemma-27B | 27B | Medical domain-adapted multimodal |
| LLaVA-13B | 13B | General-purpose open-source VLM |
| Qwen-VL-7B | 7B | Compact multimodal model |

All models are queried at zero temperature for deterministic output. 

---

### 6. Evaluation Framework

**Notebook:** `evaluation.ipynb`

Two complementary evaluation methodologies are employed:

#### a) GREEN Score (Finding-Level Accuracy)

Inspired by Stanford's Generative Radiology Report Evaluation and Error Notation (GREEN) framework. A critic LLM (LLaMA-3, zero temperature, fixed seed) compares reference and candidate reports and classifies discrepancies as Matched Findings (M), Significant Errors (S), or Insignificant Errors (I).

$$\text{GREEN} = \frac{M}{M + S + I}$$

$$\text{Hallucination Score} = \frac{S}{M + S}$$

Evaluation is performed under two conditions: ground-truth segmentation masks and predicted segmentation masks.

#### b) NLI Consistency Score (Payload Grounding)

Each sentence in the generated report is treated as a hypothesis and evaluated against chunked segments of the JSON feature payload as premises, using `cross-encoder/nli-deberta-v3-base` as the entailment model. The consistency score is the mean maximum entailment probability across all valid claims.

---

## Results

### Baseline vs. Proposed Architecture (Ground-Truth Masks)

| Model | GREEN Score ↑ | Hallucination Score ↓ |
|---|---|---|
| **LLaVA-13B** | 0.350 | 0.568 |
| Qwen-VL-7B | 0.262 | 0.699 |
| MedGemma-27B | 0.245 | 0.730 |
| **DeepSeek-R1 (ARIA)** | **0.718** | **0.192** |
| LLaMA-3 (ARIA) | 0.573 | 0.259 |
| MeditronV3 (ARIA) | 0.330 | 0.635 |

### Percentage Improvement of ARIA over Baselines

| Metric | Model | vs. LLaVA-13B | vs. Qwen-VL-7B | vs. MedGemma-27B |
|---|---|---|---|---|
| GREEN ↑ | DeepSeek-R1 | +105.1% | +174.0% | +193.1% |
| GREEN ↑ | LLaMA-3 | +63.7% | +118.7% | +133.9% |
| Hallucination ↓ | DeepSeek-R1 | −66.2% | −72.5% | −73.7% |
| Hallucination ↓ | LLaMA-3 | −54.4% | −62.9% | −64.5% |

### NLI Consistency and KG Ablation

| Model | Consistency Score ↑ | Mean KG Divergence ↓ |
|---|---|---|
| LLaMA-3 | **0.752** | 0.1384 |
| DeepSeek-R1 | 0.596 | **0.0071** |
| MeditronV3 | — (excluded) | 0.3968 |

> DeepSeek-R1 exhibits near-zero KG divergence (0.0071), indicating output stability irrespective of knowledge graph presence — a sign of strong intrinsic instruction-following. LLaMA-3 achieves higher NLI consistency (0.752) at the cost of somewhat reduced generative expressiveness.

---

## Installation and Setup

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended: ≥16 GB VRAM for LLM inference)
- Neo4j Database (Community Edition ≥ 5.x)
- Conda or virtualenv

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/syeddaniyalg/cs437-project.git
cd cs437-project

# Create and activate a conda environment
conda create -n aria_env python=3.9
conda activate aria_env

# Install core dependencies
pip install monai nibabel numpy scipy scikit-image matplotlib
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install neo4j transformers sentence-transformers
pip install pyradiomics SimpleITK
pip install jupyter notebook
```

### Dataset Acquisition

1. Register and download the BraTS 2021 dataset from the [Synapse portal](https://www.synapse.org/#!Synapse:syn27046444).
2. Extract the dataset and place subject folders under `datasets/segmentation/`.
3. Ensure each subject directory contains the four modality volumes: `*_flair.nii.gz`, `*_t1.nii.gz`, `*_t1ce.nii.gz`, `*_t2.nii.gz`, and `*_seg.nii.gz`.

### Atlas Setup

The LPBA40 probabilistic brain atlas files required for lobe localization are stored in `atlas_space/`. No additional download is required if pulling the full repository.

### Model Weights
The pre-trained model weights are already present model/segmentation

### Neo4j Configuration

You have to create a new local Neo4j database inside the system, and make sure to keep credentials and database name consistent to what is used inside `proposed_pipeline.ipynb`.

---

## Usage

All pipelines are implemented as Jupyter notebooks. Launch the notebook server from the project root:

```bash
jupyter notebook
```

### Running the Proposed Pipeline

Open `proposed_pipeline.ipynb` and execute cells sequentially. The notebook covers:
1. Data loading and SegResNet inference
2. Radiomic feature extraction and JSON serialization
3. Neo4j graph instantiation and clinical rule engine execution
4. Constrained LLM report synthesis (configure model selection in the designated cell)
5. Report export to `outputs/`

### Running the Baseline Pipeline

Open `baseline_pipeline.ipynb`, then execute cells to generate baseline reports for MedGemma-27B, LLaVA-13B, and Qwen-VL-7B.

### Server notebook
`server.ipynb` is alternative you can use if the local computer doesn't have enough resources to run inference. The `server.ipynb` notebook can be deployed on kaggle/colab and model can be initialized there. But just make sure to copy the link and use it for model inside that particular inference cell. 

## Dependencies

| Package | Purpose |
|---|---|
| `monai` | SegResNet model, volumetric inference utilities |
| `nibabel` | NIfTI file I/O |
| `numpy`, `scipy`, `scikit-image` | Numerical computing, morphological operations |
| `pyradiomics` | Radiomic feature extraction (GLCM, NGTDM) |
| `SimpleITK` | Medical image registration and processing |
| `neo4j` | Knowledge graph database driver |
| `transformers` | LLM inference (LLaMA-3, DeepSeek-R1, MeditronV3) |
| `sentence-transformers` | NLI-DeBERTa-V3 consistency scoring |
| `matplotlib`, `seaborn` | Visualization and result plotting |
| `torch` | Deep learning backend |

---

## Limitations

- **Absence of ground-truth clinical reports:** The GREEN evaluation uses LLaMA-3-generated reports from ground-truth segmentation masks as proxy references rather than radiologist-authored reports. This introduces a ceiling on evaluation fidelity.
- **Segmentation error propagation:** Errors in SegResNet outputs propagate directly into the radiomic feature payload. Upgrading to a higher-fidelity backbone (e.g., nnUNet, transformer-based architectures) is expected to yield measurable downstream improvements.
- **MeditronV3 instruction-following failure:** Despite medical domain adaptation, MeditronV3 did not reliably follow the constrained system prompt, frequently reproducing prompt instructions rather than clinical content.

---

## Future Work

1. **Higher-fidelity segmentation backbone:** Integration of nnUNet v2 or a transformer-based segmentation model to improve radiomic feature accuracy.
2. **Cross-domain generalization:** Extension to prostate MRI (PI-RADS ontology), lung CT, and abdominal imaging by substituting the segmentation backbone and clinical ontology layer.
3. **Ground-truth clinical report evaluation:** Acquisition of paired mpMRI and verified radiologist report datasets (e.g., MIMIC-CXR-style extensions for brain MRI) for clinically validated benchmarking.

---

## Authors

- **Daniyal Hussain Shah** — LUMS
- **Ali Hasnain** — LUMS

Project repository: [https://github.com/syeddaniyalg/cs437-project](https://github.com/syeddaniyalg/cs437-project)

---

## References

1. Baid et al. — The RSNA-ASNR-MICCAI BraTS 2021 Benchmark. *arXiv:2107.02314*, 2021.
2. Myronenko, A. — 3D MRI Brain Tumor Segmentation Using Autoencoder Regularization. *MICCAI BrainLesion Workshop*, 2018.
3. MONAI Consortium — An Open-Source Framework for Deep Learning in Healthcare. *arXiv:2211.02701*, 2022.
4. Ostmeier et al. — GREEN: Generative Radiology Report Evaluation and Error Notation. *ACL Findings*, 2024.
5. DeepSeek-AI — DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. *arXiv:2501.12948*, 2025.
6. Dubey et al. — The Llama 3 Herd of Models. *arXiv:2407.21783*, 2024.
7. Liu et al. — Visual Instruction Tuning (LLaVA). *NeurIPS*, 2023.
8. Bai et al. — Qwen-VL: A Versatile Vision-Language Model. *arXiv:2308.12966*, 2023.
9. He et al. — DeBERTa: Decoding-Enhanced BERT with Disentangled Attention. *ICLR*, 2021.
10. van Griethuysen et al. — Computational Radiomics System to Decode the Tumour Phenotype. *Cancer Research*, 2017.
11. Louis et al. — The 2021 WHO Classification of Tumors of the Central Nervous System. *Neuro-Oncology*, 2021.
12. Wen et al. — Updated Response Assessment Criteria for High-Grade Gliomas: RANO. *Journal of Clinical Oncology*, 2010.
13. Lewis et al. — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*, 2020.

---
