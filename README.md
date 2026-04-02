# EV User Perception Analysis — Ordinal Regression & Statistical Study

> A quantitative survey-based study investigating consumer perception factors influencing Electric Vehicle (EV) adoption in Germany, using psychometric scaling, non-parametric hypothesis testing, dimensionality reduction, and unsupervised clustering.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Research Context](#research-context)
- [Repository Structure](#repository-structure)
- [⚠️ Data Availability & Privacy](#️-data-availability--privacy)
- [Dataset Description](#dataset-description)
- [Survey Instrument](#survey-instrument)
- [Analytical Pipeline](#analytical-pipeline)
- [Key Findings](#key-findings)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Theoretical Background](#theoretical-background)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

---

## Project Overview

This project presents a data-driven analysis of user perception toward Electric Vehicles (EVs) based on primary survey data collected from respondents across Germany and other regions (January 2024). The study quantifies the relative importance of six key EV adaptation factors — including range anxiety, charging infrastructure, battery issues, charging time, autonomous driving, and vehicle-to-everything (V2X) connectivity — and statistically tests for distributional differences between them using ordinal-appropriate methods.

The analysis applies:
- **Exploratory Data Analysis (EDA)** with frequency distribution visualizations
- **Principal Component Analysis (PCA)** for latent structure discovery
- **K-Means and Hierarchical Clustering** for consumer segmentation
- **Mann-Whitney U Tests** for pairwise non-parametric hypothesis testing across all perception dimensions

---

## Research Context

The study was conducted as part of an investigation into the real-life versus perceived user experience of EV adaptation in Germany — a country with ambitious electromobility and sustainability targets. The core research question is:

> *Which consumer perception factors most significantly differentiate EV adoption intent, and how do these factors cluster across the population?*

The research is grounded in **Technology Acceptance Modeling (TAM)** and **Consumer Behavior Theory**, with relevance to sustainability transition policy and EV market strategy.

---

## Repository Structure

```
ev-perception-analysis/
│
├── notebooks/
│   └── EVPerception_OrdinalRegression.ipynb   # Main analysis notebook
│
├── data/
│   └── .gitkeep                               # Folder exists but data is NOT committed
│                                              # See Data Availability section below
├── survey/
│   └── EV_Adaptation_User_Experience_Google_Forms.pdf  # Survey instrument (public)
│
├── .gitignore                                 # Prevents accidental data commits
├── requirements.txt                           # Python dependencies
└── README.md
```

> **Note:** The `data/` folder is intentionally empty in this repository. The raw dataset is not published for privacy and GDPR reasons. See the section below for details.

---

## ⚠️ Data Availability & Privacy

### Why the data file is not in this repository

The raw survey dataset (`UserPerception.xlsx`) is **intentionally excluded** from this repository for the following reasons:

- **Respondent privacy:** The dataset contains survey responses from individuals. Publishing it publicly without explicit consent for open distribution would be ethically inappropriate.
- **GDPR compliance:** The survey was conducted in Germany and collected responses from EU residents. Under GDPR, personal or individually identifiable data must not be published without a lawful basis.
- **Google Drive security:** The notebook accesses data from the researcher's own Google Drive. Cloning this repository does **not** grant access to that Drive — each user must authenticate with their own Google account and supply their own data file.

### What you CAN do to run this notebook

**Option A — Use your own data file**
Prepare an Excel file with the same column structure shown in the Dataset Description section below, and place it at:
```
data/UserPerception.xlsx        # for local Jupyter
MyDrive/UserPerception.xlsx     # for Google Colab (your own Drive)
```

**Option B — Request an anonymised sample**
Contact the repository owner to request access to a stripped, anonymised version of the dataset suitable for academic or research use.

### How the notebook handles missing data

The data loading cell will raise a clear, descriptive error if the file is not found — it will never silently fail or attempt to access another user's Google Drive:

```
FileNotFoundError: Data file could not be located.
See README.md → Data Availability section for the expected file format.
Contact the repository owner to request an anonymised sample dataset.
```

### Expected file format

To create a compatible dataset, your Excel file must contain the following columns:

| Column | Type | Values | Description |
|---|---|---|---|
| `Vehicle Purchase` | String | See survey | Current ownership / future intent |
| `V_P` | Integer | 0 or 1 | Purchase intent (encoded) |
| `R_A` | Integer | 1–5 | Range anxiety rating |
| `C_S` | Integer | 1–5 | Charging infrastructure rating |
| `B_I` | Integer | 1–5 | Battery issues rating |
| `C_T` | Integer | 1–5 | Charging time rating |
| `A_D` | Integer | 1–5 | Autonomous driving rating |
| `V_C` | Integer | 1–5 | V2X connectivity rating |
| `Total` | Integer | 6–30 | Sum of R_A + C_S + B_I + C_T + A_D + V_C |

---

## Dataset Description

The dataset contains Likert-scale survey responses encoded as follows:

| Column | Full Name | Description | Scale |
|---|---|---|---|
| `Vehicle Purchase` | Vehicle Purchase Status | Current ownership / future intent | Categorical |
| `V_P` | Vehicle Purchase (encoded) | 1 = planning to buy, 0 = not planning | Binary |
| `R_A` | Range Anxiety | Perceived concern about battery range and road range capacity | 1–5 Likert |
| `C_S` | Charging Speed / Infrastructure | Perceived importance of charging station density and speed | 1–5 Likert |
| `B_I` | Battery Issues | Concern over battery type, thermal safety (BMS) | 1–5 Likert |
| `C_T` | Charging Time | Tolerance and importance of charge duration | 1–5 Likert |
| `A_D` | Autonomous Driving | Perceived importance of autonomous driving safety and experience | 1–5 Likert |
| `V_C` | V2X / Vehicle Connectivity | Importance of vehicle-to-everything connectivity and safety | 1–5 Likert |
| `Total` | Composite Score | Sum of all perception dimension scores | Numeric |

> All perception dimensions use a **5-point Likert scale** (1 = Not Important, 5 = Very Important), which is ordinal data. All statistical methods are chosen accordingly.

---

## Survey Instrument

The primary data was collected via a structured Google Forms survey titled *"User Experience in EV Adaptation"* (15.01.2024). The survey instrument (`survey/` folder) is included in this repository as it contains no personal data.

The survey includes:
- **Demographics:** Age group, country of residence
- **Vehicle Ownership:** Current ownership status and future purchase intent
- **Vehicle Preference:** Preferred vehicle type (BEV, ICEV, PHEV)
- **Adaptation Factors (Multi-select):** Which factors affect EV buying decision
- **Importance Ratings (Likert 1–5):** Per-factor importance ratings for Range Anxiety, Charging Infrastructure, Battery Issues, Charging Time, Autonomous Driving, and V2X Connectivity
- **Open-ended question:** Additional factors the respondent considers important for EV adoption

The survey was distributed internationally with a focus on German respondents, using SurveySwap.io for response collection.

---

## Analytical Pipeline

### 1. Data Loading & Preprocessing (Secure)
The notebook uses an environment-aware, secure loading block. It auto-detects whether it is running in Google Colab or locally, mounts the user's own Drive if in Colab, and raises a descriptive error (never silently fails) if the file is not found. No hardcoded paths or credentials are present.

```python
# Simplified illustration — see notebook for full secure implementation
DATA_FILE = 'UserPerception.xlsx'
COLAB_PATH = f'/content/drive/MyDrive/{DATA_FILE}'   # user's own Drive
LOCAL_PATH = os.path.join('data', DATA_FILE)
```

### 2. Exploratory Data Analysis (EDA)
Seaborn `countplot` visualizations show the frequency distribution of responses across each Likert dimension.

### 3. Dimensionality Reduction — PCA
Principal Component Analysis reduces the multivariate perception space, identifying latent structure and co-variance patterns.

```python
from sklearn.decomposition import PCA
```

### 4. Unsupervised Clustering
Both **K-Means** and **Hierarchical Clustering (Dendrogram)** segment respondents into behavioural profiles.

```python
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
```

### 5. Non-Parametric Hypothesis Testing — Mann-Whitney U Test
Pairwise tests across all perception dimension combinations, appropriate for ordinal Likert data.

```python
from scipy import stats
stats.mannwhitneyu(x=df['V_C'], y=df['C_S'], alternative='two-sided')
```

---

## Key Findings

| Comparison | P-Value | Interpretation |
|---|---|---|
| V_C vs C_S | 1.64e-10 | Highly significant distributional difference |
| V_C vs R_A | 5.41e-09 | Highly significant distributional difference |
| V_C vs B_I | 3.19e-07 | Significant |
| A_D vs V_C | 0.944 | **Not significant** — similar distribution |
| R_A vs C_S | 0.349 | Not significant |
| R_A vs B_I | 0.396 | Not significant |
| V_P vs all dimensions | < 1e-40 | Purchase intent is statistically distinct from every individual perception factor |

**Notable insight:** The non-significance between `A_D` and `V_C` suggests respondents perceive autonomous driving and V2X connectivity with similarly distributed importance — both emerging technologies evaluated comparably. The extreme significance of `V_P` vs all factors confirms purchase intent is a composite outcome not reducible to any single perception dimension.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
tabulate
openpyxl
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

Or in Google Colab:

```python
!pip install tabulate openpyxl
```

---

## Getting Started

### Option A — Google Colab (Recommended)

1. Place your copy of `UserPerception.xlsx` in your own Google Drive
2. Open `notebooks/EVPerception_OrdinalRegression.ipynb` in Google Colab
3. When the data loading cell runs, authenticate with **your own** Google account when prompted
4. The notebook will auto-locate the file at `MyDrive/UserPerception.xlsx`
5. Run all cells sequentially

> Your Google Drive is never accessible to anyone else running this notebook. Authentication is always per-user.

### Option B — Local Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ev-perception-analysis.git
   cd ev-perception-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your data file at `data/UserPerception.xlsx`

4. Launch Jupyter:
   ```bash
   jupyter notebook notebooks/EVPerception_OrdinalRegression.ipynb
   ```

---

## Usage

The notebook runs sequentially from top to bottom. Each section is independently documented.

| Section | Output |
|---|---|
| Data loading & preview | DataFrame shape, head/tail |
| EDA countplots | Distribution plots per dimension |
| PCA | Component variance plots |
| Clustering | Dendrogram, K-Means cluster assignments |
| Mann-Whitney U tests | Individual and summary p-value table |

---

## Theoretical Background

- **Technology Acceptance Model (TAM)** — Davis (1989)
- **Theory of Planned Behavior (TPB)** — Ajzen (1991)
- **Sustainability Transition Research** — Geels (2002)
- **Ordinal Data Analysis** — Agresti (2010)

---

## Limitations & Future Work

**Current limitations:**
- Sample may not be fully representative of the broader German population
- Binary encoding of `V_P` collapses nuance in adoption intent
- Cross-sectional design prevents causal inference

**Suggested extensions:**
- Ordinal Logistic Regression — model `V_P` as outcome predicted jointly by all perception dimensions
- Spearman Correlation Matrix — quantify ordinal pairwise associations
- Structural Equation Modeling (SEM) — test latent causal pathways
- Longitudinal follow-up — track perception changes as EV infrastructure expands

---

## License

This project is for academic and research purposes. The survey data was collected with participant consent. The raw dataset is not published. Please contact the repository owner for research collaboration or data access requests.

---

*Survey conducted: January 2024 | Analysis environment: Python 3.10, Google Colab*
