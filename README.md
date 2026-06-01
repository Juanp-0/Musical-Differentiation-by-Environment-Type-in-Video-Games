# Musical Differentiation by Environment Type in Video Games
### A Study Based on BPM, Tonality, and Console

> **Course project** · Big Data Analytics · Universidad TecMilenio · Jan–Mar 2026

---

## Overview

Do aquatic levels actually sound different from jungle or desert levels? This project investigates whether video game music shows **statistically distinguishable patterns** depending on the type of environment (aquatic, forest, jungle, ice, beach, space), using a dataset of ~30,000 MIDI files.

The analysis focuses on two measurable features: **BPM (tempo)** and **MIDI key (tonality)**, processed at scale using Apache Spark and extracted with the `pretty_midi` library.

---

## Research Question

> *Are there distinguishable musical patterns associated with different types of environments in video games?*

**Hypothesis:** Aquatic levels use slower tempos and more sustained notes; jungle levels show higher rhythmic density; desert levels lean toward minor scales.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| PySpark | Distributed data processing (30k+ records) |
| pretty_midi | MIDI feature extraction (BPM, key) |
| scikit-learn | RandomForest classification model |
| matplotlib / seaborn | Data visualization |
| numpy | Numerical operations |
| Google Colab | Development environment |

---

## Dataset

**Source:** [30,000 Video Game MIDI Files](https://www.kaggle.com/datasets/hansespinosa2/40000-video-game-midi-file) by hansespinosa2 (Kaggle)

Each record includes: MIDI file URL, song name, game title, company, console, and contributor credits.

**After cleaning:**
- Removed duplicates (by song name + game)
- Normalized versioned titles (V.1, V.2, etc.)
- Filtered out remixes and arrangements
- **Final dataset: ~17,379 unique records**

---

## Methodology

### 1. Data Cleaning
Duplicate removal, version normalization, remix filtering — all processed via PySpark SQL functions.

### 2. Environment Sampling
Songs were classified into 7 categories by keyword filtering on song names using PySpark's `regexp_like`:

| Category | Keywords |
|----------|----------|
| Aquatic | Water, Aqua, River, Lake, Labyrinth Zone… |
| Forest | Forest, Woods |
| Jungle | Jungle |
| Ice | Ice, Snow, Winter, Freeze… |
| Beach | Beach, Shore |
| Desert | Desert, Wasteland |
| Space | Space, Star, Moon |

### 3. Feature Extraction
A custom UDF registered in PySpark called `pretty_midi` on each MIDI URL to extract:
- **Key:** via `key_signature_changes`; estimated via `get_chroma` + numpy argmax when metadata was absent
- **BPM:** via `get_tempo_changes`; estimated via `estimate_tempo` as fallback

### 4. Analysis
- MIDI key distribution per environment (bar charts)
- Average BPM per environment category
- Correlation heatmap: average BPM by MIDI key × console

### 5. Machine Learning
RandomForest classifier (scikit-learn) trained to predict environment category from BPM + key:
- 70/30 train-test split
- Pipeline: StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler

---

## Key Findings

### Tonality
C Major dominates across **all** environment types — video game composers consistently favor it as a neutral, versatile key regardless of setting. Minor keys are rare across the board, which partially contradicts the original hypothesis about desert environments.

### Tempo (BPM)
Tempo shows clearer differentiation between environments:

| Category | Avg BPM |
|----------|---------|
| Jungle | ~151 *(highest)* |
| Ice | ~135 |
| Aquatic | ~134 |
| Forest | ~131 |
| Beach | ~128 |
| Space | ~127 *(lowest)* |

Jungle tracks are noticeably faster, consistent with high-energy gameplay. Beach and Space themes are the most relaxed.

### Console Influence
Hardware matters: Nintendo 64, Super Famicom, and Game Boy Advance tend toward higher BPMs; NES consistently produces lower averages — likely a reflection of hardware constraints shaping compositional style.

---

## Model Results

| Metric | Score |
|--------|-------|
| Accuracy | **20.19%** |
| Best category (F1) | Forest Songs (0.27) |
| Worst category (F1) | Jungle Songs (0.00) |

The model struggled significantly. Main reasons identified:

1. **Estimated features** — many MIDI files lacked metadata, forcing pitch/tempo estimation that may not reflect the original music
2. **Class imbalance** — some categories had far fewer songs (e.g., Jungle: 13 test samples vs Aquatic: 55)
3. **Limited feature set** — BPM + key alone cannot capture the full musical complexity needed for reliable classification

---

## Limitations & Future Work

- Add rhythmic density and note sustain as features to better capture the original hypothesis
- Apply SMOTE or other resampling techniques to address class imbalance
- Use audio-based features (spectral centroid, MFCC) instead of relying on MIDI metadata estimates
- Explore neural approaches (LSTM, CNN on piano rolls) for richer musical representation