# EEG-Complexity-VisibilityGraphs-MDD

##  Running the Network Pipeline

```bash
pip install -r requirements.txt
python main_pipeline.py
```

This repository contains the analysis code for the study:

**“EEG-Based Hidden Topographical Changes in Depression Using Complex Network Dynamics”**

The work investigates altered brain network organization in **Major Depressive Disorder (MDD)** using EEG signal complexity, visibility graph–based functional networks, and hub topology analysis.

---

##  Analysis Overview

The pipeline is modular and follows the workflow described in the paper:
EEG Preprocessing 
→ Epoching
→ Signal Complexity (Hurst Exponent)
→ Statistical Testing (Permutation + FDR)
→ Visibility Graph Construction
→ Network Metrics & Hub Classification
→ Frequency-Specific Network Analysis
→ Visualization
All network measures are computed at the **epoch level**.  
Group-level aggregation and statistics are performed in separate scripts.

---

##  Data Structure

The code expects EEG epochs organized as:

data/<br>
├── mdd/<br>
│ └── subject_X/channel_Y/epoch_Z.csv<br>
└── normal/<br>
└── subject_X/channel_Y/epoch_Z.csv<br>

For each subject and channel, EEG recordings are segmented into epochs.
Each epoch is stored as a separate CSV file containing a one-dimensional
time series (amplitude vs. time) for that channel.<br>
Each epoch file contains the EEG time-series signal for a single channel
during one epoch (10-second segment sampled at 250 Hz).

> **Note:** Raw EEG data (~4 GB) is not included due to licensing constraints.

---

##  Key Components

- **Signal Complexity:** Hurst exponent (multi-scale R/S analysis)
- **Statistical Testing:** Channel-wise permutation test with FDR correction
- **Network Construction:** Natural Visibility Graphs (NVG)
- **Network Metrics:** Degree, clustering coefficient, modularity, participation coefficient, eigenvector centrality
- **Hub Classification:** Node roles R1–R7 (Guimerà & Amaral framework)
- **Frequency Analysis:** Theta (4–7.5 Hz), Alpha (8–12 Hz), Beta (13–30 Hz)
- **PSD:** Welch-based PSD used for supporting frequency interpretation (not statistical inference)

---
## Figures
<img width="500" alt="Pipeline overview" src="https://github.com/user-attachments/assets/3bfa6ab2-6c30-47cd-ad1e-3e9a59540de4" />
<br> <b>Fig.1</b> Region-wise comparative analysis plots of (a) Power Spectral Density (PSD), (b) hurst exponent, and (c)
Connector Hubs (R6) among Major Depressive Disorder (MDD) and Healthy Control (HC) groups.<br>

<img width="500" alt="Pipeline overview" src="https://github.com/user-attachments/assets/d0b24d9a-4cf1-4c4f-888b-5e9d6f190c01" />
<br>
<b>Fig.2</b> Topographical layout of 128 EEG electrodes across the scalp, divided into frontal, parietal, temporal,
and occipital regions. Electrodes highlighted in green represent the significant channels identified for further
network analysis.
<br>

<img width="500" alt="Pipeline overview" src="https://github.com/user-attachments/assets/65c16182-702a-4677-8167-5a69aa1bd31c" />
<br> <b>Fig.3</b> PSD computed across significant electrodes for each subject. A) Whole EEG full-spectrum PSD averaged
over selected channels. B) Zoomed-in view of the Alpha band (8–12 Hz). C) Beta band (13–30 Hz). D) Theta
band (4–7.5Hz). Variance across subjects is represented as shaded region, highlighting inter-subject differences
in power across frequency bands.
<br>
<img width="500" alt="Pipeline overview" src="https://github.com/user-attachments/assets/8828a798-bf4e-460d-a8ad-05ce51e6c275" />
<br> <b>Fig.4</b> Comparative Analysis of hubs in the whole EEG and different frequency bands (alpha, beta, theta) for
MDD and Healthy Control (HC) subjects. Each panel represents a type of hub as follows: A) Provincial hubs
(R5), B) Connector hubs (R6), and C) Kinless hubs (R7).


