# Loan Approval Prediction — Data Mining Project

**IT326 · Data Mining · King Saud University**

Predicting whether a loan application will be approved, and discovering groups of similar applicants, using **Decision Tree classification** and **K-means clustering** on the Dream Housing Finance dataset.

[![Open Phase 3 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Engfatiimah/Minig_Project/blob/main/Reports/Phase3.ipynb)

---

## Table of Contents
- [Problem](#problem)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [How to Run](#how-to-run)
- [Team](#team)
- [References](#references)

---

## Problem
Loan approval is usually done manually by reviewing each applicant's details, which makes the process slow, inconsistent and open to bias. A wrong decision is costly: approving a high-risk applicant can lead to financial loss, while rejecting a reliable applicant means losing a customer.

This project applies data mining to make loan decisions more consistent and to understand the factors behind them.

## Objectives
- **Predict approval:** classify applications as Approved or Not Approved.
- **Find key attributes:** identify which attributes influence the decision most.
- **Group applicants:** cluster applicants with similar profiles.

## Dataset
| | |
|---|---|
| **Source** | Kaggle — [Loan Prediction Dream Housing Finance](https://www.kaggle.com/code/ufffnick/loan-prediction-dream-housing-finance/input) |
| **Records** | 614 applicants |
| **Attributes** | 13 |
| **Class label** | `Loan_Status` — Y (Approved) / N (Not Approved) |
| **Class distribution** | Y: 422 (68.73%) · N: 192 (31.27%) — moderately imbalanced |

<p align="center">
  <img width="50%" alt="Class label distribution" src="https://github.com/user-attachments/assets/caf27f00-4f75-4a93-ae5f-3a21841effa1" />
</p>

<details>
<summary><b>Attribute description</b></summary>

| Attribute | Description | Type |
|---|---|---|
| Loan_ID | Unique loan identifier | Nominal (ID) |
| Gender | Male / Female | Binary |
| Married | Applicant married (Yes / No) | Binary |
| Dependents | Number of dependents (0, 1, 2, 3+) | Ordinal |
| Education | Graduate / Not Graduate | Binary |
| Self_Employed | Self-employed (Yes / No) | Binary |
| ApplicantIncome | Applicant income | Numeric |
| CoapplicantIncome | Co-applicant income | Numeric |
| LoanAmount | Loan amount (in thousands) | Numeric |
| Loan_Amount_Term | Loan term (in months) | Numeric |
| Credit_History | Credit history meets guidelines (1 / 0) | Binary |
| Property_Area | Urban / Semiurban / Rural | Nominal |
| Loan_Status | Loan approved (Y / N) | Class label |

</details>


## Methodology

### 1. Data Analysis
Statistical summary, missing values analysis, outlier detection (boxplots and IQR), histograms, bar plots, scatter plots and class label distribution.

### 2. Data Preprocessing
| Step | What was done |
|---|---|
| Missing values | Mode for categorical and discrete attributes, median for `LoanAmount` (skewed, with outliers) |
| Normalization | Min-Max scaling of `ApplicantIncome`, `CoapplicantIncome` and `LoanAmount` to [0, 1] |
| Discretization | Equal-frequency binning of `LoanAmount` into Low / Medium / High |
| Encoding | Binary attributes to 0/1, `Dependents` "3+" to 3, one-hot encoding of `Property_Area`, `Loan_Status` Y = 1, N = 0 |
| Attribute selection | `Loan_ID` (identifier) and `LoanAmount_Category` (redundant) excluded from the models |

All 614 records were kept.

### 3. Classification — Decision Tree
- Holdout method with stratified sampling: **90/10, 80/20 and 70/30** training/testing splits.
- Two attribute selection measures: **Gini index** and **Information Gain (Entropy)**.
- Evaluated with the confusion matrix and four measures: **accuracy, sensitivity (recall), specificity and precision**, with Approved as the positive class.

### 4. Clustering — K-means
- Class label removed before clustering.
- Number of clusters chosen with the **Elbow method (WSS)** and the **Silhouette coefficient**.
- Three values of K compared: **K = 2, 3, 4**.

## Results

### Classification (%)
**Gini index**
| Measure | 90/10 | 80/20 | 70/30 |
|---|---|---|---|
| Accuracy | **77.42** | 74.80 | 68.11 |
| Sensitivity (Recall) | 83.72 | 80.00 | 69.29 |
| Specificity | 63.16 | 63.16 | 65.52 |
| Precision | 83.72 | 82.93 | 81.48 |

**Information Gain (Entropy)**
| Measure | 90/10 | 80/20 | 70/30 |
|---|---|---|---|
| Accuracy | 74.19 | 72.36 | 67.03 |
| Sensitivity (Recall) | 86.05 | 74.12 | 70.87 |
| Specificity | 47.37 | 68.42 | 58.62 |
| Precision | 78.72 | 84.00 | 78.95 |

**Best model: Gini, 90/10 split** — the highest accuracy and the most balanced results across the four measures.

<p align="center">
  <img width="100%" alt="Decision tree (Gini, 90/10), first three levels" src="https://github.com/user-attachments/assets/ec126d8a-34a4-4c16-ae1d-922690ecfe67" />
</p>

The first split is on **Credit_History**: applicants whose credit history does not meet the guidelines (≤ 0.5) are mostly predicted as Not Approved.

### Clustering
| | K = 2 | K = 3 | K = 4 |
|---|---|---|---|
| Average silhouette width | 0.2241 | **0.3427** | 0.3042 |
| Total within-cluster sum of squares (WSS) | 794.30 | 577.14 | 530.95 |

**Optimal number of clusters: K = 3**, based on the majority rule (Elbow method and Silhouette coefficient).

<p align="center">
  <img width="48%" alt="Elbow method" src="https://github.com/user-attachments/assets/f319aaf7-10e1-45df-8ca0-a80c5f7113bb" />
  <!-- silhouette -->
</p>

<!-- clusters -->

Each of the three clusters contains applicants from a single property area: Rural (179), Urban (202) and Semiurban (233).

## Key Findings
- **Credit_History** was the first split in the decision tree, making it the most important attribute for loan approval. This agrees with the research paper [1].
- In all six models, sensitivity was higher than specificity: the models recognize approved applications better than not approved ones, which reflects the class imbalance.
- Gini had higher accuracy in every split, while Entropy (80/20) had the highest specificity (68.42%) and precision (84.00%).
- The three clusters are separated mainly by **Property_Area** (Rural, Urban and Semiurban) rather than by financial profile.
- The research paper's Decision Tree (80/20) reached 81.87% accuracy and 98.76% recall, while ours reached 74.80% accuracy but a higher precision (82.93% vs. 79.65%). The datasets and methods differ (13,508 vs. 614 records), so the results are not directly comparable.

## How to Run
1. Click **Open in Colab** at the top of this page, or open any notebook from the `Reports` folder in [Google Colab](https://colab.research.google.com/).
2. Select **Runtime → Run all**.

The notebooks load the dataset directly from this repository, so no files need to be downloaded.

**Libraries:** pandas, NumPy, Matplotlib, seaborn, scikit-learn.

## Team
- Fatimah Bin Mohammed
- Najd Mohammed Alhargan
- Elaf Alangari
- Lara Alfurayh

## References
[1] K. Gawande, Pallavi, D. Thakur, and P. Saurabh, "An explainable ensemble learning framework for transparent and accurate loan approval prediction," *Discover Artificial Intelligence*, vol. 6, art. no. 1070, 2026, doi: 10.1007/s44163-026-01569-6.

[2] ufffnick, "Loan Prediction Dream Housing Finance," Kaggle. [Online]. Available: https://www.kaggle.com/code/ufffnick/loan-prediction-dream-housing-finance/input
