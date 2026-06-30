# Human Activity Recognition — Dimensionality Reduction without Losing Accuracy

Classifying six daily activities (walking, walking upstairs/downstairs, sitting, standing, laying) from smartphone accelerometer and gyroscope signals, and showing that the dataset's **561 engineered features can be cut to a fraction** while keeping classification accuracy high.

`machine-learning`, `human-activity-recognition`, `dimensionality-reduction`, `svd`, `random-forest`, `neural-network`, `scikit-learn`, `python`

> Individual project — UNSW, MATH5836 Data and Machine Learning (2025 T2).

## The question
High-dimensional sensor data is expensive to store and model, and prone to overfitting. So: *how few features do we actually need to classify activity reliably?* This project answers it empirically, comparing two reduction strategies across two model families.

## Dataset
The **UCI/OpenML Human Activity Recognition (HAR)** dataset — 10,299 samples, **561 features**, 6 activity classes, collected from 30 subjects wearing a waist-mounted smartphone. It loads automatically from OpenML (no manual download needed):

```python
from sklearn.datasets import fetch_openml
har = fetch_openml(name="har", version=1, as_frame=True)
X, y = har.data, har.target.astype(int)
```

## Approach
1. **Preprocessing & EDA** — checked for missing values and outliers, examined class balance, and standardised features (per experiment, to avoid leakage).
2. **Feature insight** — ranked features by P-/F-test correlation with the target and via XGBoost permutation importance to surface the top contributors.
3. **Models** — a Multilayer Perceptron (3 hidden layers, >10 neurons each) and a Random Forest classifier.
4. **Robust evaluation** — 5 independent runs with different random seeds, 80/20 train/test split each, reporting **mean ± std** of Accuracy, F1, and ROC-AUC.
5. **Dimensionality reduction** — two strategies, each swept over k = [100, 200, 300, 400, 500]:
   - **Correlation-based** — keep the k features most correlated (absolute Pearson) with the target.
   - **SVD-based** — project standardised data onto the top k eigenvectors of the covariance/correlation matrix.

## Results
- **Full feature set:** both MLP and Random Forest reached **~98% accuracy**, with strong F1 and ROC-AUC.
- **After reduction:** accuracy held **above ~93.6%** for both methods across the full k range — substantial dimensionality cuts with little performance loss.
- **Takeaway:** most of the predictive signal lives in a compact subspace; you can drop a large share of the 561 features and still classify activity reliably, which matters for on-device efficiency.

## Repository structure
```
human-activity-recognition/
├── README.md
├── PROJECT_DESCRIPTION.md     # full write-up: theory, methods, results, discussion
├── HAR_dataset.csv            #  CSV file
├── notebooks/
│   └── har_analysis.ipynb
└── images/                    # plots used in the docs
```

## Run it
```bash
git clone https://github.com/Arun-AK03/human-activity-recognition.git
cd human-activity-recognition
pip install -r requirements.txt
jupyter notebook notebooks/har_analysis.ipynb
```
The notebook fetches the data from OpenML automatically. To save a local CSV copy, run `python download_data.py`.

## Tech stack
Python · scikit-learn · TensorFlow/Keras · XGBoost · NumPy · pandas · Matplotlib/Seaborn · Jupyter

## Dataset credit
This project uses the publicly available HAR dataset. Full credit to the original authors:

> Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* 21st European Symposium on Artificial Neural Networks (ESANN 2013), Bruges, Belgium.

Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) · [OpenML (id 1478)](https://www.openml.org/d/1478). Released to the public domain.

## Author
**Arun Kumar Selvaraj** — MSc Business Data Science & Decisions, UNSW
📧 arunsp2003@gmail.com
