# Credit Card Anomaly Detection with DBSCAN

This project implements an anomaly detection system for credit card transactions using the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm. It identifies unusual transactions that deviate significantly from typical spending patterns, marking them as outliers.

## Features

* **Data Loading**: Reads transaction data from a CSV file (`creditcard.csv`).
* **Feature Selection**: Focuses on relevant features such as `Time`, `Amount`, and anonymized features (`V1` to `V4`).
* **Data Preprocessing**: Scales `Time` and `Amount` features using `StandardScaler` to ensure all features contribute equally to the clustering process.
* **DBSCAN Clustering**: Applies DBSCAN to identify dense clusters of normal transactions and label sparse points as outliers.
* **Performance Metrics**: Calculates and displays clustering metrics such as Silhouette Score and Davies-Bouldin Index for the clustered data (excluding outliers).
* **Summary Table**: Presents a clear summary of total transactions, clustered transactions, and the number of identified outliers, along with the calculated metrics.
* **Anomaly Identification**: Prints a DataFrame of the transactions identified as anomalies (outliers).
* **Visualization**: Generates a matplotlib table to visualize the clustering summary.

## Requirements

To run this script, you need the following Python libraries:

* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`

You can install these libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib
