# Anomaly detection in network traffic :globe_with_meridians:
## Overview
The problem of anomaly detection in network traffic aims to identify unusual data patterns that may indicate cyberattacks. As the number of internet-based services continues to grow, early detection of such anomalies becomes essential for protecting information systems. Fortunately, machine learning methods provide powerful tools for analyzing and classifying network traffic, enabling the recognition of complex patterns and the identification of suspicious behavior even in the absence of predefined signatures.

<ul>
  <li> <b> Goal </b>: The aim of this project is to explore and compare 4 ML models that are extremely popular in anomaly detection: <i> Isolation Forest, K-Nearest Neighbours, SVDD and Autoencoder </I> </li>
  <li> <b> Key features: </b> <ul> <li> Custom implementations and Scikit-learn wrappers </li> <li> Preprocessing pipeline for imbalanced data </li> <li> Voting methods for improved accuracy </li> </ul> </li>
  <li> <b> Used dataset: </b> UNSW-NB15 </li>
</ul>

## Installation 
```
git clone https://github.com/Ioana05/Anomaly_detection_in_network_traffic.git
cd Anomaly_detection_in_network_traffic/Script  
pip install -r requirements.txt
```

## Project structure :file_folder:
```
Script/  
├── Models/  
│   ├── initial_implementation/  # initial implementation for all the models  
│   └── sklearn_standard/        # Scikit-learn wrappers for the models + voting methods
│  
├── pipeline.py                  # Data preprocessing  
├── requirements.txt             # Dependencies  
Datasets/                        # UNSW-nb15 training and testing datasets 
```
