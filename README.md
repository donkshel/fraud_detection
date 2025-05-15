# Credit Card Fraud Detection using Machine Learning

## Project Overview
This repository contains the implementation of machine learning models to detect fraudulent credit card transactions. The project compares Random Forest and Deep Neural Network approaches, with a focus on addressing the class imbalance problem inherent in fraud detection datasets.

## Authors
- CT204/109434/22: ONYANGO GEORGE
- CT204/109397/22: FAITH NYAMOSI
- CT204/109430/22: MELCKZEDEK KIRYA
- CT204/109433/22: SHELDON KENYANI
- CT204/109420/22: VICTOR CHERUIYOT
- CT204/109423/22: PURITY KEMBOI

*Meru University of Science and Technology - Bachelor of Science in Data Science*

## Problem Statement
Credit card fraud poses significant challenges to financial institutions and consumers, leading to substantial financial losses. Traditional fraud detection methods often struggle with evolving fraud patterns and suffer from high false-positive rates. This project aims to develop machine learning models that can effectively identify fraudulent transactions while minimizing false positives.

## Dataset
We use a publicly available credit card transaction dataset from Kaggle containing transactions labeled as fraudulent or legitimate. The dataset features:
- Anonymized transaction features (V1-V28) transformed using PCA
- Transaction amount
- Time elapsed between transactions
- Class label (1 for fraud, 0 for legitimate)

The dataset exhibits significant class imbalance with fraudulent transactions representing only about 0.1% of all transactions.

## Repository Structure
```
├── data/
│   └── creditcard.csv          # Dataset (not included in repo due to size)
├── notebooks/
│   ├── 1_EDA.ipynb             # Exploratory Data Analysis
│   ├── 2_Data_Preprocessing.ipynb  # Data preprocessing and preparation
│   ├── 3_Random_Forest.ipynb   # Random Forest implementation
│   └── 4_Neural_Network.ipynb  # Deep Neural Network implementation
├── src/
│   ├── data_preprocessing.py   # Functions for data preparation
│   ├── model_evaluation.py     # Model evaluation utilities
│   ├── random_forest.py        # Random Forest implementation
│   └── neural_network.py       # Neural Network implementation
├── models/                     # Saved model files
├── results/                    # Model performance results and visualizations
├── requirements.txt            # Project dependencies
└── README.md                   # Project overview
```

## Key Features
- **Data Preprocessing**: Handling of class imbalance using SMOTE, feature scaling, and transformation
- **Random Forest Classifier**: Implementation of an optimized Random Forest model
- **Deep Neural Network**: Implementation of a multi-layer neural network using TensorFlow/Keras
- **Model Evaluation**: Comprehensive evaluation using metrics appropriate for imbalanced datasets (precision, recall, F1-score, AUPRC)
- **Comparative Analysis**: Direct comparison between traditional machine learning and deep learning approaches

## Results Summary
- The Random Forest model achieved 100% recall and 72% precision for fraud detection
- The Neural Network achieved 89% recall and 25% precision
- Random Forest produced significantly fewer false positives while detecting all fraudulent transactions
- SMOTE significantly improved model performance, particularly for the minority fraud class

## Requirements and Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle and place in the data directory
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

## Usage
1. Run the notebooks in sequence for a step-by-step walkthrough of the project
2. Alternatively, use the Python modules in the `src` directory to:
   ```python
   # Example code
   from src.data_preprocessing import preprocess_data
   from src.random_forest import train_random_forest
   
   # Load and preprocess data
   X_train, X_test, y_train, y_test = preprocess_data('data/creditcard.csv')
   
   # Train and evaluate Random Forest model
   rf_model, rf_results = train_random_forest(X_train, y_train, X_test, y_test)
   ```

## Future Work
- Implement real-time fraud detection system
- Explore additional advanced algorithms (XGBoost, LightGBM)
- Investigate feature importance and interpretability techniques
- Develop ensemble methods combining multiple models

## Deployment
App was deployed and hosted to streamlit via the link https://fraudapprevised.streamlit.app/
## License
This project is submitted as part of academic requirements for the Bachelor of Science in Data Science at Meru University of Science and Technology.

## Acknowledgments
- We thank our supervisors and advisors for their guidance
- Special thanks to the UCI Machine Learning Repository and Kaggle for providing the dataset
