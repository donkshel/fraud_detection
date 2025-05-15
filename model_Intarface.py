import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report, 
                            roc_curve, auc, precision_recall_curve)
import seaborn as sns

# Load models and components
@st.cache_resource
def load_components():
    model = joblib.load('fraud_detection_rf.pkl')
    scaler = joblib.load('robust_scaler.pkl')
    return model, scaler

model, scaler = load_components()

# Load the actual metrics
@st.cache_resource
def load_metrics():
    with open('random_forest_metrics.json', 'r') as f:
        rf_metrics = json.load(f)
    return rf_metrics

metrics = load_metrics()

# App title and description
st.title('Credit Card Fraud Detection (Random Forest)')
st.write("""
Detect fraudulent transactions using a trained Random Forest model.
Choose between uploading a CSV file or entering transaction details manually.
""")

# Evaluation metrics visualization
st.sidebar.header('Model Evaluation')
show_metrics = st.sidebar.checkbox('Show Model Evaluation Metrics', True)

if show_metrics:
    st.header('Model Performance Metrics')
    
    import json
import numpy as np

# Load the actual metrics
@st.cache_resource
def load_metrics():
    with open('random_forest_metrics.json', 'r') as f:
        rf_metrics = json.load(f)
    return rf_metrics

metrics = load_metrics()

if show_metrics:
    st.header('Model Performance Metrics (Test Set)')
    
    # Confusion Matrix
    st.subheader('Confusion Matrix')
    cm = np.array(metrics['confusion_matrix'])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legit', 'Fraud'], 
                yticklabels=['Legit', 'Fraud'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)
    
    # ROC Curve
    st.subheader('ROC Curve')
    fpr = metrics['roc_curve']['fpr']
    tpr = metrics['roc_curve']['tpr']
    roc_auc = metrics['roc_curve']['auc']
    fig, ax = plt.subplots()
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    st.pyplot(fig)
    
    # Precision-Recall Curve
    st.subheader('Precision-Recall Curve')
    precision = metrics['precision_recall_curve']['precision']
    recall = metrics['precision_recall_curve']['recall']
    fig, ax = plt.subplots()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    st.pyplot(fig)
    
    # Classification Report
    st.subheader('Classification Report')
    report = metrics['classification_report']
    st.write("""
    | Metric | Legit | Fraud | Accuracy |
    |--------|-------|-------|----------|
    | Precision | {:.2f} | {:.2f} | - |
    | Recall | {:.2f} | {:.2f} | - |
    | F1-Score | {:.2f} | {:.2f} | {:.2f} |
    """.format(
        report['0']['precision'], report['1']['precision'],
        report['0']['recall'], report['1']['recall'],
        report['0']['f1-score'], report['1']['f1-score'],
        report['accuracy']
    ))

# Input method selection
input_method = st.radio("Select input method:", 
                       ("Upload CSV file", "Manual input"))

if input_method == "Upload CSV file":
    st.header('Batch Fraud Detection')
    uploaded_file = st.file_uploader("Upload your transactions CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.write(df.head())
        
        if st.button('Detect Fraud'):
            # Preprocess
            df[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']])
            
            # Predict
            predictions = model.predict(df)
            probas = model.predict_proba(df)[:, 1]
            
            # Add results to dataframe
            df['Fraud Probability'] = probas
            df['Fraud Prediction'] = np.where(probas > 0.5, 'Fraud', 'Legit')
            
            st.success("Fraud detection complete!")
            st.write("Results:")
            st.write(df)
            
            # Download results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Results",
                csv,
                "fraud_predictions.csv",
                "text/csv",
                key='download-csv'
            )
            
            # Show fraud distribution
            fraud_count = df['Fraud Prediction'].value_counts()
            st.subheader('Fraud Distribution')
            fig, ax = plt.subplots()
            ax.pie(fraud_count, labels=fraud_count.index, autopct='%1.1f%%')
            st.pyplot(fig)

else:  # Manual input
    st.header('Single Transaction Evaluation')
    
    # Create input fields for all features
    st.subheader('Transaction Features')
    col1, col2 = st.columns(2)
    
    with col1:
        Time = st.number_input('Time (seconds since first transaction)', value=0)
        Amount = st.number_input('Amount', value=0.0)
        V1 = st.number_input('V1', value=0.0)
        V2 = st.number_input('V2', value=0.0)
        V3 = st.number_input('V3', value=0.0)
        V4 = st.number_input('V4', value=0.0)
        V5 = st.number_input('V5', value=0.0)
        V6 = st.number_input('V6', value=0.0)
        V7 = st.number_input('V7', value=0.0)
        V8 = st.number_input('V8', value=0.0)
        V9 = st.number_input('V9', value=0.0)
        V10 = st.number_input('V10', value=0.0)
        
    with col2:
        V11 = st.number_input('V11', value=0.0)
        V12 = st.number_input('V12', value=0.0)
        V13 = st.number_input('V13', value=0.0)
        V14 = st.number_input('V14', value=0.0)
        V15 = st.number_input('V15', value=0.0)
        V16 = st.number_input('V16', value=0.0)
        V17 = st.number_input('V17', value=0.0)
        V18 = st.number_input('V18', value=0.0)
        V19 = st.number_input('V19', value=0.0)
        V20 = st.number_input('V20', value=0.0)
        V21 = st.number_input('V21', value=0.0)
    
    if st.button('Evaluate Transaction'):
        # Create feature dictionary
        features = {
            'Time': Time,
            'V1': V1, 'V2': V2, 'V3': V3, 'V4': V4, 'V5': V5,
            'V6': V6, 'V7': V7, 'V8': V8, 'V9': V9, 'V10': V10,
            'V11': V11, 'V12': V12, 'V13': V13, 'V14': V14, 'V15': V15,
            'V16': V16, 'V17': V17, 'V18': V18, 'V19': V19, 'V20': V20,
            'V21': V21, 'V22': 0, 'V23': 0, 'V24': 0, 'V25': 0,
            'V26': 0, 'V27': 0, 'V28': 0, 'Amount': Amount
        }
        
        # Convert to DataFrame and preprocess
        input_df = pd.DataFrame([features])
        input_df[['Time', 'Amount']] = scaler.transform(input_df[['Time', 'Amount']])
        
        # Predict
        proba = model.predict_proba(input_df)[0][1]
        prediction = proba > 0.5
        
        # Display results
        st.subheader('Results')
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Fraud Probability", f"{proba:.2%}")
        
        with col2:
            if prediction:
                st.error("Prediction: FRAUD 🚨")
            else:
                st.success("Prediction: LEGIT ✅")
        
        # Show feature importance (if available)
        try:
            st.subheader('Top Influential Features')
            importance = model.feature_importances_
            features = input_df.columns
            importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
            importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
            
            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            st.pyplot(fig)
        except:
            pass

# Add some info about the model
st.sidebar.header('Model Information')
st.sidebar.write("""
- **Algorithm**: Random Forest
- **Classes**: Legit (0), Fraud (1)
- **Threshold**: 0.5 probability
""")

# Add a way to adjust threshold
threshold = st.sidebar.slider('Adjust fraud threshold', 0.01, 0.99, 0.5)
st.sidebar.write(f"Transactions with >{threshold:.0%} probability will be flagged as fraud.")