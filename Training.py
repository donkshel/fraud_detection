import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import joblib
import json


# Load the dataset
df = pd.read_csv('creditcard.csv')

# Show basic info
print("Shape of dataset:", df.shape)
print("\nColumns:", df.columns.tolist())

# Display first 5 rows
df.head()

#visualize the class imbalance


df['Class'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Class Distribution')
plt.xticks(ticks=[0, 1], labels=['Non-Fraud (0)', 'Fraud (1)'], rotation=0)
plt.ylabel('Number of Transactions')
plt.show()

df.describe()

# Create copies to preserve original
scaled_df = df.copy()

# Initialize the scaler
scaler = RobustScaler()

# Apply scaling to 'Time' and 'Amount'
scaled_df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

# Features (X) and target (y)
X = scaled_df.drop('Class', axis=1)
y = scaled_df['Class']

# Initial split (before resampling)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define resampling steps
over = SMOTE(sampling_strategy=0.1, random_state=42)  # Minority class = 10% of majority
under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)  # Majority class = 2x minority

# Combine using pipeline
resample_pipeline = Pipeline(steps=[('o', over), ('u', under)])

# Apply to training set only
X_resampled, y_resampled = resample_pipeline.fit_resample(X, y)


# Base model
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

# Minimal grid for quick tuning
param_grid = {
    'n_estimators': [100],
    'max_depth': [None, 10],
    'max_features': ['sqrt']
}

# Grid search with 3-fold CV
grid_search_rf = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2
)

# Train on full resampled data
grid_search_rf.fit(X_resampled, y_resampled)

# Best model
best_rf_model = grid_search_rf.best_estimator_

# Show best params
print("Best RF parameters:", grid_search_rf.best_params_)


# Get input dimension
input_dim = X_resampled.shape[1]

# Build the model
dnn_model = Sequential([
    Dense(128, input_dim=input_dim, activation='relu', 
          kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile
dnn_model.compile(optimizer='adam', 
                 loss='binary_crossentropy', 
                 metrics=[tf.keras.metrics.AUC(name='auc_pr', curve='PR')])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Fit the model
history = dnn_model.fit(
    X_resampled, y_resampled,
    validation_split=0.2,
    epochs=20,
    batch_size=256,
    callbacks=[early_stop],
    verbose=2
)

# Evaluate the model on the test set

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if len(y_pred.shape) > 1:  # For neural network
        y_pred = (y_pred > 0.5).astype(int)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        print(f"AUPRC: {average_precision_score(y_test, y_proba):.4f}")

# Use for both models
print("Random Forest Evaluation:")
evaluate_model(best_rf_model, X_test, y_test)

print("\nNeural Network Evaluation:")
evaluate_model(dnn_model, X_test, y_test)

# visualizatuon of the model features and performance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    ConfusionMatrixDisplay
)

def plot_all_metrics(model, X_test, y_test, model_name="Model"):
    if hasattr(model, 'predict_proba'):
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_probs = model.predict(X_test).flatten()
    
    y_pred = (y_probs > 0.5).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend()
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curve')
    plt.legend()
    plt.show()

# Evaluate both models
plot_all_metrics(best_rf_model, X_test, y_test, "Random Forest")
plot_all_metrics(dnn_model, X_test, y_test, "Neural Network")

# Random Forest Feature Importance
importances = best_rf_model.feature_importances_
feature_names = X.columns
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
plt.title('Top 10 Feature Importances - Random Forest')
plt.tight_layout()
plt.show()

# Save all components
joblib.dump(best_rf_model, 'fraud_detection_rf.pkl')
joblib.dump(scaler, 'robust_scaler.pkl')

# Function to save metrics to a JSON file   
def save_metrics(model, X_test, y_test, model_name):
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'roc_curve': {
            'fpr': roc_curve(y_test, y_proba)[0].tolist(),
            'tpr': roc_curve(y_test, y_proba)[1].tolist(),
            'auc': auc(roc_curve(y_test, y_proba)[0], roc_curve(y_test, y_proba)[1])
        },
        'precision_recall_curve': {
            'precision': precision_recall_curve(y_test, y_proba)[0].tolist(),
            'recall': precision_recall_curve(y_test, y_proba)[1].tolist()
        }
    }
    
    # Save to file
    with open(f'{model_name}_metrics.json', 'w') as f:
        json.dump(metrics, f)

# Save metrics for both models
save_metrics(best_rf_model, X_test, y_test, 'random_forest')

