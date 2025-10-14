import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load the dataset - FIXED PATH
print("Loading Autism_Screening_Data_Combined.csv...")
df = pd.read_csv('data/Autism_Screening_Data_Combined.csv')  # ✅ FIXED: Added .csv extension and removed /data/

print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['Class'].value_counts()}")

# Preprocess the data
# Rename columns to match app.py expectations
df = df.rename(columns={
    'A1': 'A1_Score', 'A2': 'A2_Score', 'A3': 'A3_Score', 'A4': 'A4_Score', 'A5': 'A5_Score',
    'A6': 'A6_Score', 'A7': 'A7_Score', 'A8': 'A8_Score', 'A9': 'A9_Score', 'A10': 'A10_Score',
    'Age': 'age', 'Sex': 'gender', 'Jauundice': 'jaundice', 'Family_ASD': 'austim'
})

# Convert target to binary
df['Class/ASD'] = (df['Class'] == 'YES').astype(int)

# Clean categorical features
df['gender'] = df['gender'].str.lower().str.strip()
df['jaundice'] = df['jaundice'].str.lower().str.strip()
df['austim'] = df['austim'].str.lower().str.strip()

# Create age groups
def categorize_age(age):
    if age <= 11:
        return 'child'
    elif age <= 17:
        return 'adolescent'
    else:
        return 'adult'

df['age_group'] = df['age'].apply(categorize_age)

# Define features and target
feature_columns = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age', 'gender', 'jaundice', 'austim'
]

categorical_features = ['gender', 'jaundice', 'austim']
numerical_features = ['age'] + [f'A{i}_Score' for i in range(1, 11)]

# Define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Train models for each age group
age_groups = ['child', 'adolescent', 'adult']
results = {}

for group in age_groups:
    print(f"\n{'='*70}")
    print(f"Training model for {group.upper()} age group...")
    print(f"{'='*70}")
    
    # Filter data for this age group
    group_df = df[df['age_group'] == group].copy()
    
    if len(group_df) < 50:
        print(f"⚠️ Warning: Only {len(group_df)} samples for {group} group. Skipping...")
        continue
    
    X = group_df[feature_columns].copy()
    y = group_df['Class/ASD']
    
    print(f"Samples: {len(X)}")
    print(f"ASD Distribution: {y.value_counts().to_dict()}")
    
    # Create model pipeline
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("Training model...")
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Get feature importance
    feature_names = []
    cat_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names.extend(cat_feature_names)
    feature_names.extend(numerical_features)
    
    importances = model_pipeline.named_steps['classifier'].feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Important Features:")
    print(importance_df.head(10))
    
    # Save model and importance
    model_path = f'models/{group}_model.joblib'
    importance_path = f'models/{group}_feature_importance.csv'
    
    joblib.dump(model_pipeline, model_path)
    importance_df.to_csv(importance_path, index=False)
    
    print(f"✅ Saved: {model_path}")
    print(f"✅ Saved: {importance_path}")
    
    results[group] = {
        'accuracy': accuracy,
        'samples': len(X),
        'asd_positive': y.sum(),
        'asd_negative': (y == 0).sum()
    }

print(f"\n{'='*70}")
print("TRAINING COMPLETE!")
print(f"{'='*70}")
print("\nSummary of all models:")
for group, metrics in results.items():
    print(f"\n{group.upper()}:")
    print(f"  Samples: {metrics['samples']}")
    print(f"  ASD Positive: {metrics['asd_positive']}")
    print(f"  ASD Negative: {metrics['asd_negative']}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")

print(f"\n{'='*70}")
print("All models saved to 'models/' directory!")
print(f"{'='*70}")
