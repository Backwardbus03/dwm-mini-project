import pandas as pd  # dataframes and csv IO
import joblib  # save/load sklearn pipelines
import os  # filesystem helpers
from sklearn.model_selection import train_test_split  # train/test split helper
from sklearn.ensemble import RandomForestClassifier  # our model
from sklearn.preprocessing import OneHotEncoder  # encode categorical vars
from sklearn.compose import ColumnTransformer  # apply transformers to specific cols
from sklearn.pipeline import Pipeline  # chain preprocessing + model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix  # evaluation helpers

print("=" * 80)
print("Q-CHAT-10 TODDLER MODEL TRAINING")
print("=" * 80)

# ensure models/ exists so we can write the trained pipeline later
os.makedirs('models', exist_ok=True)

# Load toddler dataset (expects the CSV to be in data/)
print("\nLoading Toddler-Autism-dataset-July-2018.csv...")
df_toddler = pd.read_csv('data/Toddler-Autism-dataset-July-2018.csv')

print(f"Dataset shape: {df_toddler.shape}")
print(f"Columns: {df_toddler.columns.tolist()}")

# Clean target column: strip stray spaces and make a 0/1 target
df_toddler['Class/ASD Traits'] = df_toddler['Class/ASD Traits '].str.strip()
df_toddler['Class_Binary'] = (df_toddler['Class/ASD Traits'] == 'Yes').astype(int)

print(f"\nTarget distribution:")
print(df_toddler['Class/ASD Traits'].value_counts())
print(f"\nClass balance:")
for label, count in df_toddler['Class/ASD Traits'].value_counts().items():
    # show basic class balance so we can spot heavy imbalance early
    print(f"  {label}: {count} ({count/len(df_toddler)*100:.1f}%)")

# Define which columns are features we will feed to the model
feature_columns = [
    'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
    'Age_Mons', 'Sex', 'Jaundice', 'Family_mem_with_ASD'
]

# categorical vs numerical split for preprocessing
categorical_features = ['Sex', 'Jaundice', 'Family_mem_with_ASD']
numerical_features = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons']

# slice out X and y
X = df_toddler[feature_columns].copy()
y = df_toddler['Class_Binary']

# lowercase and trim categorical text so OneHotEncoder doesn't get weird duplicates
print("\nCleaning categorical features...")
X['Sex'] = X['Sex'].str.lower().str.strip()
X['Jaundice'] = X['Jaundice'].str.lower().str.strip()
X['Family_mem_with_ASD'] = X['Family_mem_with_ASD'].str.lower().str.strip()

print(f"\nAge range (months): {X['Age_Mons'].min()} - {X['Age_Mons'].max()}")
print(f"Age range (years): {X['Age_Mons'].min()/12:.1f} - {X['Age_Mons'].max()/12:.1f}")

# Preprocessing: one-hot encode the small set of categorical columns, leave others as-is
print("\nCreating preprocessing pipeline...")
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
         categorical_features)
    ],
    remainder='passthrough'
)

# Build the sklearn Pipeline: preprocessing followed by classifier
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,           # 100 decision trees
        max_depth=10,                # Max tree depth to prevent overfitting
        min_samples_split=5,         # Min samples to split a node
        min_samples_leaf=2,          # Min samples in leaf node
        random_state=42,             # For reproducibility
        class_weight='balanced'      # Handle class imbalance
    ))
])

# Split data (80% train, 20% test) with stratification
print("\nSplitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Training set class distribution: {y_train.value_counts().to_dict()}")
print(f"Test set class distribution: {y_test.value_counts().to_dict()}")

# Train the pipeline on the training split
print("\n" + "=" * 80)
print("TRAINING MODEL...")
print("=" * 80)
model_pipeline.fit(X_train, y_train)
print("✅ Training complete!")

# Evaluate on the hold-out test set
print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No ASD', 'ASD']))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nTrue Negatives: {cm[0][0]}")
print(f"False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}")
print(f"True Positives: {cm[1][1]}")

# Feature importance analysis (grab the post-preprocessing feature names)
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE")
print("=" * 80)

# reconstruct feature names: one-hot output for categoricals + numerical feature names
feature_names = []
cat_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
feature_names.extend(cat_feature_names)
feature_names.extend(numerical_features)

# extract importance values from trained RandomForest
importances = model_pipeline.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
for idx, row in importance_df.head(10).iterrows():
    print(f"  {row['feature']:<25} {row['importance']:.6f}")

# Save model pipeline to models/ so app.py can load it
model_path = 'models/toddler_qchat_model.joblib'
joblib.dump(model_pipeline, model_path)
print(f"\n✅ Model saved to: {model_path}")

# Save feature importance CSV for dashboard/visualization
importance_path = 'models/toddler_qchat_feature_importance.csv'
importance_df.to_csv(importance_path, index=False)
print(f"✅ Feature importance saved to: {importance_path}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\nModel Performance Summary:")
print(f"  - Samples: {len(df_toddler)}")
print(f"  - Features: {len(feature_columns)}")
print(f"  - Test Accuracy: {accuracy:.4f}")
print(f"  - Model: Random Forest (100 estimators, max_depth=10)")
print(f"  - Saved to: {model_path}")
print("\nReady to use for Q-CHAT-10 toddler screening!")
print("=" * 80)
