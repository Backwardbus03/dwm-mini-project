import pandas as pd  # dataframes for loading and slicing csv data
import joblib  # save/load trained sklearn pipelines
import os  # filesystem helpers
from sklearn.model_selection import train_test_split  # split data for train/test
from sklearn.ensemble import RandomForestClassifier  # the model we train
from sklearn.preprocessing import OneHotEncoder  # encode categorical variables
from sklearn.compose import ColumnTransformer  # apply transforms to specific cols
from sklearn.pipeline import Pipeline  # glue preprocessing + model together
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix  # evaluation metrics

# ensure models/ folder exists so we can dump trained models there
os.makedirs('models', exist_ok=True)

# Load the combined autism screening dataset
print("Loading Autism_Screening_Data_Combined.csv...")
df = pd.read_csv('data/Autism_Screening_Data_Combined.csv')  # dataset lives under data/

print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['Class'].value_counts()}")

# --- Preprocess the raw CSV to match our app's expected column names ---
# rename survey question columns and demographic columns to match code elsewhere
df = df.rename(columns={
    'A1': 'A1_Score', 'A2': 'A2_Score', 'A3': 'A3_Score', 'A4': 'A4_Score', 'A5': 'A5_Score',
    'A6': 'A6_Score', 'A7': 'A7_Score', 'A8': 'A8_Score', 'A9': 'A9_Score', 'A10': 'A10_Score',
    'Age': 'age', 'Sex': 'gender', 'Jauundice': 'jaundice', 'Family_ASD': 'austim'
})

# convert the original Class column (YES/NO) into a binary 0/1 target column
df['Class/ASD'] = (df['Class'] == 'YES').astype(int)

# normalize categorical text fields so later OneHotEncoder behaves
df['gender'] = df['gender'].str.lower().str.strip()
df['jaundice'] = df['jaundice'].str.lower().str.strip()
df['austim'] = df['austim'].str.lower().str.strip()

# small helper: bucket raw ages into simple groups we train separate models for
def categorize_age(age):
    if age <= 11:
        return 'child'
    elif age <= 17:
        return 'adolescent'
    else:
        return 'adult'

df['age_group'] = df['age'].apply(categorize_age)

# features we will use for modeling (questions + simple demographics)
feature_columns = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age', 'gender', 'jaundice', 'austim'
]

# split out which are categorical vs numerical for preprocessing
categorical_features = ['gender', 'jaundice', 'austim']
numerical_features = ['age'] + [f'A{i}_Score' for i in range(1, 11)]

# preprocessing: one-hot encode the small set of categorical vars, pass the rest through
# drop='first' reduces collinearity, handle_unknown avoids crashes if unseen category appears
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# we'll train one model per age group so we can capture different patterns by development stage
age_groups = ['child', 'adolescent', 'adult']
results = {}

for group in age_groups:
    print(f"\n{'='*70}")
    print(f"Training model for {group.upper()} age group...")
    print(f"{'='*70}")

    # restrict the dataset to the current age group
    group_df = df[df['age_group'] == group].copy()

    # safety: skip tiny groups to avoid training broken models
    if len(group_df) < 50:
        print(f"⚠️ Warning: Only {len(group_df)} samples for {group} group. Skipping...")
        continue

    X = group_df[feature_columns].copy()
    y = group_df['Class/ASD']

    print(f"Samples: {len(X)}")
    print(f"ASD Distribution: {y.value_counts().to_dict()}")

    # pipeline: preprocess, then random forest classifier
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'  # compensate class imbalance
        ))
    ])

    # hold-out split for a quick evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # train
    print("Training model...")
    model_pipeline.fit(X_train, y_train)

    # quick evaluation on the test split
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # --- feature importance ---
    # build a human-readable list of feature names that corresponds to the model input
    feature_names = []
    # get generated names from OneHotEncoder for categorical features
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

    # Save model pipeline and feature importance CSV for use in the web app
    model_path = f'models/{group}_model.joblib'
    importance_path = f'models/{group}_feature_importance.csv'

    joblib.dump(model_pipeline, model_path)
    importance_df.to_csv(importance_path, index=False)

    print(f"✅ Saved: {model_path}")
    print(f"✅ Saved: {importance_path}")

    # store summary metrics for final printout
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
