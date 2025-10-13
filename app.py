import os
import joblib
import pandas as pd
import plotly.express as px
from flask import Flask, render_template, request
from datetime import datetime

# --- App Initialization ---
app = Flask(__name__)

# --- Helper Functions (Logic from your Streamlit files) ---

# Caching models like Streamlit's @st.cache_resource
# Flask doesn't have built-in caching, so we load them once at startup.
MODELS = {}


def load_models():
    """Loads all models into a global dictionary."""
    model_paths = {
        'adult': os.path.join('models', 'adult_model.joblib'),
        'adolescent': os.path.join('models', 'adolescent_model.joblib'),
        'child': os.path.join('models', 'child_model.joblib')
    }
    for key, path in model_paths.items():
        if os.path.exists(path):
            MODELS[key] = joblib.load(path)
    print("Models loaded successfully!")


def load_importance(model_name):
    """Loads feature importance data."""
    path = os.path.join('models', f'{model_name}_feature_importance.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def save_data(data):
    """Saves submitted data to a CSV file."""
    filepath = os.path.join('data', 'collected_data.csv')
    data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([data])
    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False)
    else:
        df.to_csv(filepath, mode='a', header=False, index=False)
    print("Data saved.")


# --- Flask Routes (Your "Pages") ---

@app.route('/')
def home():
    """Renders the homepage (from app.py)."""
    return render_template('index.html')


@app.route('/methodology')
def methodology():
    """Renders the methodology page (from methodology.py)."""
    return render_template('methodology.html')


@app.route('/dashboard')
def dashboard():
    """Renders the interactive analysis page (from dashboard.py)."""
    # Get the selected age group from the URL query parameter (e.g., /dashboard?age_group=Adult)
    data_choice = request.args.get('age_group', 'Adult')  # Default to 'Adult'

    importance_df = load_importance(data_choice.lower())
    chart_html = None

    if importance_df is not None:
        importance_df['feature'] = importance_df['feature'].str.replace('A_Score', '', regex=False).str.replace('_',
                                                                                                                ' ').str.title()

        fig = px.bar(
            importance_df.head(15),
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top 15 Most Predictive Factors for {data_choice}s',
            labels={'importance': 'Relative Importance', 'feature': 'Factor'},
            template='plotly_white'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})

        # Convert the Plotly figure to an HTML div
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    return render_template('dashboard.html', chart_html=chart_html, selected_group=data_choice)


@app.route('/screener', methods=['GET', 'POST'])
def screener():
    """Renders the screener form and handles submissions (from screener.py)."""
    prediction_result = None
    if request.method == 'POST':
        # --- This block runs ONLY after the user clicks "submit" ---
        form_data = request.form
        model_choice = form_data.get('model_choice')

        # 1. Process AQ-10 scores
        reverse_scored_keys = ["A2_Score", "A3_Score", "A4_Score", "A5_Score", "A6_Score", "A9_Score"]
        aq_scores = {}
        for i in range(1, 11):
            key = f"A{i}_Score"
            answer = form_data.get(key)  # "Yes" or "No"
            if key in reverse_scored_keys:
                aq_scores[key] = 1 if answer == "No" else 0
            else:
                aq_scores[key] = 1 if answer == "Yes" else 0

        # 2. Assemble the full input data for the model
        input_data = {
            'age': int(form_data.get('age')),
            'gender': 'm' if form_data.get('gender') == 'Male' else 'f',
            'jundice': 'yes' if form_data.get('jaundice') == 'Yes' else 'no',
            'austim': 'yes' if form_data.get('family_asd') == 'Yes' else 'no',
            'used_app_before': 'no',  # Assuming default values for missing fields
            'relation': 'Self',
            **aq_scores
        }

        # 3. Make prediction
        model_pipeline = MODELS[model_choice]
        model_features = model_pipeline.named_steps['preprocessor'].feature_names_in_
        input_df = pd.DataFrame([input_data], columns=model_features)

        prediction = model_pipeline.predict(input_df)[0]
        confidence = model_pipeline.predict_proba(input_df)[0][prediction] * 100

        # 4. Store result to pass back to the HTML template
        prediction_result = {
            "prediction": prediction,
            "confidence": f"{confidence:.2f}%"
        }

        # (Optional) Save contributed data
        if form_data.get('is_contribution') and form_data.get('actual_result'):
            input_data['ASD_actual'] = 1 if form_data.get('actual_result') == "ASD Positive" else 0
            input_data['age_group'] = model_choice
            save_data(input_data)

    # --- This runs on initial page load (GET) and after a submission (POST) ---
    return render_template('screener.html', result=prediction_result)


# --- Main execution block ---
if __name__ == '__main__':
    load_models()  # Load models once when the app starts
    app.run(debug=True)  # debug=True for development