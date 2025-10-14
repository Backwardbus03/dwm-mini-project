import os
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request
from datetime import datetime

# --- App Initialization ---
app = Flask(__name__)

# --- Helper Functions ---
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
    os.makedirs('data', exist_ok=True)
    data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([data])
    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False)
    else:
        df.to_csv(filepath, mode='a', header=False, index=False)
    print("Data saved.")


# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the homepage."""
    return render_template('index.html')


@app.route('/methodology')
def methodology():
    """Renders the methodology page."""
    return render_template('methodology.html')


@app.route('/demographics')
def demographics():
    """Renders the demographics statistics page using train.csv data."""
    # Load the demographic dataset
    try:
        demo_df = pd.read_csv('data/train.csv')
        
        # Clean data
        demo_df['ethnicity_clean'] = demo_df['ethnicity'].replace('?', 'Unknown')
        demo_df['Class/ASD_label'] = demo_df['Class/ASD'].map({0: 'No ASD', 1: 'ASD'})
        
        # Create visualizations
        charts = {}
        
        # 1. Country Distribution (Top 15)
        country_counts = demo_df['contry_of_res'].value_counts().head(15)
        fig_country = px.bar(
            x=country_counts.values,
            y=country_counts.index,
            orientation='h',
            title='Top 15 Countries in Dataset',
            labels={'x': 'Number of Participants', 'y': 'Country'},
            template='plotly_white',
            color=country_counts.values,
            color_continuous_scale='Blues'
        )
        fig_country.update_layout(showlegend=False, height=500)
        charts['country'] = fig_country.to_html(full_html=False, include_plotlyjs='cdn')
        
        # 2. Ethnicity Distribution
        ethnicity_counts = demo_df['ethnicity_clean'].value_counts()
        fig_ethnicity = px.pie(
            values=ethnicity_counts.values,
            names=ethnicity_counts.index,
            title='Ethnicity Distribution',
            template='plotly_white',
            hole=0.4
        )
        fig_ethnicity.update_traces(textposition='inside', textinfo='percent+label')
        charts['ethnicity'] = fig_ethnicity.to_html(full_html=False, include_plotlyjs='cdn')
        
        # 3. ASD by Ethnicity
        ethnicity_asd = pd.crosstab(demo_df['ethnicity_clean'], demo_df['Class/ASD_label'])
        fig_eth_asd = px.bar(
            ethnicity_asd,
            barmode='group',
            title='ASD Distribution by Ethnicity',
            labels={'value': 'Count', 'ethnicity_clean': 'Ethnicity'},
            template='plotly_white'
        )
        charts['ethnicity_asd'] = fig_eth_asd.to_html(full_html=False, include_plotlyjs='cdn')
        
        # 4. Gender Distribution
        gender_counts = demo_df['gender'].value_counts()
        fig_gender = go.Figure(data=[
            go.Bar(x=['Male', 'Female'], 
                   y=[gender_counts.get('m', 0), gender_counts.get('f', 0)],
                   marker_color=['#3b82f6', '#ec4899'])
        ])
        fig_gender.update_layout(
            title='Gender Distribution',
            xaxis_title='Gender',
            yaxis_title='Count',
            template='plotly_white'
        )
        charts['gender'] = fig_gender.to_html(full_html=False, include_plotlyjs='cdn')
        
        # 5. World Map
        country_data = demo_df.groupby('contry_of_res').agg({
            'Class/ASD': ['count', 'sum']
        }).reset_index()
        country_data.columns = ['country', 'total', 'asd_cases']
        country_data['asd_rate'] = (country_data['asd_cases'] / country_data['total'] * 100).round(2)
        
        fig_map = px.choropleth(
            country_data,
            locations='country',
            locationmode='country names',
            color='total',
            hover_name='country',
            hover_data={'total': True, 'asd_cases': True, 'asd_rate': ':.2f'},
            title='Global Distribution of Participants',
            color_continuous_scale='Viridis',
            template='plotly_white'
        )
        fig_map.update_layout(height=500)
        charts['map'] = fig_map.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Statistics
        stats = {
            'total_participants': len(demo_df),
            'countries': demo_df['contry_of_res'].nunique(),
            'ethnicities': demo_df['ethnicity_clean'].nunique(),
            'asd_rate': (demo_df['Class/ASD'].sum() / len(demo_df) * 100).round(2)
        }
        
        return render_template('demographics.html', charts=charts, stats=stats)
    
    except Exception as e:
        print(f"Error loading demographics: {e}")
        return render_template('demographics.html', charts={}, stats={}, error=str(e))


@app.route('/dashboard')
def dashboard():
    """Renders the interactive analysis page."""
    data_choice = request.args.get('age_group', 'Adult')

    importance_df = load_importance(data_choice.lower())
    chart_html = None

    if importance_df is not None:
        importance_df['feature'] = importance_df['feature'].str.replace('A_Score', '', regex=False).str.replace('_', ' ').str.title()

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
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    return render_template('dashboard.html', chart_html=chart_html, selected_group=data_choice)


@app.route('/screener', methods=['GET', 'POST'])
def screener():
    """Renders the screener form and handles submissions."""
    prediction_result = None
    if request.method == 'POST':
        form_data = request.form
        model_choice = form_data.get('model_choice')

        # Process AQ-10 scores
        reverse_scored_keys = ["A2_Score", "A3_Score", "A4_Score", "A5_Score", "A6_Score", "A9_Score"]
        aq_scores = {}
        for i in range(1, 11):
            key = f"A{i}_Score"
            answer = form_data.get(key)
            if key in reverse_scored_keys:
                aq_scores[key] = 1 if answer == "No" else 0
            else:
                aq_scores[key] = 1 if answer == "Yes" else 0

        # Assemble input data
        input_data = {
            'age': int(form_data.get('age')),
            'gender': 'm' if form_data.get('gender') == 'Male' else 'f',
            'jaundice': 'yes' if form_data.get('jaundice') == 'Yes' else 'no',
            'austim': 'yes' if form_data.get('family_asd') == 'Yes' else 'no',
            **aq_scores
        }

        # Make prediction
        model_pipeline = MODELS[model_choice]
        model_features = model_pipeline.named_steps['preprocessor'].feature_names_in_
        input_df = pd.DataFrame([input_data], columns=model_features)

        prediction = model_pipeline.predict(input_df)[0]
        confidence = model_pipeline.predict_proba(input_df)[0][prediction] * 100

        prediction_result = {
            "prediction": prediction,
            "confidence": f"{confidence:.2f}%"
        }

        # Save data if contribution checkbox is checked
        if form_data.get('is_contribution') and form_data.get('actual_result'):
            input_data['ASD_actual'] = 1 if form_data.get('actual_result') == "ASD Positive" else 0
            input_data['age_group'] = model_choice
            save_data(input_data)

    return render_template('screener.html', result=prediction_result)


# --- Main execution block ---
if __name__ == '__main__':
    load_models()
    app.run(debug=True)
