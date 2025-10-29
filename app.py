import os # For file path operations
import joblib # For loading ML models
import pandas as pd # For data manipulation
import plotly.express as px # For visualization
import plotly.graph_objects as go # Also for visualization
from flask import Flask, render_template, request # For web app framework
from datetime import datetime # For timestamping data entries


# --- App Initialization ---
app = Flask(__name__)


MODELS = {} # global cache for loaded models so we don't reload on every request


def load_models():
    # load saved ML pipelines from disk into the MODELS dict
    # we expect joblib files in the models/ directory
    model_paths = {
        'adult': os.path.join('models', 'adult_model.joblib'),
        'adolescent': os.path.join('models', 'adolescent_model.joblib'),
        'child': os.path.join('models', 'child_model.joblib'),
        'toddler_qchat': os.path.join('models', 'toddler_qchat_model.joblib')
    }

    # try loading each file if it exists, silently skip missing files
    for key, path in model_paths.items():
        if os.path.exists(path):
            # load the sklearn pipeline (or model) saved with joblib
            MODELS[key] = joblib.load(path)
    # quick console feedback so devs know which models are available
    print(f"Models loaded successfully! Available: {list(MODELS.keys())}")
 


def load_importance(model_name):
    # Load CSV that lists feature importances for a given model
    # returns a DataFrame or None if file missing
    path = os.path.join('models', f'{model_name}_feature_importance.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def save_data(data):
    # Append a single record (dict) to data/collected_data.csv
    # Creates data/ dir if missing and writes header only on first write
    filepath = os.path.join('data', 'collected_data.csv')
    os.makedirs('data', exist_ok=True)
    # add a human-friendly timestamp so later we can trace contributions
    data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([data])
    if not os.path.exists(filepath):
        # first time: write header
        df.to_csv(filepath, index=False)
    else:
        # append without header
        df.to_csv(filepath, mode='a', header=False, index=False)
    print("Data saved.")


## Routing ##
@app.route('/')
def home():
    # landing page
    return render_template('index.html')


@app.route('/methodology')
def methodology():
    return render_template('methodology.html')


@app.route('/demographics')
def demographics():
    try:
        # read the train.csv dataset (used for plotting demographics)
        demo_df = pd.read_csv('data/train.csv')


        # quick cleaning / helper columns we use in charts
        # replace missing value marker with a readable label
        demo_df['ethnicity_clean'] = demo_df['ethnicity'].replace('?', 'Unknown')
        # map numeric class to a friendly label for hover text
        demo_df['Class/ASD_label'] = demo_df['Class/ASD'].map({0: 'No ASD', 1: 'ASD'})

        charts = {}


        # 1. COUNTRY ASD RATE (Top 15 by sample size, colored by ASD rate)
        # aggregate counts and ASD case sums by country
        country_stats = demo_df.groupby('contry_of_res').agg({
            'Class/ASD': ['count', 'sum']
        }).reset_index()
        # flatten multiindex columns to something easy to work with
        country_stats.columns = ['country', 'total', 'asd_cases']
        # percentage of ASD in each country (rounded)
        country_stats['asd_rate'] = (country_stats['asd_cases'] / country_stats['total'] * 100).round(2)
        # convenience column for plotting stacked charts later
        country_stats['no_asd_rate'] = (100 - country_stats['asd_rate']).round(2)
        # Get top 15 countries by total participants
        top_countries = country_stats.nlargest(15, 'total').sort_values('asd_rate')
        fig_country = px.bar(
            top_countries,
            x='asd_rate',
            y='country',
            orientation='h',
            title='ASD Rate by Country (Top 15 Countries by Sample Size)',
            labels={'asd_rate': 'ASD Rate (%)', 'country': 'Country'},
            template='plotly_white',
            color='asd_rate',
            color_continuous_scale='RdYlGn_r',  # Red=high, Green=low
            hover_data={
                'asd_rate': ':.2f',
                'total': True,
                'asd_cases': True
            }
        ) # Horizontal bar chart
        fig_country.update_layout(showlegend=False, height=500) 
        charts['country'] = fig_country.to_html(full_html=False, include_plotlyjs='cdn')


        # 2. ETHNICITY DISTRIBUTION (Pie chart showing sample proportions)
        # simple pie chart of sample proportions by ethnicity
        ethnicity_counts = demo_df['ethnicity_clean'].value_counts()
        fig_ethnicity = px.pie(
            values=ethnicity_counts.values,
            names=ethnicity_counts.index,
            title='Ethnicity Distribution (Sample Proportions)',
            template='plotly_white',
            hole=0.4
        )
        fig_ethnicity.update_traces(textposition='inside', textinfo='percent+label')
        charts['ethnicity'] = fig_ethnicity.to_html(full_html=False, include_plotlyjs='cdn')


        # 3. ETHNICITY ASD RATE (Stacked bar showing ASD%)
        # compute ASD per ethnicity for stacked bar
        ethnicity_stats = demo_df.groupby('ethnicity_clean').agg({
            'Class/ASD': ['count', 'sum']
        }).reset_index()
        ethnicity_stats.columns = ['ethnicity', 'total', 'asd_cases']
        ethnicity_stats['asd_rate'] = (ethnicity_stats['asd_cases'] / ethnicity_stats['total'] * 100).round(2)
        ethnicity_stats = ethnicity_stats.sort_values('asd_rate', ascending=False)

        fig_eth_asd = go.Figure(data=[
            go.Bar(
                name='ASD',
                x=ethnicity_stats['ethnicity'],
                y=ethnicity_stats['asd_rate'],
                marker_color='#ef4444',
                text=[f"{x:.1f}%" for x in ethnicity_stats['asd_rate']],
                textposition='inside',
                hovertemplate='%{x}<br>ASD: %{y:.2f}%<extra></extra>'
            )
        ])
        fig_eth_asd.update_layout(
            barmode='stack',
            title='ASD Rate by Ethnicity (% Distribution)',
            xaxis_title='Ethnicity',
            yaxis_title='Percentage (%)',
            template='plotly_white',
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        charts['ethnicity_asd'] = fig_eth_asd.to_html(full_html=False, include_plotlyjs='cdn')


        # 4. GENDER ASD RATE COMPARISON (Side-by-side bars with differentiation)
        # gender breakdown - similar approach to above
        gender_stats = demo_df.groupby('gender').agg({
            'Class/ASD': ['count', 'sum']
        }).reset_index()
        gender_stats.columns = ['gender', 'total', 'asd_cases']
        gender_stats['asd_rate'] = (gender_stats['asd_cases'] / gender_stats['total'] * 100).round(2)
        gender_stats['no_asd_rate'] = (100 - gender_stats['asd_rate']).round(2)

        # pick out male / female rows (dataset uses 'm' and 'f')
        male_stats = gender_stats[gender_stats['gender'] == 'm']
        female_stats = gender_stats[gender_stats['gender'] == 'f']

        fig_gender = go.Figure(data=[
            go.Bar(
                name='ASD Rate',
                x=['Male', 'Female'],
                y=[
                    male_stats['asd_rate'].values[0] if len(male_stats) > 0 else 0,
                    female_stats['asd_rate'].values[0] if len(female_stats) > 0 else 0
                ],
                marker_color=['#3b82f6', '#ec4899'],
                text=[
                    f"{male_stats['asd_rate'].values[0]:.1f}%" if len(male_stats) > 0 else "0%",
                    f"{female_stats['asd_rate'].values[0]:.1f}%" if len(female_stats) > 0 else "0%"
                ],
                textposition='auto',
            ),
            go.Bar(
                name='No ASD Rate',
                x=['Male', 'Female'],
                y=[
                    male_stats['no_asd_rate'].values[0] if len(male_stats) > 0 else 0,
                    female_stats['no_asd_rate'].values[0] if len(female_stats) > 0 else 0
                ],
                marker_color=['#93c5fd', '#f9a8d4'],
                text=[
                    f"{male_stats['no_asd_rate'].values[0]:.1f}%" if len(male_stats) > 0 else "0%",
                    f"{female_stats['no_asd_rate'].values[0]:.1f}%" if len(female_stats) > 0 else "0%"
                ],
                textposition='auto',
                visible='legendonly' 
            )
        ])
        fig_gender.update_layout(
            title='ASD Rate by Gender (%)',
            xaxis_title='Gender',
            yaxis_title='Rate (%)',
            template='plotly_white',
            barmode='group'
        )
        charts['gender'] = fig_gender.to_html(full_html=False, include_plotlyjs='cdn')


        # 5. WORLD MAP - COLOR BY ASD RATE (Red=High, Green=Low)
        # create a dataset suitable for a choropleth world map
        country_data = demo_df.groupby('contry_of_res').agg({
            'Class/ASD': ['count', 'sum']
        }).reset_index()
        country_data.columns = ['country', 'total', 'asd_cases']
        country_data['asd_rate'] = (country_data['asd_cases'] / country_data['total'] * 100).round(2)
        country_data['no_asd_cases'] = country_data['total'] - country_data['asd_cases']
        country_data['no_asd_rate'] = (100 - country_data['asd_rate']).round(2)

        fig_map = px.choropleth(
            country_data,
            locations='country',
            locationmode='country names',
            color='asd_rate', 
            hover_name='country',
            hover_data={
                'total': True,
                'asd_cases': True,
                'no_asd_cases': True,
                'asd_rate': ':.2f',
                'no_asd_rate': ':.2f'
            },
            title='Global ASD Rate Distribution (% of ASD Cases per Country)',
            color_continuous_scale='RdYlGn_r',  # Red=high ASD rate, Green=low
            range_color=[0, 60],  # Reasonable percentage range
            template='plotly_white',
            labels={
                'asd_rate': 'ASD Rate (%)',
                'total': 'Total Samples',
                'asd_cases': 'ASD Cases',
                'no_asd_cases': 'Non-ASD Cases',
                'no_asd_rate': 'Non-ASD Rate (%)'
            }
        )
        fig_map.update_layout(height=500)
        charts['map'] = fig_map.to_html(full_html=False, include_plotlyjs='cdn')


        # STATISTICS SUMMARY
        # basic summary numbers shown on the demographics page
        total_asd = demo_df['Class/ASD'].sum()
        total_no_asd = len(demo_df) - total_asd

        stats = {
            'total_participants': len(demo_df),
            'countries': demo_df['contry_of_res'].nunique(),
            'ethnicities': demo_df['ethnicity_clean'].nunique(),
            'asd_rate': (total_asd / len(demo_df) * 100).round(2),
            'asd_count': int(total_asd),
            'no_asd_count': int(total_no_asd),
            'no_asd_rate': (total_no_asd / len(demo_df) * 100).round(2)
        }

        # finally render the template with charts and stats dict
        return render_template('demographics.html', charts=charts, stats=stats)

    except Exception as e:
        # catch-all so the page doesn't crash in production; log error to console
        print(f"Error loading demographics: {e}")
        import traceback
        traceback.print_exc()
        # render the same template but with empty charts so UI still loads
        return render_template('demographics.html', charts={}, stats={}, error=str(e))


@app.route('/dashboard')
def dashboard():
    data_choice = request.args.get('age_group', 'Adult')
    
    # Map Toddler_qchat to toddler_qchat for file loading
    model_name = data_choice.lower()
    if model_name == 'toddler_qchat':
        model_name = 'toddler_qchat'
    
    importance_df = load_importance(model_name)

    if importance_df is not None:
        # clean up feature names for display, remove technical prefixes
        importance_df['feature'] = importance_df['feature'].str.replace('A_Score', '', regex=False).str.replace('_', ' ').str.title()

        # simple horizontal bar of the top features
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
        # get the pipeline from cache and build a DataFrame with expected columns
        model_pipeline = MODELS[model_choice]
        # preprocessor.feature_names_in_ contains the columns the pipeline expects
        model_features = model_pipeline.named_steps['preprocessor'].feature_names_in_
        input_df = pd.DataFrame([input_data], columns=model_features)

        # model returns a class label and we also compute class probability for confidence
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

# Add this new route AFTER the /screener route (after line ~180)
@app.route('/early-detection', methods=['GET', 'POST'])
def early_detection():
    prediction_result = None
    
    if request.method == 'POST':
        form_data = request.form
        
        # Process Q-CHAT-10 answers (A1-A10 are binary 0/1)
        qchat_scores = {}
        for i in range(1, 11):
            key = f"A{i}"
            answer = form_data.get(key)
            qchat_scores[key] = 1 if answer == "Yes" else 0
        
        # Assemble input data
        input_data = {
            'Age_Mons': int(form_data.get('age_months')),
            'Sex': form_data.get('gender').lower(),
            'Jaundice': form_data.get('jaundice').lower(),
            'Family_mem_with_ASD': form_data.get('family_asd').lower(),
            **qchat_scores
        }
        
        # Make prediction
        # use the toddler Q-CHAT pipeline (we expect it to be loaded)
        model_pipeline = MODELS['toddler_qchat']
        # define explicit order of features the model expects
        model_features = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 
                         'Age_Mons', 'Sex', 'Jaundice', 'Family_mem_with_ASD']
        input_df = pd.DataFrame([input_data], columns=model_features)
        
        # predict class and probability
        prediction = model_pipeline.predict(input_df)[0]
        confidence = model_pipeline.predict_proba(input_df)[0][prediction] * 100
        
        prediction_result = {
            "prediction": prediction,
            "confidence": f"{confidence:.2f}%"
        }
    
    return render_template('early_detection.html', result=prediction_result)



# --- Run App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)