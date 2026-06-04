from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned Car.csv')


@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html',
                           companies=companies,
                           car_models=car_models,
                           years=year,
                           fuel_types=fuel_type)


@app.route('/get_models', methods=['POST'])
@cross_origin()
def get_models():
    """Return car models for a selected company as JSON."""
    company = request.json.get('company', '')
    filtered = car[car['name'].str.startswith(company)]['name'].unique()
    return jsonify(sorted(filtered.tolist()))


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    prediction = model.predict(
        pd.DataFrame(
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)
        )
    )
    
    estimated_price = float(np.round(prediction[0], 2))
    
    # Calculate valuation trend for surrounding model years (bounded by 1995-2019)
    try:
        target_year = int(year)
        start_year = max(1995, target_year - 3)
        end_year = min(2019, target_year + 3)
        
        # Ensure we always get a window of up to 6 years if possible
        if end_year - start_year < 6:
            if start_year == 1995:
                end_year = min(2019, 1995 + 6)
            elif end_year == 2019:
                start_year = max(1995, 2019 - 6)
                
        trend = []
        for y in range(start_year, end_year + 1):
            pred_y = model.predict(
                pd.DataFrame(
                    columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                    data=np.array([car_model, company, y, driven, fuel_type]).reshape(1, 5)
                )
            )
            trend.append({
                'year': y,
                'price': float(np.round(pred_y[0], 2))
            })
    except Exception as e:
        print("Trend calculation error:", str(e))
        trend = []

    return jsonify({
        'price': estimated_price,
        'trend': trend
    })



@app.route('/chat', methods=['POST'])
@cross_origin()
def chat():
    user_message = request.json.get('message', '')
    history_data = request.json.get('history', [])
    
    if not user_message:
        return jsonify({'response': 'Message is required.'}), 400
        
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({
            'response': (
                "⚠️ **Gemini API Key is missing.**\n\n"
                "Please configure your Gemini API Key by creating a `.env` file in the project folder with:\n"
                "```env\nGEMINI_API_KEY=your_actual_api_key_here\n```\n"
                "Then restart the server to enable the AI Chat Assistant!"
            ),
            'needs_key': True
        })

    try:
        genai.configure(api_key=api_key)
        
        # We use gemini-2.5-flash as the standard, fast, and high-quality chatbot model
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction=(
                "You are an expert, friendly AI Car Assistant for the 'AI Car Price Predictor' website.\n"
                "Your role is to help users value their cars, give tips on buying and selling used cars, "
                "and explain how our Linear Regression Machine Learning model makes predictions.\n\n"
                "Important Context:\n"
                "- Our model is trained on a dataset of around 815 car listings.\n"
                "- It predicts price using: Company, Model Name, Year of Purchase, Kilometers Driven, and Fuel Type.\n"
                "- Always respond in clean markdown format. Keep answers helpful but concise. "
                "Use bullet points when listing tips or steps."
            )
        )
        
        formatted_history = []
        for msg in history_data:
            role = 'user' if msg.get('role') == 'user' else 'model'
            formatted_history.append({
                'role': role,
                'parts': [msg.get('text', '')]
            })
            
        chat_session = model.start_chat(history=formatted_history)
        response = chat_session.send_message(user_message)
        
        return jsonify({
            'response': response.text,
            'success': True
        })
    except Exception as e:
        print("Chat Error:", str(e))
        return jsonify({
            'response': f"An error occurred while communicating with the AI model: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(debug=True)