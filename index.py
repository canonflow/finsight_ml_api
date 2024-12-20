from flask import Flask, request, jsonify
from recommendation_k_means import StockClusteringSystem
from datetime import datetime, timedelta
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd
import yfinance as yf
import json
import os
from utils import (
    extended_forecast,
    parse_data_from_file,
    load_scaler_from_gcs,
    load_model_from_gcs,
    load_csv_from_gcs,
    load_from_gcs,
    get_window
)

app = Flask(__name__)
# WINDOW_SIZE = 4  # Menyesuaikan dengan jenis saham
BUCKET_NAME = "finsight-ml-model"
DATA_FROM = 2024

@app.route('/predict', methods=['POST'])
def predict():  
    data = request.get_json()
    if not data or "stock" not in data or "steps" not in data:
        return jsonify({
            "status": "failed",
            "error": "Missing stock or steps"
        }),400

    try:
        # Get stock and steps from request
        stock = str(data['stock']).upper()
        steps = int(data['steps'])

        # Current Year
        current_year = int(datetime.now().year)
        '''
        Kalo request dari tahun 2025 dengan steps 5, maka akan prediksi s/d 2030
        Karena data sampai dengan tahun 2024, maka kita harus menambahkan 1 (2025 - 2024)
        '''
        gap = current_year - DATA_FROM
        steps += gap
        WINDOW_SIZE = get_window(stock)
        if not WINDOW_SIZE:
            return jsonify({
                "status": "failed",
                "error": "Stock unavailable"
            }), 404

        # ===== UNCOMMENT UNTUK DOWNLOAD DARI GCS =====
        # model_path = f"stock_model/{stock}/model_saham.h5"
        # scaler_path = f"stock_model/{stock}/scaler.pkl"
        # csv_path = f"stock_model/{stock}/data_saham.csv"
        #
        # local_model_path = f"models/{stock}"
        # local_scaler_path = f"scalers/{stock}"
        # local_csv_path = f"csv/{stock}"

        # Check dirs
        # if not os.path.exists(local_model_path):
        #     os.makedirs(local_model_path)
        # if not os.path.exists(local_scaler_path):
        #     os.makedirs(local_scaler_path)
        # if not os.path.exists(local_csv_path):
        #     os.makedirs(local_csv_path)

        # model = load_model_from_gcs(BUCKET_NAME, model_path, local_model_path + "/model_saham.h5")
        # scaler = load_scaler_from_gcs(BUCKET_NAME, scaler_path, local_scaler_path + "/scaler.pkl")
        # load_csv_from_gcs(BUCKET_NAME, csv_path, local_csv_path + "/data_saham.csv")

        # model, scaler = load_from_gcs(BUCKET_NAME, paths={
        #     "model_path": model_path,
        #     "local_model_path": local_model_path + "/model_saham.h5",
        #     "scaler_path": scaler_path,
        #     "local_scaler_path": local_scaler_path +"/scaler.pkl",
        #     "csv_path": csv_path,
        #     "local_csv_path": local_csv_path + "/data_saham.csv"
        # })
        # ===== UNCOMMENT UNTUK DOWNLOAD DARI GCS =====

        # Load model and scaler based on the stock
        model = tf.keras.models.load_model(f"models/{stock}/model_saham.h5")
        scaler = joblib.load(f"scalers/{stock}/scaler.pkl")

        # Get the data
        # TIME, SERIES = parse_data_from_file(local_csv_path + "/data_saham.csv")
        TIME, SERIES = parse_data_from_file(f"csv/{stock}/data_saham.csv")
        SERIES = scaler.fit_transform(SERIES.reshape(-1, 1)).flatten()

        # Generate Future Times
        future_time  = pd.date_range(start=TIME[-1], periods=52 * steps + 1, freq='W')[1:]

        # Predict
        predicted_values = extended_forecast(model, SERIES, WINDOW_SIZE, forecast_steps=steps)

        # Convert to actual price
        predicted_actual = scaler.inverse_transform([predicted_values]).flatten().tolist()

        # Percentage Change
        percentage_change = round(
            ((predicted_actual[-1] / predicted_actual[0]) - 1) * 100,
            2
        )

        # Response
        response = {
            'status': 'success',
            'prediction': {
                'prices': predicted_actual,
                'times': future_time.tolist()
            },
            'year_from': str(DATA_FROM),
            'year_to': str(DATA_FROM + steps),
            'percentage_change': percentage_change
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({
            "status": "failed",
            "error": str(e)
        }),400

    # # Return json (array)
    # return jsonify({
    #     'predictions': predicted_actual.flatten().tolist(),
    #     'year_from': str(DATA_FROM),
    #     'year_to': str(DATA_FROM + steps),
    #     'prediction': predicted_actual.flatten().tolist()[-1]
    # })
    
@app.route("/stocks", methods=['GET'])
def stocks():
    try:
        stocks = []
        tickers = [
            "^GSPC", "ADRO.JK", "ANTM.JK", "ASII.JK", "BBCA.JK", "BBNI.JK", 
            "BBRI.JK", "BMRI.JK", "CTRA.JK", "GC=F", "GGRM.JK", "IDR=X", 
            "INDF.JK", "INDY.JK", "LPKR.JK", "MYOR.JK", "PWON.JK", "UNVR.JK"
        ]
        bucket_base_url = "https://storage.googleapis.com/finsight-profile/stocks/"

        for ticker in tickers:
            stock = yf.Ticker(ticker)
            history = stock.history(period="1d")
            current_price = history['Close'].iloc[-1] if not history.empty else None
            company_info = stock.info

            stocks.append({
                "ticker": ticker,
                "image_url": f"{bucket_base_url}{ticker}.png",
                "current_price": current_price,
                "description": company_info.get('longBusinessSummary', "Description not available")
            })

        return jsonify({
            "status": "success",
            "stocks": stocks
        }), 200

    except Exception as e:
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 400

    
@app.route('/riskprofile', methods=['POST'])
def riskProfile():
    data = request.get_json()
    
    if not data or "riskProfile" not in data:
        return jsonify({
            "status": "failed",
            "error": "Missing risk profile!"
        }), 400
    
    riskProfile = data["riskProfile"]
    
    tickers = ["^GSPC", "ADRO.JK", "ANTM.JK", "ASII.JK", "BBCA.JK", "BBNI.JK", 
               "BBRI.JK", "BMRI.JK", "CTRA.JK", "GC=F", "GGRM.JK", "IDR=X", 
               "INDF.JK", "INDY.JK", "LPKR.JK", "MYOR.JK", "PWON.JK", "UNVR.JK"]
    
    bucket_base_url = "https://storage.googleapis.com/finsight-profile/stocks/"
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)
    
    try: 
        # Initialize and run the stock clustering system
        stock_system = StockClusteringSystem(
            tickers, 
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        stock_system.fetch_data()
        stock_system.create_feature_matrix()
        stock_system.preprocess_features()
        stock_system.perform_clustering(n_clusters=3)
        
        recommendations = stock_system.get_recommendations(riskProfile)
        recommendations_with_details = []
        
        for ticker in recommendations:
            stock = yf.Ticker(ticker)
            history = stock.history(period="1d")
            current_price = history['Close'].iloc[-1] if not history.empty else None
            company_info = stock.info
            
            recommendations_with_details.append({
                "ticker": ticker,
                "image_url": f"{bucket_base_url}{ticker}.png",
                "current_price": current_price,
                "description": company_info.get('longBusinessSummary', "Description not available")
            })
        
        return jsonify({
            "status": "success",
            "recommendations": recommendations_with_details
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 400

    
PORT=8080

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=PORT)
