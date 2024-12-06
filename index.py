from flask import Flask, request, jsonify
from recommendation_k_means import StockClusteringSystem
from datetime import datetime, timedelta
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd
import json
import os
from utils import (
    extended_forecast,
    parse_data_from_file,
    load_scaler_from_gcs,
    load_model_from_gcs,
    load_csv_from_gcs,
    load_from_gcs
)

app = Flask(__name__)
WINDOW_SIZE = 4  # Menyesuaikan dengan jenis saham
BUCKET_NAME = "finsight-ml-model"
DATA_FROM = 2024

@app.route('/predict', methods=['POST'])
def predict():  
    data = request.get_json()
    # Get stock and steps from request
    stock = str(data['stock']).upper()
    steps = int(data['steps'])

    # Current Year
    current_year = int(datetime.now().year)
    '''
    Kalo request dari tahun 2025 dengan steps 5, maka akan prediksi s/d 2023
    Karena data sampai dengan tahun 2024, maka kita harus menambahkan 1 (2025 - 2024)
    '''
    gap = current_year - DATA_FROM
    steps += gap

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

    # Predict
    predicted_values = extended_forecast(model, SERIES, WINDOW_SIZE, forecast_steps=steps)

    # Convert to actual price
    predicted_actual = scaler.inverse_transform([predicted_values])

    # Return json (array)
    return jsonify({
        'predictions': predicted_actual.flatten().tolist(),
        'year_from': str(DATA_FROM),
        'year_to': str(DATA_FROM + steps),
        'prediction': predicted_actual.flatten().tolist()[-1]
    })
    
@app.route('/riskprofile', methods=['POST'])
def riskProfile():
    data = request.get_json()
    
    if not data or "riskProfile" not in data:
        return jsonify({
            "status": "failed",
            "error": "Missing risk profile!"
        }),400
    
    riskProfile = data["riskProfile"]
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'PG', '^GSPC', 'BTC-USD']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  
    try: 
        stock_system = StockClusteringSystem(tickers, start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'))
        
        stock_system.fetch_data()
        stock_system.create_feature_matrix()
        stock_system.preprocess_features()
        stock_system.perform_clustering(n_clusters=3)
        
        recommendations = stock_system.get_recommendations(riskProfile)
        
        return jsonify({
            "status": "success",
            "recommendations": recommendations
        }),200
        
        
        
    except Exception as e:
        return jsonify({
            "status": "failed",
            "error": str(e)
        }),400
    

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, port=3000)