import tensorflow as tf
import numpy as np
import pandas as pd
from google.cloud import storage
import joblib

def extended_forecast(model, series, window_size, forecast_steps):
    """
    Generates a forecast using your trained model up to a specified number of future steps.
    """
    # Initialize forecast results with the original series to begin prediction
    # forecast = list(series[-window_size:])  # Start from the last known window
    #
    # for _ in range(forecast_steps):
    #     # Convert forecast list to tensor
    #     input_series = np.array(forecast[-window_size:]).reshape(1, -1)  # Shape (1, window_size)
    #
    #     # Predict the next step
    #     next_step = model.predict(input_series)[0][0]  # Get the predicted value
    #
    #     # Append the next step to the forecast
    #     forecast.append(next_step)
    #
    # # Return only the future forecasted steps
    # return np.array(forecast[-forecast_steps:])
    WEEKS_IN_YEARS = 52 * forecast_steps
    future_forecast = []
    last_window = series[-window_size:]

    for _ in range(WEEKS_IN_YEARS):
        prediction = model.predict(last_window[np.newaxis, :])
        future_forecast.append(prediction.squeeze())
        last_window = np.roll(last_window, -1)
        last_window[-1] = prediction.squeeze()

    return np.array(future_forecast)

def parse_data_from_file(filename):
  # Load the file, skipping the first three rows to remove unnecessary headers
  data = pd.read_csv(filename, skiprows=[1,2])

  # Rename price column to date
  data.rename(columns={'Price': 'Date'}, inplace=True)

  # Convert the 'Date' column to datetime format
  data['Date'] = pd.to_datetime(data['Date'])

  # Set 'Date' as the index
  data.set_index('Date', inplace=True)

  # Select only the 'Adj Close' column for forecasting
  data = data[['Adj Close']]

  data.head()

  dates = data.index.tolist()
  adj_closes = data['Adj Close'].tolist()

  return np.array(dates), np.array(adj_closes)

def load_model_from_gcs(bucket_name, model_path, local_model_path):
    # Initialize GCS client
    client = storage.Client.from_service_account_json("credential.json")

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Get the blob (file) in the bucket
    blob = bucket.blob(model_path)

    # Download the model locally
    blob.download_to_filename(local_model_path)
    print(f"Model downloaded to {local_model_path}")

    model = tf.keras.models.load_model(local_model_path)

def load_scaler_from_gcs(bucket_name, scaler_path, local_scaler_path):
    """
    Load a model file from Google Cloud Storage to the local system.

    :param bucket_name: Name of the GCS bucket
    :param model_path: Path to the model file in the GCS bucket
    :param local_model_path: Path to save the model locally
    :return: Loaded model object
    """
    # Initialize GCS client
    client = storage.Client.from_service_account_json("credential.json")

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Get the blob (file) in the bucket
    blob = bucket.blob(scaler_path)

    # Download the model locally
    blob.download_to_filename(local_scaler_path)
    print(f"Scaler downloaded to {local_scaler_path}")

    # Load the model using joblib (or your library of choice)
    scaler = joblib.load(local_scaler_path)
    print("Model loaded successfully.")
    return scaler

def load_csv_from_gcs(bucket_name, csv_path, local_csv_path):
    """
    Load a model file from Google Cloud Storage to the local system.

    :param bucket_name: Name of the GCS bucket
    :param model_path: Path to the model file in the GCS bucket
    :param local_model_path: Path to save the model locally
    :return: Loaded model object
    """
    # Initialize GCS client
    client = storage.Client.from_service_account_json("credential.json")

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Get the blob (file) in the bucket
    blob = bucket.blob(csv_path)

    # Download the model locally
    blob.download_to_filename(local_csv_path)
    print(f"CSV downloaded to {local_csv_path}")

    # # Load the model using joblib (or your library of choice)
    # scaler = joblib.load(local_csv_path)
    # print("CSV loaded successfully.")
    # return scaler

def load_from_gcs(bucket_name, paths={}):
    # Initialize GCS client
    client = storage.Client.from_service_account_json("credential.json")

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # ===== MODEL =====
    # Get the blob (file) in the bucket
    blob = bucket.blob(paths.get("model_path"))

    # Download the model locally
    blob.download_to_filename(paths.get("local_model_path"))
    model = tf.keras.models.load_model(paths.get("local_model_path"))

    # ===== SCALER =====
    blob = bucket.blob(paths.get("scaler_path"))

    # Download the model locally
    blob.download_to_filename(paths.get("local_scaler_path"))

    # Load the model using joblib (or your library of choice)
    scaler = joblib.load(paths.get("local_scaler_path"))

    # ===== CSV =====
    blob = bucket.blob(paths.get("csv_path"))

    # Download the model locally
    blob.download_to_filename(paths.get("local_csv_path"))


    return model, scaler
