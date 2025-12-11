import os
import streamlit as st
import pandas as pd
import requests
import joblib
import warnings
import numpy as np

# ---------------------------
# Streamlit page config (MUST be the first streamlit command)
# ---------------------------
st.set_page_config(page_title="Crop Recommendation", layout="wide")

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------
# Paths / constants
# ---------------------------
BASE_DIR = os.path.dirname(__file__) or "."
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.joblib")
COLUMNS_PATH = os.path.join(BASE_DIR, "model_columns.joblib")
IMAGE_DIR = os.path.join(BASE_DIR, "crop_images")

# Safe image display helper: tries use_container_width, falls back if not supported
def show_image(path_or_obj, caption=None):
    try:
        st.image(path_or_obj, caption=caption, use_container_width=True)
    except TypeError:
        st.image(path_or_obj, caption=caption)

# ---------------------------
# Load Trained Model & Encoder
# ---------------------------
try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    st.success("✅ Model, Encoder, and Columns loaded successfully.")
except Exception as e:
    st.error(
        "ERROR: Model files not found or failed to load. "
        "Ensure model.joblib, encoder.joblib, and model_columns.joblib are in the app folder."
    )
    st.exception(e)
    st.stop()

# Ensure model_columns is a list in correct order
if isinstance(model_columns, (np.ndarray, list, tuple)):
    model_columns = list(model_columns)
else:
    st.error("model_columns has unexpected type. Expecting list/ndarray.")
    st.stop()

# ---------------------------
# OpenWeather API Function
# ---------------------------
def get_weather(city: str, api_key: str):
    """
    Fetches current temperature and humidity from OpenWeather API.
    Returns (temp_C, humidity) or (None, None) on failure.
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        temp = float(data["main"]["temp"])
        humidity = float(data["main"]["humidity"])
        return temp, humidity
    except requests.exceptions.HTTPError as err:
        code = getattr(err.response, "status_code", None)
        if code == 404:
            st.error(f"City not found: {city}")
        elif code == 401:
            st.error("Invalid API Key. Please check your key.")
        else:
            st.error(f"Weather API Error: {err}")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while calling weather API: {e}")
        return None, None

# ---------------------------
# UI: Title + caption
# ---------------------------
st.title("🌾 Crop Recommendation System")
st.caption(
    "Dataset ranges (approx): N: 0–140, P: 5–145, K: 5–205, Temp: 9–44 °C, "
    "Humidity: 14–100 %, pH: 3.5–9.9, Rainfall: 20–299 mm"
)

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Step 1: Choose Input Mode")
    mode = st.radio("Select how to input weather data:", ("Manual Mode", "Automatic (City) Mode"))

    st.header("Step 2: Enter Crop Parameters")
    with st.form(key="crop_form"):
        N = st.slider("Nitrogen (N) (kg/ha)", min_value=0, max_value=140, value=70, help="Dataset range ≈ 0 – 140")
        P = st.slider("Phosphorus (P) (kg/ha)", min_value=0, max_value=150, value=50, help="Dataset range ≈ 5 – 145")
        K = st.slider("Potassium (K) (kg/ha)", min_value=0, max_value=210, value=60, help="Dataset range ≈ 5 – 205")
        ph = st.slider("Soil pH", min_value=3.0, max_value=10.0, value=6.5, step=0.1, help="Dataset range ≈ 3.5 – 9.9")

        # Defaults
        api_key = None
        city = None

        if mode == "Manual Mode":
            st.markdown("---")
            temp = st.slider("Temperature (°C)", min_value=8.0, max_value=45.0, value=26.0, step=0.1)
            humidity = st.slider("Humidity (%)", min_value=14.0, max_value=100.0, value=70.0, step=0.1)
            rainfall = st.slider("Rainfall (mm)", min_value=20.0, max_value=300.0, value=120.0, step=0.1)
        else:
            st.markdown("---")
            st.info("🌦️ API will fetch **Current Temperature** and **Humidity**.")
            # Prefer environment secret, fall back to user input field
            api_key_env = os.environ.get("OPENWEATHER_API_KEY") or st.secrets.get("OPENWEATHER_API_KEY") if hasattr(st, "secrets") else None
            api_key = api_key_env or st.text_input("Enter your OpenWeather API Key", type="password", help="Get a free key from openweathermap.org")
            city = st.text_input("Enter your City")
            st.warning("**Rainfall must be entered manually.**")
            rainfall = st.slider("Average Seasonal/Monthly Rainfall (mm)", min_value=20.0, max_value=300.0, value=120.0, step=0.1)
            temp, humidity = None, None

        submit_button = st.form_submit_button(label="🌱 Recommend Crop")

with col2:
    st.header("Step 3: Your Recommendation")

    if submit_button:
        # Validation
        if mode == "Automatic (City) Mode" and not api_key:
            st.error("Please enter your OpenWeather API Key to use Automatic Mode.")
        elif mode == "Automatic (City) Mode" and not city:
            st.error("Please enter a city name for Automatic Mode.")
        else:
            if mode == "Automatic (City) Mode":
                with st.spinner(f"Fetching current weather for {city}..."):
                    temp, humidity = get_weather(city, api_key)

            if temp is not None and humidity is not None:
                data = {
                    "N": float(N),
                    "P": float(P),
                    "K": float(K),
                    "temperature": float(temp),
                    "humidity": float(humidity),
                    "ph": float(ph),
                    "rainfall": float(rainfall),
                }
                features_df = pd.DataFrame([data])

                # Ensure column order matches the model
                missing_cols = [c for c in model_columns if c not in features_df.columns]
                if missing_cols:
                    st.error(f"Model expects columns {model_columns}. Missing: {missing_cols}")
                else:
                    features_df = features_df[model_columns]

                    try:
                        prediction_encoded = model.predict(features_df)
                        pred_array = np.array(prediction_encoded).ravel()
                        prediction_crop = encoder.inverse_transform(pred_array)[0]
                    except Exception as e:
                        st.error("Prediction failed: see details below.")
                        st.exception(e)
                        prediction_crop = None

                    if prediction_crop:
                        st.success(f"**The recommended crop is: {prediction_crop.capitalize()}**")

                        img_path = os.path.join(IMAGE_DIR, f"{prediction_crop.lower()}.jpg")
                        if os.path.exists(img_path):
                            show_image(img_path, caption=prediction_crop.capitalize())
                        else:
                            st.info(
                                "📷 Crop image not available yet. "
                                f"Add an image named `{prediction_crop.lower()}.jpg` to the 'crop_images' folder."
                            )

                        st.subheader("Inputs Used for this Prediction:")
                        if mode == "Automatic (City) Mode":
                            st.markdown(f"**Fetched from {city}:**")
                            st.markdown(f"* **Temperature:** `{temp:.2f} °C`")
                            st.markdown(f"* **Humidity:** `{humidity:.2f} %`")

                        st.markdown("**All Inputs:**")
                        st.dataframe(features_df)
            elif mode == "Automatic (City) Mode":
                st.error("Prediction failed because weather data could not be fetched.")
