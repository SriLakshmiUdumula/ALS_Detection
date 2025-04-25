from flask import Flask, render_template, request
import parselmouth
from parselmouth.praat import call
import numpy as np
import pandas as pd
import os
import pickle

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and scaler
with open("xgb_model_rem.pkl", "rb") as f:
    model = pickle.load(f)

with open("als_scaler2.pkl", "rb") as f:
    scaler = pickle.load(f)

# List of required phonation labels
PHONATION_LABELS = ['A', 'E', 'I', 'O', 'U', 'PA', 'TA', 'KA']

# Feature extraction
def extract_features(audio_path):
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    f0_values = pitch.selected_array['frequency']
    f0_values = f0_values[f0_values > 0]
    
    meanF0 = np.mean(f0_values) if len(f0_values) > 0 else 0
    stdevF0 = np.std(f0_values) if len(f0_values) > 0 else 0
    
    hnr = snd.to_harmonicity()
    hnr_values = hnr.values[hnr.values > 0]
    meanHNR = np.mean(hnr_values) if len(hnr_values) > 0 else 0
    
    pointProcess = call(snd, "To PointProcess (periodic, cc)", 75, 300)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer = call([snd, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    return {
        "meanF0Hz": meanF0,
        "stdevF0Hz": stdevF0,
        "HNR": meanHNR,
        "localJitter": localJitter,
        "localShimmer": localShimmer
    }

@app.route('/')
def index():
    return render_template('index.html', labels=PHONATION_LABELS)

@app.route('/predict', methods=['POST'])
def predict():
    # Get age and sex
    age = request.form.get("age")
    sex = request.form.get("sex")
    
    if not age or not sex:
        return "Please provide Age and Sex", 400
    
    person_data = {
        "Age (years)": int(age),
        "Sex": 0 if sex.upper() == "M" else 1
    }
    
    try:
        # Loop through each phonation label and extract features
        for label in PHONATION_LABELS:
            file = request.files.get(f"file_{label}")  # Changed this to match HTML form names
            if not file:
                return f"Missing audio file for {label}", 400
            
            save_path = os.path.join(UPLOAD_FOLDER, f"{label}.wav")
            file.save(save_path)
            
            features = extract_features(save_path)
            for key, val in features.items():
                person_data[f"{key}_{label}"] = val
        
        # Convert to DataFrame
        df = pd.DataFrame([person_data])
        
        # Prepare features - use all columns from the dataframe
        X = df
        X_scaled = scaler.transform(X)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        result = "ALS Detected" if prediction == 1 else "Healthy Control (HC)"
        
        return render_template("result.html", prediction=result)
    
    except Exception as e:
        return f"Error during prediction: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)