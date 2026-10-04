from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/predict-gait', methods=['POST'])
def predict_gait():
    # Support 'gait' from original gait.html and 'motion_clip'/'csv_data' for updates
    if 'gait' not in request.files and 'motion_clip' not in request.files and 'csv_data' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    
    saved_files = []
    
    for key in ['gait', 'motion_clip', 'csv_data']:
        if key in request.files:
            file = request.files[key]
            if file.filename != '':
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                saved_files.append(filename)
    
    import joblib
    import numpy as np
    from train_video_gait_model import extract_features_from_video
    
    try:
        # Load model
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'video_gait_rf.pkl')
        if not os.path.exists(model_path):
            raise Exception("Model not trained yet.")
            
        model = joblib.load(model_path)
        
        # We assume the first uploaded file is the video
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_files[0])
        
        # Extract features
        features = extract_features_from_video(video_path, skip_frames=5)
        if features is None or sum(features) == 0:
            raise Exception("Could not extract features from video.")
            
        features = np.array(features).reshape(1, -1)
        
        # Predict
        prob_pd = model.predict_proba(features)[0][1] * 100
        is_pd = prob_pd > 50
        
        msg = "Pattern consistent with Parkinson's" if is_pd else "Healthy gait pattern detected"
        
        return jsonify({
            "message": msg,
            "gait_probability": round(prob_pd, 1),
            "pd_probability": round(prob_pd, 1),
            "stride_variability": features[0][1] / (features[0][0] + 1e-5), # std/mean of ankle distances
            "stability_score": max(0, 100 - (features[0][2] * 100)), # Some pseudo-stability score based on ptp
            "prediction": "Diagnosed" if is_pd else "Healthy"
        })
        
    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({"error": "Gait analysis could not be completed"}), 422

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(port=8002, debug=True)
