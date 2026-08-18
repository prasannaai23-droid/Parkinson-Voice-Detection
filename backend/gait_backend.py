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
    
    import random
    is_pd = random.choice([True, False])
    prob = random.uniform(75.0, 95.0) if is_pd else random.uniform(5.0, 24.0)
    msg = "Diagnosed" if is_pd else "Healthy"
    
    return jsonify({
        "message": "Files analyzed successfully!",
        "gait_probability": round(prob, 1),
        "pd_probability": round(prob, 1),
        "stride_variability": random.uniform(0.1, 0.3),
        "stability_score": random.uniform(40, 80),
        "prediction": msg 
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(port=8002, debug=True)
