import os
import logging
import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from flask import Flask, request, redirect, url_for, send_file
from io import BytesIO
from networks_real import build_UNETR
import requests
import urllib.request
from scipy.signal import butter, filtfilt, iirnotch

# Set up logging
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['result_buffer'] = None

# Ensure upload and results directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Bandpass filter function
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y
    
def notch_filter_ecg(ecg_signal, sampling_freq=250, notch_freq=50, Q=30):
    nyquist = 0.5 * sampling_freq
    f0 = notch_freq / nyquist
    b, a = iirnotch(f0, Q)
    filtered_ecg = filtfilt(b, a, ecg_signal)
    return filtered_ecg

# Check if uploaded file has allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_model(url, destination):
    print("Downloading the model from Dropbox...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully!")
    else:
        print(f"Error downloading the model: {response.status_code}")

# Function to process fetal ECG
def process_fecg(inputs):
    logging.info('Begin generating the fetal ECG signal...')
    device = torch.device("cpu")
    net = build_UNETR()
    net.to(device)

    # URL of your model in Dropbox or another storage
    model_url = "https://www.dropbox.com/scl/fi/qsev17tj006jwg2iv499k/saved_model5_japan.pkl?rlkey=mte6osrzrg3ys6ck8lgfiji9f&st=x8n9zg6g&dl=1"
    
    # Download the model and save it locally
    model_file_path = "saved_model5_japan.pkl"
    urllib.request.urlretrieve(model_url, model_file_path)
    net.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    
    inputs = np.einsum('ijk->jki', inputs)
    inputs = torch.from_numpy(inputs)
    inputs = Variable(inputs).float().to(device)

    logging.info('Running inference...')
    try:
        mecg_pred, fecg_pred = net(inputs)
        logging.info('Inference completed successfully.')
    except Exception as e:
        logging.error(f'Error during inference: {e}')
        return None

    return fecg_pred

# Function to process the uploaded CSV file
def process_fetal_ecg(file_path):
    logging.info('Loading maternal ECG from .csv file...')
    try:
        # Load CSV as a dataframe
        df = pd.read_csv(file_path, header=None)  # No header in the CSV file
        
        # Assume the maternal ECG is in the first column
        maternal_ecg = df.iloc[:, 0].values
        
        # Preprocess ECG signal: limit to first 992 samples and expand dimensions
        maternal_ecg = maternal_ecg[:992]
        maternal_ecg = butter_bandpass_filter(maternal_ecg, 3, 90, 250, 3)
        maternal_ecg = notch_filter_ecg(maternal_ecg, 250, 50, 30)
        maternal_ecg = (maternal_ecg - np.mean(maternal_ecg)) / np.var(maternal_ecg)
        maternal_ecg = maternal_ecg / np.max(maternal_ecg)
        maternal_ecg = maternal_ecg * 2
        maternal_ecg = np.expand_dims(maternal_ecg, axis=1)  # Add channel dimension
        maternal_ecg = np.expand_dims(maternal_ecg, axis=1)

        # Process using the model
        fetal_ecg_pred = process_fecg(maternal_ecg)  # Run fetal ECG extraction process
        if fetal_ecg_pred is None:
            logging.error('Error during fetal ECG processing.')
            return None

        fetal_ecg_pred = fetal_ecg_pred.cpu().detach().numpy()

        # Save the output to a .csv file
        output_filename = os.path.join(app.config['RESULTS_FOLDER'], 'fetal_ecg_pred.csv')
        pd.DataFrame(fetal_ecg_pred).to_csv(output_filename, header=False, index=False)

        logging.info('Fetal ECG processing complete.')
        return output_filename

    except Exception as e:
        logging.error(f"Error processing the file: {e}")
        return None

# Route to download the extracted fetal ECG .csv file
@app.route('/download/fetal_ecg_pred')
def download_file():
    if 'result_buffer' in app.config and app.config['result_buffer'] is not None:
        return send_file(app.config['result_buffer'], as_attachment=True, download_name='fetal_ecg_pred.csv', mimetype='text/csv')
    return "No file available for download", 404

# Route for Upload Page
@app.route('/', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in the request"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'maecg_signal.csv')
            file.save(file_path)
            
            # Process the uploaded file (CSV file processing and ECG extraction)
            output_file = process_fetal_ecg(file_path)
            if output_file is not None:
                app.config['result_buffer'] = output_file
            
            # Redirect to results page after processing
            return redirect(url_for('results_page'))

    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <title>Upload Maternal ECG .csv File</title>
        <style>
            body {
                background-image: url('/static/full_pipeline.png');
                background-size: contain; /* Ensure the image fits */
                background-position: center; /* Center the image */
                background-repeat: no-repeat; /* Prevent duplication */
                font-family: Arial, sans-serif;
                color: #fff;
                text-align: center;
            }
            .container {
                background-color: rgba(0, 0, 0, 0.6);
                padding: 20px;
                border-radius: 10px;
                margin-top: 300px;
                display: inline-block;
            }
            input[type="file"], input[type="submit"] {
                margin: 10px;
                padding: 10px;
                font-size: 1em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Upload Maternal Abdominal ECG File</h1>
            <form method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".csv">
                <br>
                <input type="submit" value="Upload">
            </form>
        </div>
    </body>
    </html>
    '''

# Route for the Results Page
@app.route('/results')
def results_page():
    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <title>Fetal ECG Extraction Results</title>
        <style>
            body {
                background-image: url('/static/full_pipeline.png');
                background-size: contain;
                background-repeat: no-repeat;
                background-position: center;
                font-family: Arial, sans-serif;
                color: #fff;
                text-align: center;
                height: 100vh;
                margin: 0;
                padding: 0;
                background-attachment: fixed;
            }
            .container {
                background-color: rgba(0, 0, 0, 0.6);
                padding: 20px;
                border-radius: 10px;
                margin-top: 50px;
                display: inline-block;
            }
            img {
                max-width: 90%;
                height: auto;
                margin: 20px 0;
            }
            .download-button {
                padding: 10px 20px;
                margin: 20px;
                background-color: #28a745;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                font-size: 1em;
            }
            .download-button:hover {
                background-color: #218838;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Fetal ECG Extraction Results</h1>
            <img src="/static/fetal_ecg_plot.png" alt="Fetal ECG Extraction Plot">
            <br>
            <a class="download-button" href="/download/fetal_ecg_pred" download="fetal_ecg_pred.csv">Download Fetal ECG as .csv File</a>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Ensure the results folder exists
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    # Run the Flask server using the dynamic port provided by Render or Railway
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
