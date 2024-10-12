import os
import logging
import torch
import numpy as np
from torch.autograd import Variable
from scipy.io import loadmat, savemat
from flask import Flask, request, redirect, url_for, send_file
from io import BytesIO
from networks_real import build_UNETR
import requests
import urllib.request

# Set up loggin
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mat'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['result_buffer'] = None

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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



def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # Filter out keep-alive new chunks
                f.write(chunk)


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
    # Check file size to ensure it's downloaded properly
    if os.path.exists(model_file_path):
        print(f"Model file size: {os.path.getsize(model_file_path)} bytes")
    else:
        print("Model file not downloaded.")
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

# Function to process the uploaded file
def process_fetal_ecg(file_path):
    logging.info('Loading maternal ECG from .mat file...')
    try:
        mat_data = loadmat(file_path)
        maternal_ecg = mat_data.get('maecg')
        if maternal_ecg is None:
            logging.error('Error: No "maecg" key found in the .mat file.')
            return None

        maternal_ecg = maternal_ecg[0:992, 0]
        maternal_ecg = np.expand_dims(maternal_ecg, axis=1)  # Add channel dimension
        maternal_ecg = np.expand_dims(maternal_ecg, axis=1) 

        # Process using the model
        fetal_ecg_pred = process_fecg(maternal_ecg)  # Run fetal ECG extraction process
        if fetal_ecg_pred is None:
            logging.error('Error during fetal ECG processing.')
            return None

        fetal_ecg_pred = fetal_ecg_pred.cpu().detach().numpy()
        # Save the output to a .mat file in memory
        result_buffer = BytesIO()
        savemat(result_buffer, {'fetal_ecg_pred': fetal_ecg_pred})
        result_buffer.seek(0)

        logging.info('Fetal ECG processing complete.')
        return result_buffer

    except Exception as e:
        logging.error(f"Error processing the file: {e}")
        return None

# Route to download the extracted fetal ECG .mat file
@app.route('/download/fetal_ecg_pred')
def download_file():
    if 'result_buffer' in app.config:
        result_buffer = app.config['result_buffer']
        return send_file(result_buffer, as_attachment=True, download_name='fetal_ecg_pred.mat', mimetype='application/x-matlab-data')
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
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'maecg_signal.mat')
            file.save(file_path)
            
            # Process the uploaded file (MAT file processing and ECG extraction)
            result_buffer = process_fetal_ecg(file_path)
            if result_buffer is not None:
                app.config['result_buffer'] = result_buffer
            
            # Redirect to results page after processing
            return redirect(url_for('results_page'))

    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <title>Upload Maternal ECG .mat File</title>
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
                <input type="file" name="file" accept=".mat">
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
            <a class="download-button" href="/download/fetal_ecg_pred" download="fetal_ecg_pred.mat">Download Fetal ECG as .mat File</a>
            <p>When using this resource, please cite the original publication: M. Almadani, L. Hadjileontiadis and A. Khandoker, "One-Dimensional W-NETR for Non-Invasive Single Channel Fetal ECG Extraction," in IEEE Journal of Biomedical and Health Informatics, vol. 27, no. 7, pp. 3198-3209, July 2023, doi: 10.1109/JBHI.2023.3266645...</p>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Run the Flask server using the dynamic port provided by Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
