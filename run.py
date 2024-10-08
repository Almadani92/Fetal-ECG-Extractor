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

@app.route('/', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in the request"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            try:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'maecg_signal.mat')
                file.save(file_path)

                # Process the uploaded file (MAT file processing and ECG extraction)
                logging.info('Processing the uploaded file...')
                result_buffer = process_fetal_ecg(file_path)
                if result_buffer is not None:
                    app.config['result_buffer'] = result_buffer
                    # Redirect to results page after processing
                    return redirect(url_for('results_page'))
                else:
                    logging.error("Failed to generate fetal ECG.")
                    return "Failed to process the ECG file. Please check the server logs for more details."

            except Exception as e:
                logging.error(f"Error processing the file: {e}")
                return "Internal server error, please check the logs."

    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <title>Upload Maternal ECG .mat File</title>
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

@app.route('/results', methods=['GET'])
def results_page():
    if app.config['result_buffer'] is not None:
        return send_file(
            app.config['result_buffer'],
            as_attachment=True,
            download_name='fetal_ecg_pred.mat',
            mimetype='application/x-matlab-data'
        )
    else:
        return "No result available. Please upload and process an ECG file first."

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Run the Flask server using the dynamic port provided by Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
