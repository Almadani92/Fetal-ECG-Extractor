import os
import logging
import torch
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from torch.autograd import Variable
from flask import Flask, request, redirect, url_for, send_file, render_template_string
from io import BytesIO
from networks_real import build_UNETR
import requests
import urllib.request
from scipy.signal import butter, filtfilt, iirnotch
import csv
import matplotlib.pyplot as plt
from scipy.signal import decimate

# Set up logging
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['result_buffer'] = None

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    print("the input shape is ------------------->>>>>>>>>>>>",inputs.shape)
    logging.info('Running inference...')

    mecg_pred, fecg_pred = net(inputs)
    logging.info('Inference completed successfully.')


    return fecg_pred

def process_fetal_ecg(file_path, signal_length):
    logging.info('Loading maternal ECG from .csv file...')
   
    df = pd.read_csv(file_path, header=None)  # No header in the CSV file
    maternal_ecg_all_sig = df.iloc[:, 0].values
    sampling_freq = maternal_ecg_all_sig.shape[0]/signal_length
    downsampling_factor = np.int32(sampling_freq/250)
    print("downsampling_factor is -------------------->",downsampling_factor)
    if downsampling_factor >1:
        maternal_ecg_all_sig = decimate(maternal_ecg_all_sig, downsampling_factor)
    kh = np.int32(maternal_ecg_all_sig.shape[0] / 992)
    maternal_ecg_all_sig = maternal_ecg_all_sig[:992 * kh]
    fecg_pred_all_sig = np.zeros(maternal_ecg_all_sig.shape)

    for i in range(kh):
        maternal_ecg = maternal_ecg_all_sig[992*i:992*(i+1)] 
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
        fecg_pred_all_sig[992*i:992*(i+1)] = fetal_ecg_pred.squeeze() 

    # Stack maternal_ecg and fetal_ecg_pred as two columns
    combined_ecg = np.column_stack((fecg_pred_all_sig, maternal_ecg_all_sig))

    # Save the combined signals to a .csv file
    output_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'fetal_and_maternal_ecg_signals.csv')
    np.savetxt(output_csv_path, combined_ecg, delimiter=",", header="Extracted_Fetal_ECG,Maternal_abdominal_ECG", comments='')

    # Plot subplots for maternal and fetal ECG
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(maternal_ecg_all_sig[-992:], label="Maternal ECG", color='blue')
    ax[0].set_title("Maternal ECG")
    ax[0].set_xlabel("Samples")
    ax[0].set_ylabel("Amplitude")
    ax[0].legend()

    ax[1].plot(fecg_pred_all_sig[-992:], label="Fetal ECG Prediction", color='red')
    ax[1].set_title("Fetal ECG Prediction")
    ax[1].set_xlabel("Samples")
    ax[1].set_ylabel("Amplitude")
    ax[1].legend()

    # Adjust layout and save the plot in the static folder
    plt.tight_layout()
    plot_path = os.path.join('static', 'fetal_ecg_plot.png')
    plt.savefig(plot_path)
    plt.close()

    logging.info('Maternal and fetal ECG processing complete.')
    return output_csv_path


@app.route('/download/fetal_ecg_pred')
def download_file():
    output_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'fetal_and_maternal_ecg_signals.csv')
    if os.path.exists(output_csv_path):
        return send_file(output_csv_path, as_attachment=True, download_name='fetal_and_maternal_ecg_signals.csv', mimetype='text/csv')
    else:
        logging.error('No file available for download.')
        return "No file available for download", 404

        
@app.route('/welcome')
def welcome_page():
    return render_template_string('''
    <!doctype html>
    <html lang="en">
    <head>
        <title>Welcome to the Fetal ECG Extraction Program</title>
        <style>
            body {
                background-color: #f4f4f9;
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                text-align: center;
            }
            .container {
                margin: 50px auto;
                padding: 20px;
                width: 80%;
                background-color: #ffffff;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                border-radius: 10px;
            }
            h1 {
                color: #333;
            }
            p {
                font-size: 1.1em;
                color: #666;
            }
            a {
                color: #007bff;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
            .download-link {
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to the Fetal ECG Extraction Program</h1>
            <p>When using this resource, please cite the original publication:</p>
            <p>M. Almadani, L. Hadjileontiadis, and A. Khandoker, "One-Dimensional W-NETR for Non-Invasive Single Channel Fetal ECG Extraction," 
            <br>in IEEE Journal of Biomedical and Health Informatics, vol. 27, no. 7, pp. 3198-3209, July 2023, doi: 10.1109/JBHI.2023.3266645.</p>
            
            <p>This program allows you to generate the fetal ECG signal from your own maternal abdominal single-lead data.</p>
            <p>The program accepts data in <strong>.csv</strong> format. You can download an example file 
            <a href="https://www.dropbox.com/scl/fi/2fbch6cgffw6ha22d8jqw/maecg.csv?rlkey=d66575fgwuuq3ety23gphhtb1&st=b1bay904&dl=1" class="download-link" download>here</a>.</p>
            
            <p>The minimum data length to be processed is 4 seconds.</p>
            
            <p>We ensure that your uploaded data will not be stored or saved on our servers. The data is processed solely for generating fetal ECG, and once processing is complete, the data is discarded.</p>
            
            <p>This program is designed for research and academic purposes only. It is not intended for commercial or clinical use.</p>
            
            <p>Please note that this tool is created to assist researchers and students, and as such, we cannot guarantee that the generated fetal ECG signal will be 100% accurate.</p>
            
            <p><a href="/upload">Go to Upload Page</a></p>
        </div>
    </body>
    </html>
    ''')

    
# Redirect the root route to /welcome
@app.route('/')
def home_page():
    return redirect(url_for('welcome_page'))
    

# Route for Upload Page
@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in the request"
        if 'signal_length' not in request.form or not request.form['signal_length']:
            return "No signal length provided"

        file = request.files['file']
        signal_length = request.form['signal_length']  # Get the signal length

        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'maecg_signal.csv')
            file.save(file_path)

            # Process the uploaded file (CSV file processing and ECG extraction)
            result_buffer = process_fetal_ecg(file_path, float(signal_length))  # Pass signal length to the processing function
            if result_buffer is not None:
                app.config['result_buffer'] = result_buffer

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
            input[type="file"], input[type="number"], input[type="submit"] {
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
                <input type="file" name="file" accept=".csv" required>
                <br>
                <label for="signal_length">Uploaded Signal Length in Seconds:</label>
                <input type="number" name="signal_length" min="0.1" step="0.001" required>
                <br>
                <input type="submit" value="Upload">
            </form>
        </div>
    </body>
    </html>
    '''
    
    
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
            <a class="download-button" href="/download/fetal_ecg_pred" download="fetal_and_maternal_ecg_signals.csv">Download Fetal ECG as .csv File</a>
            <p>When using this resource, please cite the original publication: M. Almadani, L. Hadjileontiadis and A. Khandoker, "One-Dimensional W-NETR for Non-Invasive Single Channel Fetal ECG Extraction," in IEEE Journal of Biomedical and Health Informatics, vol. 27, no. 7, pp. 3198-3209, July 2023, doi: 10.1109/JBHI.2023.3266645...</p>
        </div>
    </body>
    </html>
    '''


if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Run the Flask server using the dynamic port provided by Render or Railway
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
