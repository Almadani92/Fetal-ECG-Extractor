import os
import io
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid main thread issues
from flask import Flask, request, redirect, url_for, render_template_string, send_file
from werkzeug.utils import secure_filename
from scipy.io import loadmat, savemat
import numpy as np
import torch
from torch.autograd import Variable
from scipy.signal import butter, filtfilt, iirnotch
from networks_real import build_UNETR
import timeit
import matplotlib.pyplot as plt

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

def process_fecg(inputs):
    print('Begin generating the fetal ECG signal...')
    device = torch.device("cpu")
    net = build_UNETR()
    net.to(device)
    net.load_state_dict(torch.load('saved_model5_japan.pkl', map_location=torch.device('cpu')))

    
    inputs = np.einsum('ijk->jki', inputs)
    inputs = torch.from_numpy(inputs)
    inputs = Variable(inputs).float()
    inputs.to(device)
    
    mecg_pred, fecg_pred = net(inputs)
    return fecg_pred

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'mat'

# Function to process fetal ECG from the uploaded file
def process_fetal_ecg(file_path):
    # Load the .mat file
    mat_data = loadmat(file_path)
    maternal_ecg = mat_data.get('maecg')

    if maternal_ecg is None:
        print(f"Error: 'maecg' key not found in the uploaded .mat file {file_path}")
        return None

    # Preparing the maternal ECG for processing
    maternal_ecg = maternal_ecg[0:992, 0]
    maternal_ecg = butter_bandpass_filter(maternal_ecg, 3, 90, 250, 3)
    maternal_ecg = notch_filter_ecg(maternal_ecg, 250, 50, 30)
    maternal_ecg = (maternal_ecg - np.mean(maternal_ecg)) / np.var(maternal_ecg)
    maternal_ecg = maternal_ecg / np.max(maternal_ecg)
    maternal_ecg = maternal_ecg * 2
    maternal_ecg = np.expand_dims(maternal_ecg, axis=1)  # Add channel dimension
    maternal_ecg = np.expand_dims(maternal_ecg, axis=1)

    # Process using the UNETR model
    fetal_ecg_pred = process_fecg(maternal_ecg)  # Run fetal ECG extraction process
    fetal_ecg_pred = fetal_ecg_pred.cpu().detach().numpy()

    # Save the result as a MAT file in a bytes buffer (in-memory)
    result_buffer = io.BytesIO()
    savemat(result_buffer, {'fetal_ecg_pred': fetal_ecg_pred})
    result_buffer.seek(0)

    # Plot maternal and fetal ECG signals and save the figure
    plt.figure(figsize=(10, 6))

    # Plot maternal ECG
    plt.subplot(2, 1, 1)
    plt.plot(maternal_ecg.flatten(), color='blue')
    plt.title('Maternal Abdominal ECG Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')

    # Plot fetal ECG prediction
    plt.subplot(2, 1, 2)
    plt.plot(fetal_ecg_pred.flatten(), color='green')
    plt.title('Fetal ECG Prediction')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('static/fetal_ecg_plot.png')
    plt.close('all')  # Close all figures

    return result_buffer

# Route for the Results Page
@app.route('/results')
def results_page():
    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <title>Fetal ECG Extraction Results</title>
        <!-- CSS and structure -->
    </head>
    <body>
        <div class="container">
            <h1>Fetal ECG Extraction Results</h1>
            <img src="/static/fetal_ecg_plot.png" alt="Fetal ECG Extraction Plot">
            <br>
            <a class="download-button" href="/download/fetal_ecg_pred" download="fetal_ecg_pred.mat">Download Fetal ECG as .mat File</a>
        </div>
    </body>
    </html>
    '''

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
        <!-- CSS and structure -->
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

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Run the Flask server using the dynamic port provided by Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
