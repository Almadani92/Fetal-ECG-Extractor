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
def process_fecg(inputs, kh):
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
    
    fecg_pred_all_sig = np.zeros(inputs.shape)

    for i in range(kh):
        maternal_ecg = inputs[992*i:992*(i+1)] 
        maternal_ecg = butter_bandpass_filter(maternal_ecg, 3, 90, 250, 3)
        maternal_ecg = notch_filter_ecg(maternal_ecg, 250, 50, 30)
        maternal_ecg = (maternal_ecg - np.mean(maternal_ecg)) / np.var(maternal_ecg)
        maternal_ecg = maternal_ecg / np.max(maternal_ecg)
        maternal_ecg = maternal_ecg * 2

        maternal_ecg = np.expand_dims(maternal_ecg, axis=1)  # Add channel dimension
        maternal_ecg = np.expand_dims(maternal_ecg, axis=1)
        
        maternal_ecg = np.einsum('ijk->jki', maternal_ecg)
        maternal_ecg = torch.from_numpy(maternal_ecg)
        maternal_ecg = Variable(maternal_ecg).float().to(device)

        logging.info('Running inference...')
        mecg_pred, fecg_pred = net(maternal_ecg)
        logging.info('Inference completed successfully.')


        fecg_pred = fecg_pred.cpu().detach().numpy()
        fecg_pred_all_sig[992*i:992*(i+1)] = fecg_pred.squeeze() 
    
    return fecg_pred_all_sig


def process_fetal_ecg(file_path, signal_length):
    logging.info('Loading maternal ECG from .csv file...')
   
    df = pd.read_csv(file_path, header=None)  # No header in the CSV file
    maternal_ecg_all_sig = df.iloc[:, 0].values
    # maternal_ecg_all_sig = maternal_ecg_all_sig[1:]
    sampling_freq = maternal_ecg_all_sig.shape[0]/signal_length
    downsampling_factor = np.int32(sampling_freq/250)
    print("downsampling_factor is -------------------->",downsampling_factor)
    if downsampling_factor >1:
        maternal_ecg_all_sig = decimate(maternal_ecg_all_sig, downsampling_factor)
    kh = np.int32(maternal_ecg_all_sig.shape[0] / 992)
    maternal_ecg_all_sig = maternal_ecg_all_sig[:992 * kh]
    
    fecg_pred_all_sig = process_fecg(maternal_ecg_all_sig, kh)  # Run fetal ECG extraction process
    
    # Stack maternal_ecg and fetal_ecg_pred as two columns
    combined_ecg = np.column_stack((fecg_pred_all_sig, maternal_ecg_all_sig))

    # Save the combined signals to a .csv file
    output_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'fetal_and_maternal_ecg_signals.csv')
    np.savetxt(output_csv_path, combined_ecg, delimiter=",", header="Extracted_Fetal_ECG,Maternal_abdominal_ECG", comments='')

    # Create a time array from 0 to 4 seconds, assuming 992 samples over 4 seconds
    time_array = np.linspace(0, 4, 992)

    # Plot subplots for maternal and fetal ECG
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Maternal ECG
    ax[0].plot(time_array, maternal_ecg_all_sig[0:992], label="Maternal ECG", color='blue')
    ax[0].set_title("Maternal ECG")
    ax[0].set_xlabel("Time (seconds)")  # Update x-axis label to "Time (seconds)"
    ax[0].set_ylabel("Amplitude")
    ax[0].legend()

    # Fetal ECG Prediction
    ax[1].plot(time_array, fecg_pred_all_sig[0:992], label="Fetal ECG Prediction", color='red')
    ax[1].set_title("Fetal ECG Prediction")
    ax[1].set_xlabel("Time (seconds)")  # Update x-axis label to "Time (seconds)"
    ax[1].set_ylabel("Amplitude")
    ax[1].legend()

    # Adjust layout and save the plot in the static folder
    plt.tight_layout()
    plot_path = os.path.join('static', 'fetal_ecg_plot.png')
    plt.savefig(plot_path)
    plt.close()

    logging.info('Maternal and fetal ECG processing complete.')
    return output_csv_path


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
                text-align: left;  /* Align text to the left */
            }
            .container {
                margin-left: 30px;  /* Add left margin */
                padding: 20px;
                width: 90%;  /* Make the container wider */
                background-color: #ffffff;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                border-radius: 10px;
            }
            h1, h2 {
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
            /* Style for the citation box */
            .citation-box {
                background-color: #f0f0f0;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #ccc;
                margin-bottom: 20px;
                color: #333;
            }
            /* Style for the author section */
            .authors {
                margin-top: 50px;
            }
            .authors h2 {
                margin-bottom: 20px;
            }
            .author {
                display: flex;  /* Use flex to align items horizontally */
                align-items: flex-start;  /* Align text with the top of the image */
                margin-bottom: 40px;  /* Add spacing between authors */
            }
            .author img {
                width: 150px;
                height: 150px;
                border-radius: 50%;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                margin-right: 20px;  /* Space between the image and the text */
            }
            .author-info {
                max-width: 600px;  /* Limit the width of the text */
            }
            .author-info p {
                margin: 5px 0;  /* Reduce spacing between paragraphs */
                color: #333;
                font-weight: bold;
            }
            .author-info .bio {
                font-weight: normal;
                color: #666;
                font-size: 0.95em;
            }
            /* Style for the "Go to Upload Page" button */
            .upload-link {
                display: block;  /* Make it a block element */
                text-align: center;  /* Center the text */
                background-color: #28a745;  /* Green background */
                color: white;  /* White text */
                padding: 15px 20px;  /* Padding for the box */
                border-radius: 5px;  /* Rounded corners */
                margin: 40px auto;  /* Center it with top/bottom margin */
                width: 200px;  /* Fixed width */
                text-decoration: none;  /* No underline */
                font-weight: bold;  /* Bold text */
            }
            .upload-link:hover {
                background-color: #218838;  /* Darker green on hover */
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to the Fetal ECG Extraction Program</h1>
            <div class="citation-box">
                <p>If you use this tool in your research, please cite the following publication:</p>
                <p>M. Almadani, L. Hadjileontiadis, and A. Khandoker, "One-Dimensional W-NETR for Non-Invasive Single Channel Fetal ECG Extraction," 
                <br>in IEEE Journal of Biomedical and Health Informatics, vol. 27, no. 7, pp. 3198-3209, July 2023, doi: 10.1109/JBHI.2023.3266645.</p>
            </div>
            
            <p>This program allows you to generate fetal ECG signals from single-lead maternal abdominal ECG data. It is designed for researchers and students working in biomedical engineering and related fields.</p>

            <p>The program accepts data in <strong>.csv</strong> format. You can download an example file 
            <a href="https://www.dropbox.com/scl/fi/2fbch6cgffw6ha22d8jqw/maecg.csv?rlkey=d66575fgwuuq3ety23gphhtb1&st=b1bay904&dl=1" class="download-link" download>here</a>.</p>
            
            <p>Please note that the minimum required data length for processing is 4 seconds. Any input shorter than this will not be processed.</p>

            <p>We take your privacy seriously. Your uploaded data is not stored or saved on our servers. It is processed solely for generating the fetal ECG signal, and all data is automatically deleted after the processing is complete.</p>
            
            <p>This tool is intended for research and educational purposes only. It is not designed or approved for clinical or commercial use.</p>
            
            <p>While we strive to make this tool as accurate as possible, please be aware that the generated fetal ECG signal may not be 100% accurate. Use it as a helpful resource, but always validate results against your own data and research.</p>

            <!-- Author section with biographies -->
            <div class="authors">
                <h2>Meet the authors of this project:</h2>
                
                <!-- Murad Almadani -->
                <div class="author">
                    <img src="/static/Murad.png" alt="Murad Almadani">
                    <div class="author-info">
                        <p>Murad Almadani (Coressponding Author)</p>
                        <p class="bio">Murad Almadani is a Ph.D. researcher in the Biomedical Engineering department at Khalifa University, UAE. His research focuses on signal and image processing, computer vision, and artificial intelligence.</p>
                    </div>
                </div>

                <!-- Leontios J. Hadjileontiadis -->
                <div class="author">
                    <img src="/static/Leontios.jpg" alt="Leontios J. Hadjileontiadis">
                    <div class="author-info">
                        <p>Leontios J. Hadjileontiadis</p>
                        <p class="bio">Leontios J. Hadjileontiadis is a Professor and Chair of the Biomedical Engineering department at Khalifa University, UAE, as well as a Professor at ECE-Aristotle University of Thessaloniki. His research spans advanced signal processing, machine learning, intelligent biomedical systems, and biomusic composition.</p>
                    </div>
                </div>

                <!-- Ahsan Khandoker -->
                <div class="author">
                    <img src="/static/Ahsan.png" alt="Ahsan Khandoker">
                    <div class="author-info">
                        <p>Ahsan Khandoker</p>
                        <p class="bio">Ahsan Khandoker is a Professor and theme leader of the Healthcare Engineering Innovation Center (HEIC) at Khalifa University. His multidisciplinary research covers bio-signal processing, bioinstrumentation, nonlinear modeling, and artificial intelligence applied to sleep apnea, cardiovascular diseases, fetal cardiology, and psychiatry.</p>
                    </div>
                </div>
            </div>

            <p>If you require any assistance or have questions, please feel free to contact the corresponding author at <a href="mailto:murad.almadani@gmail.com">murad.almadani@gmail.com</a>.</p>

            <a href="/upload" class="upload-link">Go to Upload Page</a>  <!-- Updated link to be centered in a green box -->
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
                font-family: Arial, sans-serif;
                color: #333;
                text-align: center;
            }
            .container {
                padding: 20px;
                border-radius: 10px;
                display: inline-block;
            }
            img {
                max-width: 90%;
                height: auto;
                margin: 20px 0;
            }
            .feedback-form {
                margin-top: 30px;
            }
            .feedback-form a {
                text-decoration: none;
                background-color: #28a745;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            .download-link {
                margin-top: 20px;
                display: block;
                text-decoration: none;
                background-color: #007bff;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            /* Style for the citation box */
            .citation-box {
                background-color: #f0f0f0;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #ccc;
                margin-top: 40px;
                color: #333;
                max-width: 700px;
                margin: 40px auto;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Fetal ECG Extraction Results of First 4 Seconds of Uploaded Data</h1>
            <img src="/static/fetal_ecg_plot.png" alt="Fetal ECG Extraction Plot">

            <!-- Feedback Form Link -->
            <div class="feedback-form">
                <h3>We value your feedback!</h3>
                <p>Please <a href="https://docs.google.com/forms/d/e/1FAIpQLSd_mb6cEj5CioG1j_y343KoBnrHbV6XqvIb5w2uit7pZs0mBA/viewform?usp=sf_link" target="_blank">click here</a> to rate the program and provide suggestions.</p>
            </div>

            <!-- Download Results Button -->
            <a href="/download/fetal_ecg_pred" class="download-link">Download Full Results (.csv)</a>

            <!-- Citation Box -->
            <div class="citation-box">
                <p>If you use this tool in your research, please cite the following publication:</p>
                <p>M. Almadani, L. Hadjileontiadis, and A. Khandoker, "One-Dimensional W-NETR for Non-Invasive Single Channel Fetal ECG Extraction, in IEEE Journal of Biomedical and Health Informatics, vol. 27, no. 7, pp. 3198-3209, July 2023, doi: 10.1109/JBHI.2023.3266645" 
            </div>
        </div>
    </body>
    </html>
    '''


@app.route('/download/fetal_ecg_pred')
def download_file():
    output_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'fetal_and_maternal_ecg_signals.csv')
    if os.path.exists(output_csv_path):
        return send_file(output_csv_path, as_attachment=True, download_name='fetal_and_maternal_ecg_signals.csv', mimetype='text/csv')
    else:
        logging.error('No file available for download.')
        return "No file available for download", 404


if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Run the Flask server using the dynamic port provided by Render or Railway
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
