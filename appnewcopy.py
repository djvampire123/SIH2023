from concurrent.futures import ThreadPoolExecutor
import logging
from flask import Flask, jsonify, redirect, request, render_template, send_from_directory, url_for
from flask_executor import Executor
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
from flask import Flask, render_template, session
from flask import Flask, jsonify, redirect, request, render_template, send_from_directory, url_for
from flask_executor import Executor
import os
import subprocess
import threading

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
STATIC_FOLDER = 'static'  # Set this to your static folder path

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

executor = Executor()
executor.init_app(app)

def run_extraction_script():
    subprocess.run(['python', 'extract_and_save_frames.py'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/project', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        magnification_factor = request.form.get('magnificationFactor', type=float)
        amplification_factor = request.form.get('amplificationFactor', type=float)

        if magnification_factor is not None:
            mag_fac_path = os.path.join('uploads', f'mag_fac.txt')
            with open(mag_fac_path, 'w') as mag_file:
                mag_file.write(str(magnification_factor))
                
        if amplification_factor is not None:
            amp_fac_path = os.path.join('uploads', f'amp_fac.txt')
            with open(amp_fac_path, 'w') as amp_file:
                amp_file.write(str(amplification_factor))
        if file:
            filename = "originalvideo.mp4"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            folder_path = 'static'
            file_name = 'frame_0.jpg'
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
            # Run extraction script in a separate thread
            threading.Thread(target=run_extraction_script).start()
            return render_template('processing.html') 
    return render_template('project.html')


@app.route('/check-file')
def check_file():
    file_exists = os.path.exists(os.path.join(app.config['STATIC_FOLDER'], 'frame_0.jpg'))
    return jsonify({'file_exists': file_exists})


@app.route('/selector')
def selector():
    return render_template('selector.html')

coords_dir = 'uploads'

def get_next_coords_filename():
    existing_files = [f for f in os.listdir(coords_dir) if f.startswith('coords') and f.endswith('.txt')]
    file_count = len(existing_files)
    return os.path.join(coords_dir, f'coords{file_count + 1}.txt')

@app.route('/save_coordinates', methods=['POST'])
def save_coordinates():
    try:
        data = request.data.decode("utf-8")  # Read the plain text data
        if data:
            coords_filename = get_next_coords_filename()
            with open(coords_filename, 'w') as file:
                file.write(data)
            return jsonify({'message': f'Coordinates saved to {coords_filename} successfully'})
        else:
            return jsonify({'message': 'No data received'})
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'})
        
@app.route('/run_scripts', methods=['GET', 'POST'])
def run_scripts():
    try:
        # Paths for the scripts (modify as needed)
        mask = 'mask_gen.py'
        ampli = 'ampli.py'
        magni = 'magnification.py'
        ana = 'freq_anlst.py'
        
        subprocess.run(['python', mask], check=True)
        subprocess.run(['python', magni], check=True)
        subprocess.run(['python', ana], check=True)

        with ThreadPoolExecutor() as executor:
            ampli_future = executor.submit(subprocess.run, ['python', ampli], check=True)
            ampli_future.result()

        # Return statement inside the try block
        return redirect(url_for('index'))
    
    except Exception as e:
        # Handle the exception here
        return jsonify({'error': str(e)})
    
@app.route('/back_to_index')
def back_to_index():
    return redirect(url_for('index'))


@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True)