# import required libraries
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from generator import buildGenerator, process_file
from werkzeug.utils import secure_filename

import os


# initialize app instance
app = Flask(__name__, static_url_path='/static')

# initialize generator model instance
generator = buildGenerator()
generator.load_weights('AnimeColourisationModel.h5')

# file upload configurations
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# file download configurations
DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/downloads/'
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            process_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename)
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('index.html')
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='localhost', port=port, debug=True)