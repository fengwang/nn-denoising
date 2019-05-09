import webbrowser
import threading
import tempfile
from nn_denoising import predict

from flask import Flask, redirect, url_for
from flask import render_template
from flask import request

app = Flask(__name__)
#app.config.update(dict(
#    SECRET_KEY="powerful secretkey",
#    WTF_CSRF_SECRET_KEY="a csrf secret key"
#))

app.config['SECRET_KEY'] = "empty string"

allowed_upload_extensions = ['tif', 'tiff']

@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and ('.' in file.filename) and (file.filename.rsplit('.',1)[1] in allowed_upload_extensions):
            tmp_input_noisy_path = tempfile.NamedTemporaryFile(delete=False).name+file.filename
            file.save( tmp_input_noisy_path )
            tmp_output_denoisedpath = './static/predicted_' + file.filename
            predict( tmp_input_noisy_path, prediction_path=tmp_output_denoisedpath )
            # TODO: here render with download link
            return '<img src=' + url_for('static',filename='predicted_'+file.filename) + '>'
        else:
            return "Error: only files with extension ".join(allowed_upload_extensions) + " are currently supported."
    return render_template('index.html')

@app.route('/')
def index():
    return redirect(url_for('upload') )

def run_app( debug=True, port='8897' ):
    app.run(debug=debug, port=port)


if __name__ == '__main__':
    threading.Thread(target=run_app, args=(False, '1234') ).start()
    webbrowser.open_new( 'http://127.0.0.1:1234' )

