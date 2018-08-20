from flask import Flask, flash, redirect, render_template, request, session, abort, url_for
import glob
import numpy as np
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
import os
from ImageRetreival import ImageRetrieval
import argparse

app = Flask(__name__, static_url_path='')

filenames = []
file_urls = []


dropzone = Dropzone(app)

# Dropzone settings

app.config['DROPZONE_UPLOAD_MULTIPLE'] = False
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'
# TODO this.
app.config['SECRET_KEY'] = 'supersecretkey'


app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)


@app.route("/", methods=['GET', 'POST'])
def index():
    if "file_urls" not in session:
        session['file_urls'] = []
    file_urls = session['file_urls']

    if request.method == 'POST':
        print('inPostMethod')
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            filename = photos.save(file, name=file.filename)
            file_urls.append(photos.url(filename))
            print(filename)
        session['file_urls'] = file_urls
        return "uploading"
    return render_template('index.html')


@app.route("/test")
def test():
    return "Test Works"


@app.route("/results")
def results():
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('index'))

    file_urls = session['file_urls']
    session.pop('file_urls', None)
    print(file_urls)
    return deepretrieval(os.path.join('./uploads', file_urls[0].split('/')[-1]))
    # return render_template('deepretrieval.html',name=file_urls[0])


@app.route("/deepretrieval/<string:name>/")
def deepretrieval(name):
    match = ig.get_match(name)
    print('MatherFHFHHFFHHFHF', match[0])
    random_image = filenames[match[0]]
    random_image = '/'.join(random_image.split('/')[1:])
    print(random_image)
    return render_template('retrieval_main.html', **locals())
    # return "Hello World"
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Split images from folder into test and validation')
    parser.add_argument('--dataset_folder', dest='folder_name',
                        help='Folder with dataset', required=True)
    parser.add_argument(
        '--create_index', action='store_true', dest='create_index')
    args = parser.parse_args()
    folder_name = args.folder_name

    filenames = glob.glob('static/PetImages/**/*.jpg')
    ig = ImageRetrieval(folder_name)
    ig.create_index(args.create_index)
    app.run(host='0.0.0.0', port=5050, debug=True)
