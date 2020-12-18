import os
from flask import Flask, request, redirect, url_for,flash,render_template,Response, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import json
import base64
import cv2
from model import model
from torchvision import transforms
from util import util
import matplotlib.pyplot as plt
try:
    from StringIO import StringIO ## for Python 2
except ImportError:
    from io import StringIO, BytesIO ## for Python 3


UPLOAD_FOLDER = 'static/data'


app = Flask(__name__)




@app.route('/', methods=['GET', 'POST'])
def upload_file():

        return render_template('Demo.html')


@app.route('/get_image', methods=['POST'])
def get_image():
    file = request.files['imgInp']
    fast = request.form['type']
    if fast == 'fast':
        imsize = 256
    else:
        imsize = 512
    # Read the image via file.stream
    
    image_string = base64.b64encode(file.read())
    image = Image.open(file.stream)

    img_np=image.resize((imsize,imsize))#(image)
    img_np=transforms.ToTensor()(img_np)
    if img_np.shape[0] == 3:
        img_np = img_np[:3,:,:]
    image = model(img_np.unsqueeze(dim=0))
    img = util.tensor2im(image[:1,:,:,:])
    pil_img = Image.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    image_string = base64.b64encode(buff.getvalue())
    plt.imsave('test.jpg',img)
    
    #cv2.imwrite('test.jpg',image[0,:,:,:].detach().cpu().numpy().transpose(1,2,0))
    # print(img_np)
    # image.save('test.jpg')
    response = app.response_class(
        response=json.dumps(image_string.decode('utf-8')),
        status=200,
        mimetype='application/json'
    )
    # print(str(image_string))
    return response
    # return jsonify({'msg': 'success', 'size': str(image_string)})

app.run(debug=True,host='0.0.0.0')