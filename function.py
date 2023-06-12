import cv2
import os
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

def calcular_precisao(valor):
    valor = valor[0][0]  
    if valor < 0.5 or valor > 0.5:
        diferenca = abs(valor - round(valor))
        precisao = (0.5 - diferenca) * 2
    else:
        diferenca = abs(valor - 0.5)
        precisao = diferenca * -2
    return round(precisao * 100, 2)

@app.route('/analisar-foto', methods=['POST'])
def analisar_foto():
    photo = request.files['photo']
    img = cv2.imdecode(np.fromstring(photo.read(), np.uint8), cv2.IMREAD_COLOR)
    resize = tf.image.resize(img, (256, 256))
    new_model = load_model(os.path.join('C:\\Users\\Sullivan\\Documents\\mypet-flask\\.venv\\models', 'dogHealthClassifier.h5'))
    yhatNew = new_model.predict(np.expand_dims(resize/255, 0))
    precisao = calcular_precisao(yhatNew)

    print("yhatNew:", yhatNew)

    if yhatNew > 0.5:
        resultado = 'Cachorro com sarna'
    else:
        resultado = 'Cachorro saudável'

    response = {
        'resultado': resultado,
        'precisao': precisao
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()
