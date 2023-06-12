## Importante: não execute este arquivo, pois isso fará com que o 
# sistema não funcione corretamente.
from PIL import Image
import hashlib
import sqlite3
import tensorflow as tf 
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.metrics import Precision, Recall, BinaryAccuracy
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import imgaug as ia
from imgaug import augmenters as iaa

# 1.0 Instalar dependências e configuração
# Evitar erros de falta de memória definindo o crescimento do consumo de memória da GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.list_physical_devices('GPU')

# 1.2 Remover imagens problemáticas

data_dir = 'data' 
image_exts = ['jpeg','jpg', 'bmp', 'png']
for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Imagem não está na lista de extensões {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Problema com a imagem {}'.format(image_path))
            # os.remove(image_path)
            
# 1.3 Carregar os Dados

data = tf.keras.utils.image_dataset_from_directory('data')

# 1.4 Escalar os Dados

data = data.map(lambda x,y: (x/255, y))
data.as_numpy_iterator().next()

# 1.5 Dividir os Dados

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

# 1.5.1 Aplicar a Augmentação de Dados
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
augmentation = iaa.Sequential([
    sometimes(iaa.Fliplr(0.5)), # flip horizontalmente
    sometimes(iaa.Affine(
        rotate=(-20, 20), # girar de -20 a +20 graus
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)} # escalar de 80% a 120%
    ))
])

# Função para aplicar a augmentação de dados
def apply_augmentation(images, labels):
    augmented_images = tf.numpy_function(lambda x: augmentation.augment_images(x), [images], tf.float32)
    augmented_images = tf.ensure_shape(augmented_images, (batch_size, 256, 256, 3))
    return augmented_images, labels


# Aplicar a augmentação de dados ao conjunto de treinamento
batch_size = 32  # Definir o tamanho do lote
train_augmented = train.map(apply_augmentation)

# 1.6 Construir o Modelo de Aprendizado Profundo

model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# 1.6.1 Compilar o Modelo de Aprendizado Profundo
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy', Precision(), Recall()])

model.summary()

# 1.7 Treinar
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
# ...

epochs = 20  # Defina o número de epochs como um valor inteiro
hist = model.fit(train_augmented, epochs=epochs, validation_data=val, callbacks=[tensorboard_callback, es])

# ...

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# 1.9 Avaliar
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)


model.save(os.path.join('models','imageclassifier_modificado.h5'))
# %%
