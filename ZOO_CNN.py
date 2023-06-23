import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
#from keras.applications.densenet import DenseNet201, preprocess_input
from keras.applications.nasnet import NASNetMobile, NASNetLarge, preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Input
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
import tensorflow as tf

#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)


classes = ['Class1.1', 'Class1.2', 'Class1.3', 'Class2.1', 'Class2.2', 'Class3.1','Class3.2', 'Class4.1', 'Class4.2', 'Class5.1', 'Class5.2', 'Class5.3','Class5.4', 'Class6.1', 'Class6.2', 'Class7.1', 'Class7.2', 'Class7.3','Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7', 'Class9.1', 'Class9.2', 'Class9.3', 'Class10.1', 'Class10.2', 'Class10.3', 'Class11.1', 'Class11.2', 'Class11.3', 'Class11.4', 'Class11.5', 'Class11.6']

def ext(nm):
    return nm + ".jpg"

df_train = pd.read_csv("training_solutions_rev1.csv")

#print(df_train.head())

df_train["ID"] = df_train["GalaxyID"].astype(str).apply(ext)

#print(df_train["ID"])

image = ImageDataGenerator(fill_mode='nearest', cval=0, rescale=1./255, rotation_range=270, zoom_range = 0.1 ,horizontal_flip=True, vertical_flip=True, validation_split=0.2)

train_generator = image.flow_from_dataframe(dataframe= df_train,directory="coloca el directorio del archivo donde estan las imagenes de entrenamiento",
        x_col="ID",
        y_col=classes,
        subset="training",
        batch_size=22,
        seed=4,
        class_mode="other", 
        target_size=(224, 224))

valid_generator = image.flow_from_dataframe(
    dataframe=df_train,
    directory="directory",
    x_col="ID",
    y_col=classes,
    subset="validation",
    batch_size=22,
    seed=4,
    class_mode="other",
    target_size=(224, 224))

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

#llamando al modelo pre-entrenado de keras (en mi caso elegi NasNetMobile)

# modificamos las imagenes de entradas para que sean igual a las imagenes de nuestro set de datos
img_reshape = (224,224,3)

modelo_DN = NASNetMobile(include_top = False,
                        input_shape = img_reshape)
#include_top nos permite decirle al modelo si se incluye la capa totalmente conectada en la parte superior de la red

Flat = Flatten()(modelo_DN.output)

#Agregamos el numero de salidas, en nuestro caso son 37 (clases)
Flat = Dense(len(classes), activation = 'sigmoid')(Flat)

MODELO = Model(inputs = modelo_DN.input, outputs = Flat)

#print(MODELO.summary())

for layer in MODELO.layers:
    layer.trainable = True
#Este for fue para reentrenar todas las neuronas

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, decay = 0.0004)

MODELO.compile(optimizer, loss = 'mse', metrics = ["accuracy"])

#Se usara Callbacks de keras para obtener los detalles de nuestra red

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, verbose=1, mode='auto')

history = LossHistory()

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(
        filepath='Model.hdf5',
    verbose=1,
    save_best_only=True)

hist = MODELO.fit_generator(train_generator,
                  steps_per_epoch=STEP_SIZE_TRAIN,
                  validation_data=valid_generator,
                  validation_steps=STEP_SIZE_VALID,
                  epochs= 60,
                  callbacks=[history, checkpointer, early_stopping])
                            
plt.figure(figsize=(12, 8))
plt.plot(hist.epoch, hist.history['loss'], label='Training Loss')
plt.plot(hist.epoch, hist.history['val_loss'], label='Validation', linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.legend()
plt.show()








