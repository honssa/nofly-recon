import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import numpy as np
import Config
from metricas import Metricas
import cv2


def crear_modelo():

    from tensorflow.keras import layers
    inputs = tf.keras.Input(shape=(Config.IMX_ALTO, Config.IMX_ANCHO, 3))
    modelo = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    modelo = layers.MaxPooling2D((2, 2))(modelo)
    modelo = layers.Conv2D(64, (3, 3), activation='relu')(modelo)
    modelo = layers.MaxPooling2D((2, 2))(modelo)
    modelo = layers.Conv2D(64, (3, 3), activation='relu')(modelo)
    modelo = layers.Flatten()(modelo)
    modelo = layers.Dense(64, activation='relu')(modelo)
    outputs = layers.Dense(3, activation='softmax')(modelo)
    modelo = tf.keras.models.Model(inputs, outputs)
    return modelo


def cargar_datos(directorio):
    carpetas = os.listdir(directorio)
    imaxes = []; etiquetas = []
    for idc, carpeta in enumerate(carpetas):
        for ficheiro in os.listdir(os.path.join(directorio, carpeta)):
            imx = cv2.imread(os.path.join(directorio, carpeta, ficheiro))
            imaxes.append(imx); etiquetas.append(idc)
    return np.asarray(imaxes), etiquetas


if __name__=="__main__":

    DATASET_TAM = 0
    for root, dirs, fichs in os.walk(Config.RUTA_DATASET):
        DATASET_TAM += len(fichs)
    
    clases = []
    for idc, carpeta in enumerate(os.listdir(Config.RUTA_DATASET)):
        clases.append(carpeta)
    
    m = Metricas(Config.CLASES)
    m.inicializar_semente()

    callbacks_lista = []
    callbacks_lista.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
        min_delta=Config.MIN_DELTA, patience=Config.PACIENCIA, 
        verbose=1, mode='min'))
    callbacks_lista.append(tf.keras.callbacks.ModelCheckpoint(filepath='./conv_simple.h5', 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min'))

    adam = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )

    for k in range(Config.NUM_ITERACIONS):

        imaxes, etiquetas = cargar_datos(Config.RUTA_DATASET)
        X_ad, X_test, Y_ad, Y_test = train_test_split(imaxes, etiquetas, test_size=Config.TEST_SPLIT, random_state=Config.SEMENTE)
        Y_ad = tf.keras.utils.to_categorical(Y_ad, num_classes=len(np.unique(etiquetas)))
        Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=len(np.unique(etiquetas))) 
        modelo = crear_modelo()
        modelo.compile(optimizer=adam, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        modelo.summary()
        progresion = modelo.fit(X_ad, Y_ad, epochs=Config.EPOCHS, batch_size=Config.BATCH_TAM, 
                           validation_split=Config.VALIDACION_SPLIT, callbacks=callbacks_lista)

        prediccion = modelo.predict(X_test, batch_size=32)
        m.anotar_metricas(np.argmax(Y_test, axis=1), np.argmax(prediccion,axis=1))
        #cnf_matrix = m.compute_confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(prediccion,axis=1))

    #modelo.save('mc_estradas-libre.h5')
    m.calcular_media_dt()
    m.grafica_adestramento(progresion)
