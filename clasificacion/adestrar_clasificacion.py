import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import numpy as np
import Config
from metricas import Metricas
import cv2


def crear_modelo():
    modelo_inicial = tf.keras.applications.resnet50.ResNet50(weights="imagenet", 
                    input_shape=(Config.IMX_ANCHO,Config.IMX_ALTO,3),
                    include_top=False)
    for i in range(len(modelo_inicial.layers) - 1):
        modelo_inicial.layers[i].trainable = False

    modelo = modelo_inicial.layers[len(modelo_inicial.layers) - 2].output
    modelo = tf.keras.layers.Flatten()(modelo)
    modelo = tf.keras.layers.Dense(400, activation='relu')(modelo)
    modelo = tf.keras.layers.Dense(3, activation='softmax')(modelo)
    modelo = tf.keras.models.Model(modelo_inicial.input,  modelo)
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
    
    print(clases) 
    m = Metricas(clases)
    m.inicializar_semente()

    callbacks_lista = []
    callbacks_lista.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
        min_delta=Config.MIN_DELTA, patience=Config.PACIENCIA, 
        verbose=1, mode='min'))
    callbacks_lista.append(tf.keras.callbacks.ModelCheckpoint(filepath='./novo_modelo.h5', 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min'))

    for k in range(Config.NUM_ITERACIONS):

        imaxes, etiquetas = cargar_datos(Config.RUTA_DATASET)
        X_ad, X_test, Y_ad, Y_test = train_test_split(imaxes, etiquetas, test_size=Config.TEST_SPLIT, random_state=Config.SEMENTE)
        Y_ad = tf.keras.utils.to_categorical(Y_ad, num_classes=len(np.unique(etiquetas)))
        Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=len(np.unique(etiquetas))) 
        modelo = crear_modelo()
        #modelo.summary()
        modelo.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        progresion = modelo.fit(X_ad, Y_ad, epochs=Config.EPOCHS, batch_size=Config.BATCH_TAM, 
                           validation_split=Config.VALIDACION_SPLIT, callbacks=callbacks_lista)

        prediccion = modelo.predict(X_test, batch_size=32)
        m.anotar_metricas(np.argmax(Y_test, axis=1), np.argmax(prediccion,axis=1))
        #cnf_matrix = m.compute_confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(prediccion,axis=1))

    #modelo.save('mc_estradas-libre.h5')
    m.calcular_media_dt()
    m.grafica_adestramento(progresion)
