import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import numpy as np
import Config
import cv2
import pandas as pd
import shutil
from sklearn.metrics import cohen_kappa_score
def cargar_datos(directorio):
    carpetas = os.listdir(directorio)
    imaxes = []; etiquetas = []
    for idc, carpeta in enumerate(carpetas):
        for ficheiro in os.listdir(os.path.join(directorio, carpeta)):
            imx = cv2.imread(os.path.join(directorio, carpeta, ficheiro))
            imaxes.append(imx); etiquetas.append(idc)
    return np.asarray(imaxes), etiquetas

def gardar_progresion(prograsion, ficheiro):
    df = pd.DataFrame(progresion.history)
    with open(ficheiro, 'w') as f:
        df.to_csv(f)

def mkdir_se_non_existe(direccion):
    try:
        os.mkdir(direccion)
    except FileExistsError:
        pass

def crear_modelo_vello(num_capas_conxeladas):
    modelo_inicial = tf.keras.applications.resnet50.ResNet50(weights="imagenet", 
                    input_shape=(Config.IMX_ANCHO,Config.IMX_ALTO,len(Config.CLASES)),
                    include_top=False)

    modelo = tf.keras.layers.GlobalMaxPooling2D()(modelo_inicial.layers[-1].output)
    modelo = tf.keras.layers.Dense(len(Config.CLASES), activation=tf.keras.activations.softmax)(modelo)
    modelo = tf.keras.models.Model(inputs=modelo_inicial.input, outputs=modelo)
    
    for i in range(0, num_capas_conxeladas):
        modelo.layers[i].trainable = False

    return modelo


def crear_modelo(num_capas_conxeladas):
    modelo_inicial = tf.keras.applications.resnet50.ResNet50(weights="imagenet", 
                    input_shape=(Config.IMX_ANCHO,Config.IMX_ALTO,len(Config.CLASES)),
                    include_top=False)

    modelo = modelo_inicial.layers[len(modelo_inicial.layers) - 2].output
    modelo = tf.keras.layers.Flatten()(modelo)
    modelo = tf.keras.layers.Dense(400, activation='relu')(modelo)
    modelo = tf.keras.layers.Dense(len(Config.CLASES), activation=tf.keras.activations.softmax)(modelo)
    modelo = tf.keras.models.Model(inputs=modelo_inicial.input, outputs=modelo)
    
    for i in range(0, num_capas_conxeladas):
        modelo.layers[i].trainable = False

    return modelo


if __name__=="__main__":

    # CREAR DIRECTORIOS PARA GARDAR A PROGRESION E OS MODELOS
    dir_progresion = "history"
    dir_modelo = "models"
    mkdir_se_non_existe(dir_progresion)
    mkdir_se_non_existe(dir_modelo)
    resultados_ficheiro = 'resultados.csv'
    columnas = ['CAPAS CONXELADAS', 'LR INICIAL', 'LR DEGRADACION', 'TAM DE BATCH', 'PERDA ADESTRAMENTO',
                'PRECISION ADESTRAMENTO', 'PERDA VALIDACION', 'PRECISION VALIDACION', 'KAPPA']
    if (not os.path.isfile("./resultados.csv")):
        resultados = pd.DataFrame(columns = columnas)
        start = 0
    else:
        with open(resultados_ficheiro, 'r') as f:
            resultados = pd.read_csv(f, index_col = 0)
        start = len(resultados.index)
     
    for k in range(start,start+20):
        imaxes, etiquetas = cargar_datos(Config.RUTA_DATASET)
        X_ad, X_val, Y_ad, Y_val = train_test_split(imaxes, etiquetas, test_size=Config.TEST_SPLIT, random_state=1)
        Y_ad = tf.keras.utils.to_categorical(Y_ad, num_classes=len(Config.CLASES))
        Y_val = tf.keras.utils.to_categorical(Y_val, num_classes=len(Config.CLASES)) 

        # INCIALIZAR HIPERPARAMETROS

        exp = np.random.uniform(1,3)
        lr_inicial = 10 ** (-exp)
        lr_degrad = np.random.uniform(0.7, 1)
        num_capas_conxeladas = np.random.randint(150, 176)
        batch_tam = 2 ** np.random.randint(3,8)

        # IDENTIFICAR OS 3 MELLORES MODELOS

        if (start > 0):
            mellores_modelos = resultados['KAPPA']
            mellores_modelos = mellores_modelos.copy()
            mellores_modelos = mellores_modelos.sort_values(ascending=False)
            ultimo_lugar = mellores_modelos.index[2]
            ultima_metrica = mellores_modelos[ultimo_lugar]
        
        # CREAR E ADESTRAR O MODELO

        callbacks_lista = []
        callbacks_lista.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                min_delta=Config.MIN_DELTA, patience=Config.PACIENCIA, 
                                verbose=1, mode='min'))
        print(f"Adestrando cos hiperparametros: \
            \nNumero de capas conxeladas: {num_capas_conxeladas}, \
            \nRatio de aprendizaxe inicial: {lr_inicial}, \
            \nRatio de degradacion da aprendizaxe: {lr_degrad}, \
            \nTamaño de lote: {batch_tam}")

        modelo = crear_modelo(num_capas_conxeladas)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = lr_inicial,
                decay_steps = 500,
                decay_rate = lr_degrad)
        modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), 
                        loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        progresion = modelo.fit(X_ad, Y_ad, epochs=Config.EPOCHS, batch_size=batch_tam, 
                validation_split=Config.VALIDACION_SPLIT, callbacks=callbacks_lista)

        # GARDAR RESULTADOS

        train_perf = modelo.evaluate(X_ad, Y_ad, verbose=1)
        val_perf = modelo.evaluate(X_val, Y_val, verbose=1)
        
        prediccion = modelo.predict(X_val, batch_size=batch_tam)
        #print("Orixinais: " + str(np.argmax(Y_val, axis=1)))
        #print("Predecidas: " + str(np.argmax(Y_val, axis=1)))
        kappa = cohen_kappa_score(np.argmax(Y_val, axis=1), np.argmax(prediccion, axis=1))
        hyper_param_arr = [num_capas_conxeladas, lr_inicial, lr_degrad, batch_tam]
        resultados.loc[k] = hyper_param_arr + train_perf + val_perf + [kappa]
 
        with open(resultados_ficheiro, 'w') as f:
            resultados.to_csv(f)

        gardar_progresion(progresion, os.path.join(".", dir_progresion, str(k) + ".csv"))


        # QUEDARSE COS TRES MELLORES MODELOS 
        if (start > 0):
            val_acc = val_perf[1]
            if kappa > ultima_metrica:
            #if val_acc > ultima_metrica:
                modelo.save(os.path.join(".", dir_modelo, str(k)))
                shutil.rmtree(os.path.join(".",dir_modelo, str(ultimo_lugar)))
        else:
            modelo.save(os.path.join(".", dir_modelo, str(k)) )
