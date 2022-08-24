import tensorflow as tf
import Config
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import os, cv2
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
class UAVzr(tf.keras.utils.Sequence):
    def __init__(self, batch_tam, imx_dim, ad_input_dirs, ad_mascara_dirs):
        self.batch_tam = batch_tam
        self.imx_dim = imx_dim
        self.ad_input_dirs = ad_input_dirs
        self.ad_mascara_dirs = ad_mascara_dirs

    def __len__(self):
        return len(self.ad_mascara_dirs) // self.batch_tam

    def __getitem__(self, idx):
        # Devolve a tupla (input, mascara) correspondente a o numero de batch
        i = idx * self.batch_tam
        batch_input_imx_dirs = self.ad_input_dirs[i : i + self.batch_tam]
        batch_mascara_imx_dirs = self.ad_mascara_dirs[i : i + self.batch_tam]
        # Lista que conten todas as imaxes [Ancho x Alto x Cor]
        x = np.zeros((self.batch_tam,) + self.imx_dim + (3,), dtype="float32")
        for j, path in enumerate(batch_input_imx_dirs):
            imx = tf.keras.preprocessing.image.load_img(path, target_size=self.imx_dim)
            x[j] = imx
        y = np.zeros((self.batch_tam,) + self.imx_dim + (1,), dtype="uint8")
        for j, path in enumerate(batch_mascara_imx_dirs):
            imx = tf.keras.preprocessing.image.load_img(path, target_size=self.imx_dim, color_mode="grayscale")
            y[j] = np.expand_dims(imx, 2)
        y = tf.keras.utils.to_categorical(y, num_classes=len(Config.NUM_CLASES))
        return x, y


def cargar_datos(directorio):
    input_dir = os.path.join(directorio, "imx")
    gt_dir = os.path.join(directorio, "gt")
    ficheiros = os.listdir(input_dir)
    random.shuffle(ficheiros)
    ad_input_dirs = []; ad_mascara_dirs = []
    for fich in ficheiros:
        ad_input_dirs.append(os.path.join(input_dir, fich))
        ad_mascara_dirs.append(os.path.join(gt_dir, fich))

    test_input_dirs = ad_input_dirs[0:int(Config.TEST_SPLIT*len(ad_input_dirs))]
    ad_input_dirs = ad_input_dirs[len(test_input_dirs):len(ad_input_dirs)]
    val_input_dirs = ad_input_dirs[0:int(Config.VALIDACION_SPLIT*len(ad_input_dirs))]
    ad_input_dirs = ad_input_dirs[len(val_input_dirs):len(ad_input_dirs)]
    
    test_mascara_dirs = ad_mascara_dirs[0:int(Config.TEST_SPLIT*len(ad_mascara_dirs))]
    ad_mascara_dirs = ad_mascara_dirs[len(test_mascara_dirs):len(ad_mascara_dirs)]
    val_mascara_dirs = ad_mascara_dirs[0:int(Config.VALIDACION_SPLIT*len(ad_mascara_dirs))]
    ad_mascara_dirs = ad_mascara_dirs[len(val_mascara_dirs):len(ad_mascara_dirs)]
    
    ad_gen = UAVzr(Config.BATCH_TAM, (Config.IMX_ALTO, Config.IMX_ANCHO), ad_input_dirs, ad_mascara_dirs )
    val_gen = UAVzr(Config.BATCH_TAM, (Config.IMX_ALTO, Config.IMX_ANCHO), val_input_dirs, val_mascara_dirs )
    test_gen = UAVzr(Config.BATCH_TAM, (Config.IMX_ALTO, Config.IMX_ANCHO), test_input_dirs, test_mascara_dirs )
    return ad_gen, val_gen, test_gen




#https://keras.io/examples/vision/oxford_pets_image_segmentation/
def unet_keras():
    img_size = (Config.IMX_ANCHO,Config.IMX_ALTO)
    num_classes = len(Config.NUM_CLASES)
    inputs = tf.keras.layers.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = tf.keras.layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.UpSampling2D(2)(x)

        # Project residual
        residual = tf.keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = tf.keras.layers.Conv2D(filters, 1, padding="same")(residual)
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = tf.keras.layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = tf.keras.models.Model(inputs, outputs)
    return model


def IoU_coeff(y_true, y_pred):
    axes = (0,1) 
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask - intersection
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


if __name__=="__main__":

    ious = np.array([])
    for k in range(Config.NUM_ITERACIONS): 
        callbacks_lista = []
        callbacks_lista.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
            min_delta=Config.MIN_DELTA, patience=Config.PACIENCIA, 
            verbose=1, mode='min'))
        callbacks_lista.append(tf.keras.callbacks.ModelCheckpoint(filepath='./seg_estradas.h5', 
                                 monitor='val_loss',
                                 verbose=1, 
                                 save_best_only=True,
                                 mode='min'))

        ad, val, test = cargar_datos(Config.RUTA_DATASET)
        modelo = unet_keras()
        #modelo.summary()
        metricas = ["accuracy",tf.keras.metrics.MeanIoU(len(Config.NUM_CLASES), name=None, dtype=None)]
        modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=metricas)#tf.losses.SparseCategoricalCrossentropy, metrics=metricas)
        progresion = modelo.fit(ad, epochs=Config.EPOCHS, validation_data=val, callbacks=callbacks_lista)
        
        iou = 0
        for i,elemento in enumerate(test):
            imx, gt = elemento
            predicion = modelo.predict(imx, batch_size=1)
            gt = np.argmax(gt, axis=3)
            predicion = np.argmax(predicion, axis=3)
            tmp = IoU_coeff(gt[0], predicion[0])
            iou += tmp
            tmp = IoU_coeff(gt[1], predicion[1])
            iou += tmp

        iou = iou / (i*Config.BATCH_TAM)
        ious = np.append(ious, iou)
        print("IoU media en test:" + str(iou))
   
        print(ious)
    Iou_media = np.average(ious)
    Iou_dt = np.std(ious)
    print(f"IoU media = {Iou_media}, IoU dt = {Iou_dt}")
