import tensorflow as tf
from pyproj import Transformer
import math
import cv2
import numpy as np
from PIL import Image
import os
import shutil
from modeloFactoria import ModeloFactoria

class ModeloApp():
    def __init__(self):
        self.xlimite = (330000,340000)
        self.ylimite = (5698000,5708000)
        self.modelo = ModeloFactoria()
        self.validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        #modeloUnetEd_dir = "./seg_edificios_minLOSS.h5"
        #modeloUnetEst_dir = "./seg_estradas_minLOSS.h5"
        #modeloResNet_dir = "./mc_tam50_3It.h5"
        #modeloConv_dir = "./mc_tam50_3It.h5"
        
        #self.modeloConv = None#tf.keras.models.load_model(modelo_dir,compile=False)
        #self.modeloUnetEd = None#tf.keras.models.load_model(modelo_edificios_dir,compile=False)
        #self.modeloUnetEst = None#tf.keras.models.load_model(modelo_estradas_dir,compile=False)
        #self.modeloResNet = None

    def wgs84_a_EPSG25832(self, x, y):
        transformer = Transformer.from_crs("wgs 84", "epsg:25832")
        x2, y2 = transformer.transform(x, y)
        x2 = math.floor(x2); y2 = math.floor(y2)
        return (x2, y2)
    
    def getImx(self, lat, lon):
        x,y = self.wgs84_a_EPSG25832(lat,lon)
        # Pillas o mosaico
        mosx = x - x%1000; mosx+=500
        mosy = y - y%1000; mosy+=500

        offset_x = x - mosx
        offset_y = y - mosy
        
        imx_res = None
        if offset_x > 0 and offset_y > 0:
            # Recortar o apropiado de cada imaxe veciña
            adx_ht = f"x_{mosx+1000}_y_{mosy}.png"
            adx_ht = cv2.imread("./fotos_descargadas/"+adx_ht) 
            adx_ht = adx_ht[0:10000-offset_y*10, 0:offset_x*10]
           
            adx_vt = f"x_{mosx}_y_{mosy+1000}.png"
            adx_vt = cv2.imread("./fotos_descargadas/"+adx_vt) 
            adx_vt = adx_vt[10000-offset_y*10:10000, offset_x*10:10000]
           
            adx_diag = f"x_{mosx+1000}_y_{mosy+1000}.png"
            adx_diag = cv2.imread("./fotos_descargadas/"+adx_diag)
            adx_diag = adx_diag[10000-offset_y*10:10000, 0:offset_x*10]
            
            ctr = f"x_{mosx}_y_{mosy}.png"
            ctr = cv2.imread("./fotos_descargadas/"+ctr)
            ctr = ctr[0:10000-offset_y*10, offset_x*10:10000]
            tmp1 = np.hstack((ctr, adx_ht))
            tmp2 = np.hstack((adx_vt, adx_diag))
            imx_res = np.vstack((tmp2, tmp1))

        elif offset_x > 0 and offset_y < 0:

            adx_ht = f"x_{mosx+1000}_y_{mosy}.png"
            adx_ht = cv2.imread("./fotos_descargadas/"+adx_ht) 
            adx_ht = adx_ht[abs(offset_y)*10:10000, 0:offset_x*10]
           
            adx_vt = f"x_{mosx}_y_{mosy-1000}.png"
            adx_vt = cv2.imread("./fotos_descargadas/"+adx_vt) 
            adx_vt = adx_vt[0:abs(offset_y)*10, offset_x*10:10000]
           
            adx_diag = f"x_{mosx+1000}_y_{mosy-1000}.png"
            adx_diag = cv2.imread("./fotos_descargadas/"+adx_diag)
            adx_diag = adx_diag[0:abs(offset_y)*10, 0:offset_x*10]
            
            ctr = f"x_{mosx}_y_{mosy}.png"
            ctr = cv2.imread("./fotos_descargadas/"+ctr)
            ctr = ctr[abs(offset_y)*10:10000, offset_x*10:10000]
            tmp1 = np.hstack((ctr, adx_ht))
            tmp2 = np.hstack((adx_vt, adx_diag))
            imx_res = np.vstack((tmp1, tmp2))
            
        elif offset_x < 0 and offset_y > 0:

            adx_ht = f"x_{mosx-1000}_y_{mosy}.png"
            adx_ht = cv2.imread("./fotos_descargadas/"+adx_ht) 
            adx_ht = adx_ht[0:10000-offset_y*10, 10000-abs(offset_x)*10:10000]
           
            adx_vt = f"x_{mosx}_y_{mosy+1000}.png"
            adx_vt = cv2.imread("./fotos_descargadas/"+adx_vt) 
            adx_vt = adx_vt[10000-offset_y*10:10000, 0:10000-abs(offset_x)*10]
           
            adx_diag = f"x_{mosx-1000}_y_{mosy+1000}.png"
            adx_diag = cv2.imread("./fotos_descargadas/"+adx_diag)
            adx_diag = adx_diag[10000-offset_y*10:10000, 10000-abs(offset_x)*10:10000]
            
            ctr = f"x_{mosx}_y_{mosy}.png"
            ctr = cv2.imread("./fotos_descargadas/"+ctr)
            ctr = ctr[0:10000-offset_y*10, 0:10000-abs(offset_x)*10]
            tmp1 = np.hstack((adx_ht, ctr))
            tmp2 = np.hstack((adx_diag, adx_vt))
            imx_res = np.vstack((tmp2, tmp1))

        elif offset_x < 0 and offset_y < 0:
            
            adx_ht = f"x_{mosx-1000}_y_{mosy}.png"
            adx_ht = cv2.imread("./fotos_descargadas/"+adx_ht) 
            adx_ht = adx_ht[abs(offset_y)*10:10000, 10000-abs(offset_x)*10:10000]
           
            adx_vt = f"x_{mosx}_y_{mosy-1000}.png"
            adx_vt = cv2.imread("./fotos_descargadas/"+adx_vt) 
            adx_vt = adx_vt[0:abs(offset_y)*10, 0:10000-abs(offset_x)*10]
           
            adx_diag = f"x_{mosx-1000}_y_{mosy-1000}.png"
            adx_diag = cv2.imread("./fotos_descargadas/"+adx_diag)
            adx_diag = adx_diag[0:abs(offset_y)*10, 10000-abs(offset_x)*10:10000]
            
            ctr = f"x_{mosx}_y_{mosy}.png"
            ctr = cv2.imread("./fotos_descargadas/"+ctr)
            ctr = ctr[abs(offset_y)*10:10000, 0:10000-abs(offset_x)*10]
            tmp1 = np.hstack((adx_ht, ctr))
            tmp2 = np.hstack((adx_diag, adx_vt))
            imx_res = np.vstack((tmp1, tmp2))
     
        return imx_res


    def alphaMerge(self, small_foreground, background, top, left):
        """
        Puts a small BGRA picture in front of a larger BGR background.
        :param small_foreground: The overlay image. Must have 4 channels.
        :param background: The background. Must have 3 channels.
        :param top: Y position where to put the overlay.
        :param left: X position where to put the overlay.
        :return: a copy of the background with the overlay added.
        """
        result = background.copy()
        # From everything I read so far, it seems we need the alpha channel separately
        # so let's split the overlay image into its individual channels
        fg_b, fg_g, fg_r, fg_a = cv2.split(small_foreground)
        # Make the range 0...1 instead of 0...255
        fg_a = fg_a / 255.0
        # Multiply the RGB channels with the alpha channel
        label_rgb = cv2.merge([fg_b * fg_a, fg_g * fg_a, fg_r * fg_a])

        # Work on a part of the background only
        height, width = small_foreground.shape[0], small_foreground.shape[1]
        part_of_bg = result[top:top + height, left:left + width, :]
        # Same procedure as before: split the individual channels
        bg_b, bg_g, bg_r = cv2.split(part_of_bg)
        # Merge them back with opposite of the alpha channel
        part_of_bg = cv2.merge([bg_b * (1 - fg_a), bg_g * (1 - fg_a), bg_r * (1 - fg_a)])

        # Add the label and the part of the background
        cv2.add(label_rgb, part_of_bg, part_of_bg)
        # Replace a part of the background
        result[top:top + height, left:left + width, :] = part_of_bg
        return result
    
    def separar_cores_comb(self, imx_edi, imx_est):
        w,h = imx_edi.shape
        nova_imx = np.zeros((w, h, 4), np.uint8)
        # BGR
        for i in range(w):
            for j in range(h):
                if (imx_edi[i][j] == 1) and (imx_est[i][j] == 0):
                    nova_imx[i][j] = (255,0,0,255)
                elif (imx_edi[i][j] == 0) and (imx_est[i][j] == 1):
                    nova_imx[i][j] = (0,0,255,255)
                elif (imx_edi[i][j] == 0) and (imx_est[i][j] == 0):
                    nova_imx[i][j] = (0,255,0,0)
                elif (imx_edi[i][j] == 1) and (imx_est[i][j] == 1):
                    nova_imx[i][j] = (102,0,102,255)

        return nova_imx
    
    def predecir(self, imx, id_modelo):
        modelo = self.modelo.getModelo(id_modelo)
        if id_modelo == 0 or id_modelo == 1: # Convolucional 0, ResNet 1
            orixinal = imx
            class_labels=['Edificio','Estrada','Libre']
            # Descompoñer a imaxe
            imxAltura = imxAnchura = 10000; altura = anchura = 50
            if os.path.exists("./tmp"):
                shutil.rmtree("./tmp/")
            os.mkdir("./tmp"); k=0
            for i in range(0, imxAltura, altura):
                for j in range(0, imxAnchura, anchura):
                    #caixa = (j, i, j + anchura, i + altura)
                    tmp = imx[i:i+altura, j:j+anchura]
                    #a = imx.crop(caixa)
                    nome_tmp = './tmp/{num:0{width}}.png'.format(num=k, width=5)
                    cv2.imwrite(nome_tmp, tmp)
                    #a.save(nome_tmp)
                    k += 1
       
            self.validation_generator = tf.keras.preprocessing.image.DirectoryIterator(
                    "./tmp",
                    self.validation_datagen,
                    target_size = (50,50),
                    color_mode = "rgb",
                    class_mode = "categorical",
                    batch_size = 16,
                    shuffle = False
                )
            
            imx_dirs = sorted([os.path.join("./tmp", image_id) for image_id in os.listdir("./tmp")])
            predecidas = []
            voltas = ((10000//50) * (10000//50)) // 16
            for i in range(voltas):
                batch = np.array([cv2.imread(imx_dirs[i*16])])
                for j in range(1,16):
                    batch = np.concatenate((batch, np.array([ cv2.imread(imx_dirs[i*16+j]) ]) ), axis=0)
                prediccions = modelo.predecir(batch)
                #if id_modelo == 0: #Conv
                #    prediccions = self.modeloConv.predict(batch)
                #elif id_modelo == 1: # ResNet
                #    prediccions = self.modeloResNet.predict(batch)

                #prediccions = self.modelo.predict(batch)
                prediccions = np.argmax(prediccions, axis=1)
                for bidx in range(16):
                    pred_labl = class_labels[prediccions[bidx]]
                    img = np.ones((4,4,4), dtype=np.uint8)
                    if pred_labl == "Edificio":
                        img = img * (0,0,255,255)
                    elif pred_labl == "Estrada":
                        img = img * (255,0,0,255)
                    elif pred_labl == "Libre":
                        img = img * (0,255,0,0)
                    
                    predecidas.append(img)
            
            arr_filas = []; idx = 0
            imx_arr = np.array([]);
            
            xc = 10000 // 50
            for fila in range(xc):
                for columna in range(xc):
                    if not imx_arr.size:
                        imx_arr = predecidas[idx]
                    else:
                        imx_arr = np.hstack((imx_arr, predecidas[idx]))
                    idx += 1

                arr_filas.append(imx_arr)
                imx_arr = np.array([])

            imx_arr = np.array([])
            for fila in arr_filas:
                if not imx_arr.size:
                    imx_arr = fila
                else:
                    imx_arr = np.vstack((imx_arr,fila))

            imx_arr = np.uint8(imx_arr)
            orixinal = cv2.resize(orixinal, (800, 800),interpolation = cv2.INTER_NEAREST)
            imx_arr = self.alphaMerge(imx_arr, orixinal, 0, 0)
            imx_arr = cv2.cvtColor(imx_arr, cv2.COLOR_RGB2BGR)
            return imx_arr

        elif id_modelo == 2: # Unet 
            orixinal = imx
            imx_arr = []; tam = 2000 
            if os.path.exists("./tmp"):
                shutil.rmtree("./tmp/")
            os.mkdir("./tmp")
            contador = 0
            for j in range(0,10000,tam):
                for i in range(0, 10000, tam):
                    imx_ = imx[j:j+tam, i:i+tam]
                    imx_ = cv2.resize(imx_,(512,512), interpolation=cv2.INTER_NEAREST)
                    nome = '{num:0{width}}'.format(num=contador, width=2)
                    cv2.imwrite(f"./tmp/{nome}.png", imx_)
                    contador += 1
            
            predecidas = []
            for i, foto in enumerate(sorted(os.listdir("./tmp"))):
                x = np.zeros((1,) + (512,512) + (3,), dtype="float32")
                imx = tf.keras.preprocessing.image.load_img(os.path.join("./tmp",foto), target_size=(512,512))
                x[0] = imx
                predicion_edificios, predicion_estradas = modelo.predecir(x)
                #predicion_estradas = self.modeloUnetEst.predict(x, batch_size=1)
                #predicion_edificios = self.modeloUnetEd.predict(x, batch_size=1)
                predicion_estradas = np.argmax(predicion_estradas, axis=3)[0,:,:]
                predicion_edificios = np.argmax(predicion_edificios, axis=3)[0,:,:]
                predicion = self.separar_cores_comb(predicion_edificios, predicion_estradas)
                predecidas.append(predicion)
                cv2.imwrite(f"./tmp/seg-{i}.png", predicion)
            
            arr_filas = []
            imx_arr = np.array([]); idx=0
            xc = 2560 // 512 
            for fila in range(xc):
                for columna in range(xc):
                    if not imx_arr.size:
                        imx_arr = predecidas[idx]
                    else:
                        imx_arr = np.hstack((imx_arr, predecidas[idx]))
                    idx += 1

                arr_filas.append(imx_arr)
                imx_arr = np.array([])

            imx_arr = np.array([])
            for fila in arr_filas:
                if not imx_arr.size:
                    imx_arr = fila
                else:
                    imx_arr = np.vstack((imx_arr,fila))
            
            orixinal = cv2.resize(orixinal, (2560, 2560),interpolation = cv2.INTER_NEAREST)
            imx_arr = self.alphaMerge(imx_arr, orixinal, 0, 0)
            imx_arr = cv2.cvtColor(imx_arr, cv2.COLOR_RGB2BGR)
            
            return imx_arr.astype(np.uint8)

        #return imx_arr




#c = Coord2Imx()

#51.447552169022686, 6.607968338512571   (INF, DEREITA)
#51.45194217257074, 6.607515772899683    (SUP, DEREITA)
#51.44690254025112, 6.602836954918035    (INF, ESQUERDA)
#51.45192609614042, 6.601053285890925    (SUP, ESQUERDA)
#c.getImx(51.44690254025112, 6.60283695491803)

