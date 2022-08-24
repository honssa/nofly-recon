import os
import sys
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository import Gio, GdkPixbuf, GLib
from PIL import Image, ImageOps

#from IPython.display import Image, display
from tensorflow.keras.models import load_model
from segmentation_models import get_preprocessing
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
import math
import numpy as np
import cv2 as cv
import shutil
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator


np.set_printoptions(threshold=sys.maxsize)
class Vista():
    def __init__(self):
        self.botn_imx = Gtk.Button(label="...")
        self.botn_dir = Gtk.Button(label="...")
        self.botn_seg = Gtk.Button(label="->")
        self.botn_ant = Gtk.Button(label="<-")
        self.botn_ira = Gtk.Button(label="Ir")
        self.entrad_num = Gtk.Entry()

        etiq1 = Gtk.Label("Abrir modelo:")
        self.marcador_imx = Gtk.Label("")
        self.botn1 = Gtk.Button(label="...")
        self.etiq_modelo = Gtk.Label("")
        etiq2 = Gtk.Label("Escoller entrada de datos:")
        combo1 = Gtk.ListStore(int, str)
        combo1.append([1, "Imaxe"])
        combo1.append([2, "Coordenadas"])
        combo1.append([3, "Directorio"])
        combo = Gtk.ComboBox.new_with_model_and_entry(combo1)
        combo.connect("changed", self.on_name_combo_changed)
        combo.set_entry_text_column(1)
        self.btnAC = Gtk.Button(label="Ver resultados")
        
        self.box_fila = Gtk.Box(spacing=6)
        self.box_fila.pack_start(etiq1, False, True, 0)
        self.box_fila.pack_start(self.botn1, False, True, 0)
        self.box_fila.pack_start(self.etiq_modelo, False, True, 0)

        box_fila2 = Gtk.Box(spacing=6)
        box_fila2.pack_start(etiq2, False, True, 0)
        box_fila2.pack_start(combo, False, True, 0)

        self.box_buttons = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0, margin=0)
        self.box_buttons.pack_start(self.box_fila, False, True, 0)
        self.box_buttons.pack_start(box_fila2, False, True, 0)
        self.box_buttons.pack_end(self.btnAC, False, True, 0)
        self.box_buttons.set_hexpand(False); self.box_buttons.set_vexpand(False);

        # Espazo para a Imaxe:
        self.imx = Gtk.Image()
        #self.imx.set_from_file("./probar.png")

        self.box_ult = Gtk.Box(spacing=6)
        self.evBox = Gtk.EventBox()
        self.evBox.add(self.imx)
        grid = Gtk.Grid(margin=0, column_spacing=0, row_spacing=0)
        grid.set_hexpand(False); grid.set_vexpand(False)
        grid.attach(self.box_buttons, 0, 0, 1, 1)
        grid.attach(self.evBox, 0, 1, 1, 3)
        grid.attach(self.box_ult, 0, 4, 1, 1)
        self._win = Gtk.Window(title="Aplicacion")
        self._win.connect("delete-event", Gtk.main_quit)
        self._win.add(grid)

        self._win.set_default_size(1000, 600)
        self.show_win() 
    def on_name_combo_changed(self, combo):
        if (combo.get_child().get_text() == "Imaxe"):
            box_c_imx = Gtk.Box(spacing=6)
            etiq_imx = Gtk.Label("Abrir Imaxe:")
            self.etiq_imx_dir = Gtk.Label("")
            box_c_imx.pack_start(etiq_imx, False, True, 0)
            box_c_imx.pack_start(self.botn_imx, False, True, 0)
            box_c_imx.pack_start(self.etiq_imx_dir, False, True, 0)
            self.box_buttons.pack_start(box_c_imx, False, True, 0)
        elif (combo.get_child().get_text() == "Directorio"):
            box_c_dir = Gtk.Box(spacing=6)
            etiq_imx = Gtk.Label("Abrir Directorio:")
            self.etiq_dir_dir = Gtk.Label("")
            box_c_dir.pack_start(etiq_imx, False, True, 0)
            box_c_dir.pack_start(self.botn_dir, False, True, 0)
            box_c_dir.pack_start(self.etiq_dir_dir, False, True, 0)
            self.box_buttons.pack_start(box_c_dir, False, True, 0)
        self.show_win()



    def show_win(self):
        self._win.show_all()
        #self.xan = XanelaPrcpal(self)

    def cargarModelo(self, par, c):
        fexpl = XanExplFicheiros(0, c)
        fexpl.connect("destroy", Gtk.main_quit)

    def cargarImaxe(self, par, c):
        fexpl = XanExplFicheiros(1, c)
        fexpl.connect("destroy", Gtk.main_quit)
    
    def cargarDirectorio(self, par, c):
        fexpl = XanExplFicheiros(2, c, True)
        fexpl.connect("destroy", Gtk.main_quit)


    def set_modelo_dir(self, modelo_dir):
        self.etiq_modelo.set_label(modelo_dir)
        self.box_fila.pack_start(self.etiq_modelo, False, True, 0)
        self.show_win()
    
    def set_imaxe_dir(self, imaxe_dir):
        self.etiq_imx_dir.set_label(imaxe_dir)
        self.box_fila.pack_start(self.etiq_imx_dir, False, True, 0)
        self.show_win()
    
    def set_directorio_dir(self, directorio_dir):
        self.etiq_dir_dir.set_label(directorio_dir)
        self.box_fila.pack_start(self.etiq_dir_dir, False, True, 0)
    
    def connect(self, c):
        self.botn1.connect("clicked", self.cargarModelo, c)
        self.botn_imx.connect("clicked", self.cargarImaxe, c)
        self.botn_dir.connect("clicked", self.cargarDirectorio, c)
        self.btnAC.connect("clicked", c.cargar_resultado)
        self.botn_seg.connect("clicked", c.cargar_resultado, "+1")
        self.botn_ant.connect("clicked", c.cargar_resultado, "-1")
        self.botn_ira.connect("clicked", c.cargar_resultado, "%")


    def actualizarImx(self, imx):
        self.imx.set_from_pixbuf(imx)
        self._win.show_all()
         #self.imx.set_from_file("./probar.png")

    def actualizarImxDir(self, imxLista, indice, tam):
        self.imx.set_from_pixbuf(imxLista)


        self.box_ult.pack_start(self.botn_ant, False, True, 0)
        self.marcador_imx.set_label(str(indice)+"/"+str(tam))
        self.box_ult.pack_start(self.marcador_imx, False, True, 0)
        self.box_ult.pack_start(self.botn_seg, False, True, 0)
        self.box_ult.pack_start(self.entrad_num, False, True, 0)
        self.box_ult.pack_start(self.botn_ira, False, True, 0)
        self._win.show_all()
        


class Controlador():
    def __init__(self, vista, modelo):
        self._vista = vista
        self._modelo = modelo
        self._vista.connect(self)
        self.modelo_dir = ""
        self.imaxe_dir = ""
        self.directorio_dir = ""
        self.coordenadas = None
    
    def set_modelo(self, modelo_dir):
        self.modelo_dir = modelo_dir
        self._vista.set_modelo_dir(modelo_dir)

    def set_imaxe(self, imaxe_dir):
        self.imaxe_dir = imaxe_dir
        self._vista.set_imaxe_dir(imaxe_dir)
    
    def set_directorio(self, dire_dir):
        self.directorio_dir = dire_dir
        self._vista.set_directorio_dir(dire_dir)


    def cargar_resultado(self, cousa, despr=""):
        if despr == "%":
            despr = self._vista.entrad_num.get_text()
        if self.modelo_dir == "":
            print("Non hai modelo, error")
            exit()
        elif (self.imaxe_dir == "") and (not self.coordenadas) and (self.directorio_dir == ""):
            print("Non hai nin imaxe nin coordenadas nin directorio, error")
            exit()
        else:
            if (not self.coordenadas) and (self.directorio_dir == ""):
                imx = self._modelo.componher_imx(self.modelo_dir, self.imaxe_dir)
                self._vista.actualizarImx(imx)
            elif (self.coordenadas) and (self.directorio_dir == "") and (self.imaxe_dir == ""):
                self._modelo.predict_coord(self.modelo_dir, self.coordenadas)
            elif (not self.coordenadas) and (self.imaxe_dir == ""):
                idx,tam,imxLista = self._modelo.predict_dir(self.modelo_dir, self.directorio_dir, despr)
                #self._vista.actualizarImxDir(imxLista,idx,tam)

    def cargar_directorio(self, cousa):
        pass

class Modelo():
    def __init__(self):
        self.idx_imx = 0
        self.validation_datagen = ImageDataGenerator(rescale=1./255)

    def componher_imx_pix(self, modelo_dir, imaxe_dir):
        import matplotlib
        from matplotlib import pyplot as plt
        modelo = load_model(modelo_dir, compile=False)
        import Config
        class_labels=["Edificio","Estrada","Libre"]
        
        # Descompoñer a imaxea
        imxAltura = imxAnchura = 1000; altura = anchura = Config.IMX_ALTO
        if os.path.exists("./tmp"):
            shutil.rmtree("./tmp/")
        os.mkdir("./tmp"); os.mkdir("./tmp/st"); k=0
        imx = Image.open(imaxe_dir)
        for i in range(0, imxAltura, 10):
            for j in range(0, imxAnchura, 10):
                caixa = (j, i, j + anchura, i + altura)
                a = imx.crop(caixa)
                nome_tmp = './tmp/{num:0{width}}.png'.format(num=k, width=6)
                a.save(nome_tmp)
                k += 1
        print("separacion realizada") 
        self.validation_generator = keras.preprocessing.image.DirectoryIterator(
                "./tmp",
                self.validation_datagen,
                target_size = (Config.IMX_ALTO,Config.IMX_ANCHO),
                color_mode = "rgb",
                class_mode = "categorical",
                batch_size = Config.BATCH_TAM,
                shuffle = False
            )

        imx_dirs = sorted([os.path.join("./tmp", image_id) for image_id in os.listdir("./tmp")])
        predecidas = []
        voltas = ((1000//10) * (1000//10)) // Config.BATCH_TAM
        for i in range(voltas):
            #test_img, test_lbl = self.validation_generator.__next__()
            batch = np.array([cv.imread(imx_dirs[i*Config.BATCH_TAM])])
            for j in range(1,Config.BATCH_TAM):
                batch = np.concatenate((batch, np.array([ cv.imread(imx_dirs[i*Config.BATCH_TAM+j]) ]) ), axis=0)
            prediccions = modelo.predict(batch)
            prediccions = np.argmax(prediccions, axis=1)
            for bidx in range(Config.BATCH_TAM):
                pred_labl = class_labels[prediccions[bidx]]
                img = np.ones((1,1,3), dtype=np.uint8)
                if pred_labl == "Edificio":
                    img = img * (255,0,0)
                elif pred_labl == "Estrada":
                    img = img * (0,0,255)
                elif pred_labl == "Libre":
                    img = img * (0,255,0)
                
                pred_imx = Image.fromarray(img.astype(np.uint8))
                pred_imx.save("./tmp/st/"+"{num:0{width}}.png".format(num=(i*Config.BATCH_TAM+bidx),width=6))
                predecidas.append(img)
        
        print("remontar") 
        arr_filas = []; idx = 0
        imx_arr = np.array([]);
        #imx_arr = predecidas[idx]
        xc = 1000 // 10 
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

        imx_arr = imx_arr.astype(np.uint8)
        #print("Forma final: " + str(imx_arr))
        imx_composta = Image.fromarray(imx_arr)

        #plt.imshow(imx_composta)
        #plt.title("Imaxe")
        #plt.show()
        return imx_composta
        

    def componher_imx(self, modelo_dir, imaxe_dir):
        import matplotlib
        from matplotlib import pyplot as plt
        # Imaxe de 10000x10000
        modelo = load_model(modelo_dir, compile=False)

        import Config
        class_labels=['Edificio','Estrada','Libre']
        # Descompoñer a imaxea
        imxAltura = imxAnchura = 10000; altura = anchura = Config.IMX_ALTO
        if os.path.exists("./tmp"):
            shutil.rmtree("./tmp/")
        os.mkdir("./tmp"); os.mkdir("./tmp/st"); k=0
        imx = Image.open(imaxe_dir)
        for i in range(0, imxAltura, altura):
            for j in range(0, imxAnchura, anchura):
                caixa = (j, i, j + anchura, i + altura)
                a = imx.crop(caixa)
                nome_tmp = './tmp/{num:0{width}}.png'.format(num=k, width=5)
                a.save(nome_tmp)
                k += 1
       
        self.validation_generator = keras.preprocessing.image.DirectoryIterator(
                "./tmp",
                self.validation_datagen,
                target_size = (Config.IMX_ALTO,Config.IMX_ANCHO),
                color_mode = "rgb",
                class_mode = "categorical",
                batch_size = Config.BATCH_TAM,
                shuffle = False
            )

        imx_dirs = sorted([os.path.join("./tmp", image_id) for image_id in os.listdir("./tmp")])
        predecidas = []
        voltas = ((10000//Config.IMX_ALTO) * (10000//Config.IMX_ANCHO)) // Config.BATCH_TAM
        for i in range(voltas):
            #test_img, test_lbl = self.validation_generator.__next__()
            batch = np.array([cv.imread(imx_dirs[i*Config.BATCH_TAM])])
            for j in range(1,Config.BATCH_TAM):
                batch = np.concatenate((batch, np.array([ cv.imread(imx_dirs[i*Config.BATCH_TAM+j]) ]) ), axis=0)
            prediccions = modelo.predict(batch)
            prediccions = np.argmax(prediccions, axis=1)
            for bidx in range(Config.BATCH_TAM):
                pred_labl = class_labels[prediccions[bidx]]
                img = np.ones((4,4,3), dtype=np.uint8)
                if pred_labl == "Edificio":
                    img = img * (255,0,0)
                elif pred_labl == "Estrada":
                    img = img * (0,0,255)
                elif pred_labl == "Libre":
                    img = img * (0,255,0)
                
                pred_imx = Image.fromarray(img.astype(np.uint8))
                pred_imx.save("./tmp/st/"+"{num:0{width}}.png".format(num=(i*Config.BATCH_TAM+bidx),width=5))
                predecidas.append(img)
        
        arr_filas = []; idx = 0
        imx_arr = np.array([]);
        #imx_arr = predecidas[idx]
        
        xc = 10000 // Config.IMX_ALTO 
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

        imx_arr = imx_arr.astype(np.uint8)
        #print("Forma final: " + str(imx_arr))
        imx_composta = Image.fromarray(imx_arr)

        #plt.imshow(imx_composta)
        #plt.title("Imaxe")
        #plt.show()
        return imx_composta

        

    def predict_imx(self, modelo_dir, imaxe_dir):
        # Cargar modelo (asumimos que todos son resnet50)
       
        # Direccion da imaxe 
        #categoria = ["Edificios", "Estradas", "Libre"][random.randint(0,2)]
        #imaxe_dir += categoria + "/" + str(i)

        self.validation_generator = self.validation_datagen.flow_from_directory(
                            imaxe_dir,
                            color_mode="rgb",
                            target_size=(125, 125),
                            batch_size=16,
                            class_mode="categorical",
                            shuffle=True)
        # Prediccion
        test_img, test_lbl = self.validation_generator.__next__()
        modelo = load_model(modelo_dir, compile=False)
        predictions=modelo.predict(test_img)

        predictions = np.argmax(predictions, axis=1)
        test_labels = np.argmax(test_lbl, axis=1)

        n=random.randint(0, test_img.shape[0] - 1)
        image = test_img[n] * 255
        class_labels=['Edificio','Estrada','Libre']
        orig_labl = class_labels[test_labels[n]]
        pred_labl = class_labels[predictions[n]]

       # import matplotlib
       # from matplotlib import pyplot as plt
       # plt.imshow(image)
       # plt.title("Original label is:")
       # plt.show()
        print(image)

        # Xuntalo todo
        img = np.zeros((125,125,3), np.uint8)
        if pred_labl == "Edificio":
            img[:,:] = (255,0,0)
        if pred_labl == "Estrada":
            img[:,:] = (0,255,0)
        if pred_labl == "Libre":
            img[:,:] = (0,0,255)
        stac = np.hstack((image,img))


        stac = Image.fromarray(np.uint8(stac))
        glibbytes = GLib.Bytes.new( stac.tobytes() )
        gdkpixbuf = GdkPixbuf.Pixbuf.new_from_data( glibbytes.get_data(), GdkPixbuf.Colorspace.RGB, False, 8, \
                stac.width, stac.height, len( stac.getbands() )*stac.width, None, None ) # Line B

        return gdkpixbuf
    def predict_coord(self, modelo_dir, coord):
        pass
    
    def predict_dir(self, modelo_dir, directorio_dir, despr):
        #cnnModelo = keras.models.load_model(modelo_dir)
        #directorio_dir += "/Imaxes"
        #test_input_dirs = sorted([os.path.join(directorio_dir, fnome) \
        #    for fnome in os.listdir(directorio_dir)])
        k = 1
        imaxes = os.listdir(directorio_dir)
        for imx in imaxes:
            print(os.path.join(directorio_dir,imx))
            imx_seg = self.componher_imx(modelo_dir, os.path.join(directorio_dir,imx))
            imx_seg.save(directorio_dir+"/s"+imx.split(".jpg")[0]+".png")
            k+=1

    
    def separar_cores(self, imx_arr):
        w, h = imx_arr.shape
        for i in range(w):
            for j in range(h):
                if imx_arr[i][j] == 1: # Urbanas
                    imx_arr[i][j] = 255
                else:
                    imx_arr[i][j] = 0 # Libre

    def separar_cores_seg(self, imx_arr):
        w, h = imx_arr.shape
        for i in range(w):
            for j in range(h):
                if imx_arr[i][j] == 1: # Urbanas
                    imx_arr[i][j] = 255
                else:
                    imx_arr[i][j] = 0 # Libre




class XanExplFicheiros(Gtk.Window):
    def __init__(self, imx, c, folder=False):
        super().__init__(title="Explorador de Arquivos")
        
        if folder:
            dialog = Gtk.FileChooserDialog( 
                    title="Escolla o directorio", 
                    parent=self, 
                    action = Gtk.FileChooserAction.SELECT_FOLDER,
                )
        else:
            dialog = Gtk.FileChooserDialog(
                title="Escolla o ficheiro", parent=self, action=Gtk.FileChooserAction.OPEN
            )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL, 
            Gtk.STOCK_OPEN,
            Gtk.ResponseType.OK,
        )
        
        if imx == 1:
            self.add_filters2(dialog)
        elif imx == 0:
            self.add_filters(dialog)
        elif imx == 2:
            self.add_filters3(dialog)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            if imx == 1:
                c.set_imaxe(dialog.get_filename())
            elif imx == 0:
                c.set_modelo(dialog.get_filename())
            elif imx == 2:
                c.set_directorio(dialog.get_filename())
            print("File selected: " + dialog.get_filename())
        elif response == Gtk.ResponseType.CANCEL:
            print("Cancel clicked")

        dialog.destroy()

    def add_filters(self, dialog):
        filtro_modelo = Gtk.FileFilter()
        filtro_modelo.set_name("ficheiro Modelo")
        filtro_modelo.add_pattern("*.h5")
        dialog.add_filter(filtro_modelo)

    def add_filters2(self, dialog):
        filtro_imaxe = Gtk.FileFilter()
        filtro_imaxe.set_name("ficheiro Imaxe")
        filtro_imaxe.add_pattern("*")
        dialog.add_filter(filtro_imaxe)
    
    def add_filters3(self, dialog):
        filtro_imaxe = Gtk.FileFilter()
        filtro_imaxe.set_name("directorio")



        



if __name__ == "__main__":

    keras.backend.clear_session()
    vista = Vista()
    modelo = Modelo()
    controlador = Controlador(vista, modelo)
    
    Gtk.main()
    #modelo = keras.models.load_model(sys.argv[1])
    #modelo.predict()
