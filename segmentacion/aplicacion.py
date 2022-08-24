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
import tensorflow as tf


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
                imx = self._modelo.predict_imx(self.modelo_dir, self.imaxe_dir)
                self._vista.actualizarImx(imx)
            elif (self.coordenadas) and (self.directorio_dir == "") and (self.imaxe_dir == ""):
                self._modelo.predict_coord(self.modelo_dir, self.coordenadas)
            elif (not self.coordenadas) and (self.imaxe_dir == ""):
                idx,tam,imxLista = self._modelo.predict_dir(self.modelo_dir, self.directorio_dir, despr)
                self._vista.actualizarImxDir(imxLista,idx,tam)

    def cargar_directorio(self, cousa):
        pass

#class UAVzr(tf.keras.utils.Sequence):
#    def __init__(self, batch_tam, imx_dim, ad_input_dirs):
#        self.batch_tam = batch_tam
#        self.imx_dim = imx_dim
#        self.ad_input_dirs = ad_input_dirs
#
#    def __len__(self):
#        return len(self.ad_input_dirs) // self.batch_tam
#
#    def __getitem__(self, idx):
#        # Devolve a tupla (input, mascara) correspondente a o numero de batch
#        i = idx * self.batch_tam
#        batch_input_imx_dirs = self.ad_input_dirs[i : i + self.batch_tam]
#        # Lista que conten todas as imaxes [Ancho x Alto x Cor]
#        x = np.zeros((self.batch_tam,) + self.imx_dim + (3,), dtype="float32")
#        for j, path in enumerate(batch_input_imx_dirs):
#            imx = tf.keras.preprocessing.image.load_img(path, target_size=self.imx_dim)
#            x[j] = imx
#        return x


def mkdir_se_non_existe(direccion):
    try:
        os.mkdir(direccion)
    except FileExistsError:
        pass


class Modelo():
    def __init__(self):
        self.idx_imx = 0
    
    def componher_imx(self, modelo_dir, imaxe_dir):
        # Partir a imaxe en 5x5 seccions cada seccion redimensionase a 515x512 pìxels
        import matplotlib
        from matplotlib import pyplot as plt
        #modelo = load_model(modelo_dir, compile=False)
        modelo_estradas = load_model("/home/wouter/TFG/nofly-recon/segmentacion/programa/seg_estradas_maxIoU.h5") 
        modelo_edificios = load_model("/home/wouter/TFG/nofly-recon/segmentacion/programa/seg_edificios_maxIoU.h5") 
        import Config
        import cv2
        class_labels=["Libre", "Edificio"]
        
        imx = cv2.imread(imaxe_dir)
        imx_arr = []; tam = 2000 
        mkdir_se_non_existe("./tmp")
        for f in os.listdir("./tmp"):
            os.remove(os.path.join("./tmp", f))
        contador = 0
        for j in range(0,10000,tam):
            for i in range(0, 10000, tam):
                imx_ = imx[j:j+tam, i:i+tam]
                imx_ = cv2.resize(imx_,(512,512), interpolation=cv2.INTER_NEAREST)
                nome = '{num:0{width}}'.format(num=contador, width=2)
                print(nome)
                cv2.imwrite(f"./tmp/{nome}.png", imx_)
                contador += 1
        
        predecidas = []
        for i, foto in enumerate(sorted(os.listdir("./tmp"))):
            #imx = cv2.imread(os.path.join("./tmp",foto))
            #imx = np.expand_dims(imx,axis=0)
            x = np.zeros((1,) + (512,512) + (3,), dtype="float32")
            imx = tf.keras.preprocessing.image.load_img(os.path.join("./tmp",foto), target_size=(512,512))
            x[0] = imx
            predicion_estradas = modelo_estradas.predict(x, batch_size=1)
            predicion_edificios = modelo_edificios.predict(x, batch_size=1)
            predicion_estradas = np.argmax(predicion_estradas, axis=3)[0,:,:]
            predicion_edificios = np.argmax(predicion_edificios, axis=3)[0,:,:]
            predicion = self.separar_cores_comb(predicion_edificios, predicion_estradas)
            predecidas.append(predicion)
            cv.imwrite(f"./tmp/seg-{i}.png", predicion)
        
        arr_filas = []
        imx_arr = np.array([]); idx=0
        xc = 2560 // Config.IMX_ALTO 
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
        
        imx_composta = Image.fromarray(imx_arr.astype(np.uint8))
        
        #imx_composta = Image.fromarray(imx_arr)
        #plt.imshow(imx_composta)
        #plt.title("Imaxe")
        #plt.show()
        return imx_composta

    def predict_imx(self, modelo_dir, imaxe_dir):
        # Cargar modelo (asumimos que todos son resnet50)
        modelo = load_model(modelo_dir, compile=False)
        # Preprocesamento
        preprocess_input = get_preprocessing("resnet50")
       
        # Imaxe orixinal
        x_test = preprocess_input( [cv.imread(imaxe_dir)] )

        # Mascara
        dirs = imaxe_dir.split("Imaxes")
        y_test = [cv.imread(dirs[0]+"Mascaras"+dirs[1], 0)]
        #y_test = np.expand_dims(y_test, axis=3); 
        y_test = y_test[0]
        self.separar_cores(y_test)

        # Prediccion
        prediccion = modelo.predict(preprocess_input( np.expand_dims(x_test[0],0) ))
        prediccion = np.argmax(prediccion, axis=3)[0,:,:]
        self.separar_cores_seg(prediccion)

        # Xuntalo todo
        x_test = cv.cvtColor(x_test[0], cv.COLOR_RGB2BGR)
        y_test = cv.cvtColor(y_test, cv.COLOR_GRAY2BGR) 

        #print(stac1.shape)
        print(prediccion)
        prediccion = cv.cvtColor(np.float32(prediccion), cv.COLOR_GRAY2BGR)
        stac1 = np.hstack((x_test, y_test))
        stac2 = np.hstack((stac1,prediccion))
        print(prediccion)
        stac2 = Image.fromarray(np.uint8(stac2))
        glibbytes = GLib.Bytes.new( stac2.tobytes() )
        gdkpixbuf = GdkPixbuf.Pixbuf.new_from_data( glibbytes.get_data(), GdkPixbuf.Colorspace.RGB, False, 8, \
                stac2.width, stac2.height, len( stac2.getbands() )*stac2.width, None, None ) # Line B
        # VELLO CODIGO
        #cnnModelo = keras.models.load_model(modelo_dir)
        #val_gen = UAVImaxes([imaxe_dir])
        #predicts = cnnModelo.predict(val_gen)
        #segImx = predicts[0]  
        #segImx = np.argmax(segImx, axis=-1)
        #segImx = np.uint8(segImx)
        #print("FOLRMA: " + str(segImx.shape))
        ##segImx = np.expand_dims(segImx, axis=-1)
        ##segImx = ImageOps.autocontrast(keras.preprocessing.image.array_to_img(segImx))
        #
        #self.separar_cores_seg(segImx)

        ###### NON BORRAR QUE AO MELLOR E IMPORTANTE ########
        ##segImx = keras.preprocessing.image.array_to_img(segImx)
        ###### NON BORRAR QUE AO MELLOR E IMPORTANTE ########


        ## Primeiro vai a Imaxe normal
        #i1 = cv.imread(imaxe_dir)
        #i1 = cv.cvtColor(i1, cv.COLOR_RGB2BGR)
        ## Despois a mascara
        #dirs = imaxe_dir.split("Imaxes")
        #i2 = cv.imread(dirs[0]+"Mascaras"+dirs[1], 0)
        #print("Dim I2: " + str(i2.shape))
        #self.separar_cores(i2)
        #i2 = cv.cvtColor(i2, cv.COLOR_GRAY2BGR)
        #stac1 = np.hstack((i1,i2))
        ## Por ultimo, a segmentada
        #i3 = cv.cvtColor(segImx, cv.COLOR_GRAY2BGR)
        #stac2 = np.hstack((stac1,i3))
        #stac2 = Image.fromarray(stac2)
        #glibbytes = GLib.Bytes.new( stac2.tobytes() )
        #gdkpixbuf = GdkPixbuf.Pixbuf.new_from_data( glibbytes.get_data(), GdkPixbuf.Colorspace.RGB, False, 8, \
        #        stac2.width, stac2.height, len( stac2.getbands() )*stac2.width, None, None ) # Line B

        return gdkpixbuf
    def predict_coord(self, modelo_dir, coord):
        pass
    
    def predict_dir(self, modelo_dir, directorio_dir, despr):
        k = 1
        imaxes = os.listdir(directorio_dir)
        for imx in imaxes:
            imx_seg = self.componher_imx(modelo_dir, os.path.join(directorio_dir,imx))
            imx_seg.save(directorio_dir+"/s"+imx.split(".png")[0]+".png")
            k+=1

    def predict_dir2(self, modelo_dir, directorio_dir, despr):
        #cnnModelo = keras.models.load_model(modelo_dir)
        directorio_dir += "/Imaxes"
        test_input_dirs = sorted([os.path.join(directorio_dir, fnome) \
            for fnome in os.listdir(directorio_dir)])
        
        if despr == "+1":
            self.idx_imx += 1
            min(self.idx_imx,len(test_input_dirs))
        elif despr == "-1":
            self.idx_imx -= 1
            max(self.idx_imx, 0)
        elif despr == "":
            self.idx_imx = 0
        else:
            self.idx_imx = int(despr)
        # Crear lista de gdkpixbuf para mostrar
        gdkpixbuf_lista = []
        return (self.idx_imx,
                len(test_input_dirs),
                self.predict_imx(modelo_dir, test_input_dirs[self.idx_imx]) )



    def separar_cores_comb(self, imx_edi, imx_est):
        w,h = imx_edi.shape
        nova_imx = np.zeros((w, h, 3), np.uint8)
        # BGR
        for i in range(w):
            for j in range(h):
                if (imx_edi[i][j] == 1) and (imx_est[i][j] == 0):
                    nova_imx[i][j] = (0,0,255)
                elif (imx_edi[i][j] == 0) and (imx_est[i][j] == 1):
                    nova_imx[i][j] = (255,0,0)
                elif (imx_edi[i][j] == 0) and (imx_est[i][j] == 0):
                    nova_imx[i][j] = (0,255,0)
                elif (imx_edi[i][j] == 1) and (imx_est[i][j] == 1):
                    nova_imx[i][j] = (102,0,102)

        return nova_imx



    def separar_cores(self, imx_arr):
        w, h = imx_arr.shape
        for i in range(w):
            for j in range(h):
                if imx_arr[i][j] == 1: # Urbanas
                    imx_arr[i][j] = 255
                elif imx_arr[i][j] == 0:
                    imx_arr[i][j] = 0
                else:
                    imx_arr[i][j] = 0 # Libre

    def separar_cores_seg(self, imx_arr):
        w, h = imx_arr.shape
        for i in range(w):
            for j in range(h):
                if imx_arr[i][j] == 1: # Urbanas
                    imx_arr[i][j] = 255
                elif imx_arr[i][j] == 2: # Estradas
                    imx_arr[i][j] = 128
                else:
                    imx_arr[i][j] = 0 # Libre


class UAVImaxes(keras.utils.Sequence):
    def __init__(self, input_imxs, imx_size=(128,128)):
        self.imx_size = imx_size
        self.input_imxs = input_imxs
    def __len__(self):
        return len(self.input_imxs)
    def __getitem__(self, idx):
        i = idx
        x = np.zeros((len(self.input_imxs),) + self.imx_size + (3,), dtype="float32")
        for j, path in enumerate(self.input_imxs):
            #print("PATH: "+ str(path))
            imx = load_img(path, target_size=self.imx_size)
            x[j] = imx
        return x



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
