import os
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gio, GdkPixbuf, GLib
import cv2
from PIL import Image
import numpy as np
import cv2
from controlador import Controlador


class XanelaFicheiros(Gtk.Window):
    def __init__(self, v):
        super().__init__(title="Gardar ficheiro")
        
        dialog = Gtk.FileChooserDialog( 
                title="Escolla o directorio", 
                parent=self, 
                action = Gtk.FileChooserAction.SAVE,
            )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL, 
            Gtk.STOCK_SAVE,
            Gtk.ResponseType.OK,
        )
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            imx_tmp = cv2.cvtColor(v.fich, cv2.COLOR_RGB2BGR) 
            cv2.imwrite(dialog.get_filename(),imx_tmp)
            #c.set_directorio(dialog.get_filename())
        elif response == Gtk.ResponseType.CANCEL:
            print("Cancel clicked")
        dialog.destroy()


class ExploradorFicheiros(Gtk.Window):
    def __init__(self, c):
        super().__init__(title="Explorador de Arquivos")
        
        dialog = Gtk.FileChooserDialog( 
                title="Escolla a imaxe", 
                parent=self, 
                action = Gtk.FileChooserAction.OPEN,
            )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL, 
            Gtk.STOCK_OPEN,
            Gtk.ResponseType.OK,
        )
        filtro_imaxe = Gtk.FileFilter()
        filtro_imaxe.set_name("ficheiro imaxe")
        filtro_imaxe.add_pattern("*.png")
        dialog.add_filter(filtro_imaxe)
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            c.directorio_dir = dialog.get_filename()
            c.etiq_imx.set_label(dialog.get_filename())

            orixinal = cv2.imread(dialog.get_filename())
            preview = cv2.resize(orixinal, (400,400), interpolation = cv2.INTER_NEAREST)
            preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
            preview = Image.fromarray(preview)
            glibbytes = GLib.Bytes.new( preview.tobytes() )
            gdkpixbuf = GdkPixbuf.Pixbuf.new_from_data( glibbytes.get_data(), GdkPixbuf.Colorspace.RGB, False, 8,\
                    preview.width, preview.height, len( preview.getbands() )*preview.width, None, None )

            #Line B
            c.imx.set_from_pixbuf(gdkpixbuf)
            c.raiz._win.show_all()
            #c.set_directorio(dialog.get_filename())
        elif response == Gtk.ResponseType.CANCEL:
            print("Cancel clicked")
        dialog.destroy()



class ApCoordenadas():
    def __init__(self, raiz):
        for elemento in raiz._win.get_children():
            raiz._win.remove(elemento)
        self.fich = None # Ficheiro recibido do modelo
        label_lat = Gtk.Label("Latitude: ")    
        label_lon = Gtk.Label("Lonxitude: ")
        label_tipo_modelo = Gtk.Label("Tipo de modelo: ")
        self.entrad_lat = Gtk.Entry()
        self.entrad_lon = Gtk.Entry()
        self.boton_aceptar = Gtk.Button(label="Aceptar")
        self.boton_atras = Gtk.Button(label="Atras")
        combo = Gtk.ListStore(int, str)
        combo.append([1, "ResNet50"])
        combo.append([2, "Convolucional"])
        combo.append([3, "U-Net"])
        self.seleccionador = Gtk.ComboBox()
        self.combo = Gtk.ComboBox.new_with_model_and_entry(combo)
        self.combo.set_entry_text_column(1)
        self.combo.set_active(0)
        self.box_buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10, margin=25)
        self.box_buttons.pack_start(label_lat, False, True, 0)
        self.box_buttons.pack_start(self.entrad_lat, False, True, 0)
        self.box_buttons.pack_start(label_lon, False, True, 0)
        self.box_buttons.pack_start(self.entrad_lon, False, True, 0)
        #self.box_buttons.pack_start(self.boton_aceptar, False, True, 0)
        #self.box_buttons.pack_start(self.boton_atras, False, True, 0)
        self.box_buttons.pack_end(self.combo, False, True, 0)
        self.box_buttons.pack_end(label_tipo_modelo, False, True, 0)
        self.box_buttons.set_hexpand(True); self.box_buttons.set_vexpand(False);
        #self.box_buttons.pack_start(self.boton_aceptar, False, True, 0)
        #self.box_buttons.pack_start(self.boton_atras, False, True, 0)
        
        self.box_aceptar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10, margin=25)
        self.box_aceptar.pack_start(self.boton_aceptar, False, True, 0)
        self.box_atras = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10, margin=25)
        self.box_atras.pack_start(self.boton_atras, False, True, 0)
        self.boton_gardar = Gtk.Button(label="Gardar")
        #self.box_buttons2.pack_start(self.boton_gardar, False, True, 0)
        #self.box_buttons2.pack_start(self.boton_gardar, False, True, 0)
        self.imx = Gtk.Image()
        self.evBox = Gtk.EventBox()
        self.evBox.add(self.imx)
        grid = Gtk.Grid(margin=0, column_spacing=0, row_spacing=0)
        grid.set_hexpand(False); grid.set_vexpand(False)
        grid.attach(self.box_buttons, 0, 0, 1, 1)
        grid.attach(self.box_aceptar, 0, 1, 1, 1)
        grid.attach(self.evBox, 0, 2, 1, 1)
        grid.attach(self.box_atras, 0, 3, 1, 1)
        #grid.attach(self.box_buttons2, 0, 2, 1, 1)
        
        preview = np.uint8(np.ones((400,400,3)) * 255)
        preview = Image.fromarray(preview)
        glibbytes = GLib.Bytes.new( preview.tobytes() )
        gdkpixbuf = GdkPixbuf.Pixbuf.new_from_data( glibbytes.get_data(), GdkPixbuf.Colorspace.RGB, False, 8, \
                preview.width, preview.height, len( preview.getbands() )*preview.width, None, None )
        self.imx.set_from_pixbuf(gdkpixbuf)
        self.raiz = raiz
        self.raiz._win.add(grid)

    def connect(self, c):
        self.boton_aceptar.connect("clicked", self.obter_resultados, c) 
        self.boton_gardar.connect("clicked", self.gardar_resultados)
        self.boton_atras.connect("clicked", self.atras)

    def obter_resultados(self, cousa, c):
        id_modelo = 0
        if self.combo.get_child().get_text() == "Convolucional":
            id_modelo = 0
        elif self.combo.get_child().get_text() == "ResNet50":
            id_modelo = 1
        elif self.combo.get_child().get_text() == "U-Net":
            id_modelo = 2
        self.fich = c.xenerarPredCoord((self.entrad_lat.get_text(), self.entrad_lon.get_text(), \
                id_modelo ) )
        preview = cv2.resize(self.fich, (400, 400),interpolation = cv2.INTER_NEAREST)
        preview = Image.fromarray(preview)
        glibbytes = GLib.Bytes.new( preview.tobytes() )
        gdkpixbuf = GdkPixbuf.Pixbuf.new_from_data( glibbytes.get_data(), GdkPixbuf.Colorspace.RGB, False, 8, \
            preview.width, preview.height, len( preview.getbands() )*preview.width, None, None ) # Line B
        self.imx.set_from_pixbuf(gdkpixbuf)
        self.box_aceptar.pack_start(self.boton_gardar, False, True, 0)
        self.raiz._win.show_all()

    def gardar_resultados(self, cousa):
        explorador = XanelaFicheiros(self)
        #cv2.imwrite(·"")

    def atras(self, cousa):
        self.raiz.cambiarAMenu()

class ApSubirImx():
    def __init__(self, raiz):
        for elemento in raiz._win.get_children():
            raiz._win.remove(elemento)
        self.fich = None
        self.directorio_dir = ""
        self.label_cargar = Gtk.Label("Cargar imaxe: ")
        self.label_tipo_modelo = Gtk.Label("Tipo de Modelo: ")
        self.boton_imx = Gtk.Button(label="...")
        self.boton_aceptar = Gtk.Button(label="Aceptar")
        self.boton_atras = Gtk.Button(label="Atras")
        self.etiq_imx = Gtk.Label("")
        combo = Gtk.ListStore(int, str)
        combo.append([1, "ResNet50"])
        combo.append([2, "Convolucional"])
        combo.append([3, "U-Net"])
        self.seleccionador = Gtk.ComboBox()
        self.combo = Gtk.ComboBox.new_with_model_and_entry(combo)
        self.combo.set_entry_text_column(1)
        self.combo.set_active(0)
        self.box_buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0, margin=25)
        self.box_buttons.pack_start(self.label_cargar, False, True, 0)
        self.box_buttons.pack_start(self.boton_imx, False, True, 0)
        self.box_buttons.pack_start(self.etiq_imx, False, True, 0)
        self.box_buttons.pack_start(self.label_cargar, False, True, 0)
        #self.box_buttons.pack_start(self.boton_aceptar, False, True, 0)
        #self.box_buttons.pack_start(self.boton_atras, False, True, 0)
        self.box_buttons.pack_end(self.combo, False, True, 0)
        self.box_buttons.pack_end(self.label_tipo_modelo, False, True, 0)
        self.box_buttons.set_hexpand(True); self.box_buttons.set_vexpand(False);
        self.box_aceptar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10, margin=25)
        self.box_aceptar.pack_start(self.boton_aceptar, False, True, 0)
        self.box_atras = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10, margin=25)
        self.box_atras.pack_start(self.boton_atras, False, True, 0)

        self.boton_gardar = Gtk.Button(label="Gardar")

        self.imx = Gtk.Image()
        self.evBox = Gtk.EventBox()
        self.evBox.add(self.imx)
        grid = Gtk.Grid(margin=0, column_spacing=0, row_spacing=0)
        grid.set_hexpand(False); grid.set_vexpand(False)
        #self.box_buttons2 = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0, margin=0)
        grid.attach(self.box_buttons, 0, 0, 1, 1)
        grid.attach(self.box_aceptar, 0, 1, 1, 1)
        grid.attach(self.evBox, 0, 2, 1, 1)
        grid.attach(self.box_atras, 0, 3, 1, 1)
        
        preview = np.uint8(np.ones((400,400,3)) * 255)
        preview = Image.fromarray(preview)
        glibbytes = GLib.Bytes.new( preview.tobytes() )
        gdkpixbuf = GdkPixbuf.Pixbuf.new_from_data( glibbytes.get_data(), GdkPixbuf.Colorspace.RGB, False, 8, \
                preview.width, preview.height, len( preview.getbands() )*preview.width, None, None )
        self.imx.set_from_pixbuf(gdkpixbuf)

        self.raiz = raiz
        self.raiz._win.add(grid)
    
    def connect(self, c):
        self.boton_atras.connect("clicked", self.atras)
        self.boton_imx.connect("clicked", self.cargarDirectorio)
        self.boton_aceptar.connect("clicked", self.obter_resultados, c)
        self.boton_gardar.connect("clicked", self.gardar_resultados)
    
    def atras(self, cousa):
        self.raiz.cambiarAMenu()

    def cargarDirectorio(self, nome_ficheiro):
        explorador = ExploradorFicheiros(self)
    
    def obter_resultados(self, cousa, c):
        imx = cv2.imread(self.directorio_dir)
        id_modelo = 0
        if self.combo.get_child().get_text() == "Convolucional":
            id_modelo = 0
        elif self.combo.get_child().get_text() == "ResNet50":
            id_modelo = 1
        elif self.combo.get_child().get_text() == "U-Net":
            id_modelo = 2
        self.fich = c.xenerarPredImx((imx, id_modelo))
        preview = cv2.resize(self.fich, (400, 400),interpolation = cv2.INTER_NEAREST)
        preview = Image.fromarray(preview)
        glibbytes = GLib.Bytes.new( preview.tobytes() )
        gdkpixbuf = GdkPixbuf.Pixbuf.new_from_data( glibbytes.get_data(), GdkPixbuf.Colorspace.RGB, False, 8, \
            preview.width, preview.height, len( preview.getbands() )*preview.width, None, None ) # Line B
        self.imx.set_from_pixbuf(gdkpixbuf)
        self.box_aceptar.pack_start(self.boton_gardar, False, True, 0)
        #self.box_buttons2.pack_start(self.boton_gardar, False, True, 0)
        self.raiz._win.show_all()
    
    def gardar_resultados(self, cousa):
        explorador = XanelaFicheiros(self)


class Vista():
    def __init__(self):
        self.controlador = Controlador()
        self.boton_coord = Gtk.Button(label="Introducir coordenadas")
        self.boton_subirImx = Gtk.Button(label="Subir Imaxe")
        self.box_buttons = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0, margin=250)
        self.box_buttons.pack_start(self.boton_coord, True, False, 0)
        self.box_buttons.pack_start(self.boton_subirImx, False, True, 20)
        self.box_buttons.set_hexpand(True); self.box_buttons.set_vexpand(True);
        grid = Gtk.Grid(margin=0, column_spacing=0, row_spacing=0)
        grid.set_hexpand(False); grid.set_vexpand(True)
        grid.attach(self.box_buttons, 0, 0, 1, 1)
        
        self._win = Gtk.Window(title="Aplicacion")
        self._win.connect("delete-event", Gtk.main_quit)
        self._win.add(grid)
        self._win.set_default_size(900, 600)
        self.interfaz = self
        self.connect()
        self.show_win() 
        Gtk.main()

    def layout(self): 
        self.boton_coord = Gtk.Button(label="Introducir coordenadas")
        self.boton_subirImx = Gtk.Button(label="Subir Imaxe")
        self.box_buttons = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0, margin=250)
        self.box_buttons.pack_start(self.boton_coord, True, False, 0)
        self.box_buttons.pack_start(self.boton_subirImx, True, False, 0)
        self.boton_coord.set_hexpand(False)
        self.box_buttons.set_hexpand(True); self.box_buttons.set_vexpand(True);
        grid = Gtk.Grid(margin=0, column_spacing=0, row_spacing=0)
        grid.set_hexpand(False); grid.set_vexpand(False)
        grid.attach(self.box_buttons, 0, 0, 1, 1) 
        self._win.add(grid)

    def connect(self):
        self.boton_coord.connect("clicked", self.cambiarAcoord)
        self.boton_subirImx.connect("clicked", self.cambiarAsubImx)
    
    def cambiarAcoord(self, cousa):
        a = ApCoordenadas(self)
        a.connect(self.controlador)
        self._win.show_all()

    def cambiarAsubImx(self, cousa):
        a = ApSubirImx(self)
        a.connect(self.controlador)
        self._win.show_all()

    def cambiarAMenu(self):
        for elemento in self._win.get_children():
            self._win.remove(elemento)
        self.layout()
        self.connect()
        self.show_win()


    def show_win(self):
        self._win.show_all()
