#
# Script de python para a creacion dun dataset coa seguinte 
# estructura no directorio actual:
# 
# - DS
#    |
#    |- Adestramento        (70% do total de imaxes)
#    |      |
#    |      |- Imaxes
#    |      |
#    |      |- Mascaras
#    |
#    |- Validacion          (25% do total de imaxes)
#    |      |
#    |      |- Imaxes
#    |      |
#    |      |- Mascaras
#    |
#    |- Test                (5 % do total de imaxes)
#    |      |
#    |      |- Imaxes
#    |      |
#    |      |- Mascaras
#
#
#
# Extrae informacion xeografica de OSM (www.openstreetmap.org) 
# para a realizacion das mascaras, facendo uso da
# Open Data Database License (ODbL)



import os
import shutil
import sys
import math
import osmium
import numpy as np
from PIL import Image, ImageDraw, ImagePath
import cv2



class UAVzrHandler(osmium.SimpleHandler): # Zonas restrinxidas UAV
    def __init__(self, lon1, lat1, lon2, lat2):
        super(UAVzrHandler, self).__init__()
        self.areas_residenciais = []
        self.areas_industriais = []
        self.infraestructura = []
        self.parques = []
        self.lugar = ((lat1, lon1), (lat2, lon2))
        self.tipos_estrada = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', \
                              'residential', 'railway']
        self.tipos_edificio = ['apartments', 'detached', 'dormitory', 'hotel', 'residential', 'house',
                                'commercial', 'industrial', 'office', 'retail', 'supermarket', 'warehouse',
                                'civic', 'college', 'government', 'hospital', 'public', 'school', 'university',
                                'yes']

    def atopase_na_rexion(self, nodos):
        for nodo in nodos:
            location = nodo.location
            if (location.lat > self.lugar[1][0] and location.lat < self.lugar[0][0]) and \
                    (location.lon < self.lugar[1][1] and location.lon > self.lugar[0][1]):
                        return True
        return False
    
    #(w.tags.get('landuse') in ['residential', 'comercial']) or \
    def way(self, w):
        if self.atopase_na_rexion(w.nodes):
            if ( w.tags.get('building') in self.tipos_edificio ):
                locs = []
                for nodo in w.nodes:
                    locs.append(nodo.location)
                self.areas_residenciais.append(locs)
            elif ((w.tags.get('highway') in self.tipos_estrada) or (w.tags.get('landuse') == 'railway')): 
                locs = []
                for nodo in w.nodes:
                    locs.append(nodo.location)
                self.infraestructura.append(locs)
            elif ((w.tags.get('landuse') in ['meadow']) or (w.tags.get('leisure') in ['park','garden'])):
                locs = []
                for nodo in w.nodes:
                    locs.append(nodo.location)
                self.parques.append(locs)




def establecer_encadre(lat_deg, lon_deg, RADIO_VLOS=500):
    RADIO_TERRESTRE = 6371000
    lat_rad = math.radians(lat_deg)
    lat1 = math.degrees(RADIO_VLOS / RADIO_TERRESTRE + lat_rad)

    # Calculo de n_lon deg partindo de (n_lat_deg, n_lon_deg)
    n_lat_rad = math.radians(lat1)
    lon_rad = math.radians(lon_deg)
    cte1 = 1 - math.cos(n_lat_rad)**2
    cte2 = math.cos(n_lat_rad)**2
    n_lon_rad = lon_rad - math.acos((math.cos(RADIO_VLOS/RADIO_TERRESTRE) - cte1) / cte2)
    lon1 = math.degrees(n_lon_rad)
    d_lon = 2 * abs(lon1 - lon_deg)
    d_lat = 2 * abs(lat1 - lat_deg)
    lon2 = lon1 + d_lon
    lat2 = lat1 - d_lat
    return (lon1, lat1, lon2, lat2)

def recortar_sat(imx, altura, anchura, dirc):
    nome = dirc.split(".")[1]
    imxAnchura, imxAltura = imx.size
    k = 0
    for i in range(0, imxAltura, altura):
        for j in range(0, imxAnchura, anchura):
            caixa = (j, i, j + anchura, i + altura)
            a = imx.crop(caixa)
            nome_tmp = "." + nome.split("-")[0] + "-%s.png" % k
            a.save(nome_tmp)
            k+=1

def recortar(imx, altura, anchura, dirc):
    nome = dirc.split(".")[1]
    imxAnchura, imxAltura = imx.size
    k = 0
    for i in range(0, imxAltura, altura):
        for j in range(0, imxAnchura, anchura):
            caixa = (j, i, j + anchura, i + altura)
            a = imx.crop(caixa)
            nome_tmp = "." + nome.split("-")[0] + "-%s.png" % k
                #a = filtrar(a)
                # Verificar se e edificio / estrada / libre
            
            num_libre = np.count_nonzero(np.array(a) == 0)
            num_estradas = np.count_nonzero(np.array(a) == 2)
            num_edificios = np.count_nonzero(np.array(a) == 1)
            print("NUM px estradas: " + str(num_estradas) + " \nNUM px edificios: " + str(num_edificios))
            tmp_tmp = nome_tmp.split("Mascaras")
            if (num_estradas > num_edificios) and (num_estradas == 2500):
                shutil.move(tmp_tmp[0] + "Imaxes" + tmp_tmp[1], tmp_tmp[0] + "Estradas" + tmp_tmp[1])
            elif (num_edificios > num_estradas) and (num_edificios == 2500):
                shutil.move(tmp_tmp[0] + "Imaxes" + tmp_tmp[1], tmp_tmp[0] + "Edificios" + tmp_tmp[1])
            elif (num_libre == 2500):
                shutil.move(tmp_tmp[0] + "Imaxes" + tmp_tmp[1], tmp_tmp[0] + "Libre" + tmp_tmp[1])
            k += 1


def a_coord(loc, dim=(10000,10000)):
    # Recibes unha coordenada e devolve
    # A posicion en pixeles da imaxe
    anchura, altura = dim
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    x = (loc.lon - lon1) / dlon * anchura
    y = (loc.lat - lat1) / dlat * altura
    return (x,y)


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Erro ao parsear os argumentos\n \
               Uso: python crear_ds.py [dir_a_imaxes_satelite] [dir_a_ficheiro_osm_pbf]")
        exit()


    # Borra o directorio se existe previamente
    if os.path.exists("./DS"):
        shutil.rmtree("./DS/")

    # Crea a estructura de directorios
    os.mkdir("./DS")
    os.mkdir("./DS/Adestramento")
    os.mkdir("./DS/Adestramento/Edificios")
    os.mkdir("./DS/Adestramento/Estradas")
    os.mkdir("./DS/Adestramento/Libre")
    os.mkdir("./DS/Adestramento/Imaxes")
    os.mkdir("./DS/Adestramento/Mascaras")

    directorio = sys.argv[1]

    for imx_dir in os.listdir(directorio):
        dir_mascara = "./DS/Adestramento/Mascaras/" + str(imx_dir)
        dir_sat = "./DS/Adestramento/Imaxes/" + str(imx_dir)
        #if imx_dir.split("-")[1] != "0.png":
        #    imx_sat = Image.open(os.path.join(directorio, imx_dir))
        #    imx_sat.save(dir_sat)
        #    continue
        coord = imx_dir.split("_")
        lat_nat = float(coord[1])
        lat_dec = float(coord[2])
        lon_nat = float(coord[4])
        lon_dec = float(coord[5].split(".")[0])
        coord_lat = lat_nat + (lat_dec/1000000)
        coord_lon = lon_nat + (lon_dec/1000000)

        lon1, lat1, lon2, lat2 = establecer_encadre(coord_lat, coord_lon)
        bbox = (lon1, lat1, lon2, lat2)

        uavzr = UAVzrHandler(lon1, lat1, lon2, lat2)
        uavzr.apply_file(sys.argv[2], locations=True)

        img = Image.new("L", (10000, 10000), 0)
        imgd = ImageDraw.Draw(img)
        for rexion in uavzr.areas_residenciais:
            lptos = []
            for nodo in rexion:
                lptos.append(a_coord(nodo))
            imgd.polygon(lptos, fill=1)
        

        for rexion in uavzr.parques:
            lptos = []
            for nodo in rexion:
                lptos.append(a_coord(nodo))
            imgd.polygon(lptos, fill=0)
        
        for rexion in uavzr.infraestructura:
            lptos = ()
            for nodo in rexion:
                lptos += a_coord(nodo)

            imgd.line(lptos, fill=2, width=70) 


        print("ESTO: " + str(dir_mascara))

        imx_sat = Image.open(os.path.join(directorio, imx_dir))
        recortar_sat(imx_sat, 50, 50, dir_sat)
        recortar(img, 50, 50, dir_mascara)
