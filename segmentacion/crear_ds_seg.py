import os
import osmium
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImagePath
import pyproj

class UAVzrHandler(osmium.SimpleHandler): # Zonas restrinxidas UAV
    def __init__(self):
        super(UAVzrHandler, self).__init__()
        self.areas_residenciais = [];
        self.autoestradas = [] # motorway
        self.nacionais = [] # trunk, primary
        self.autonomicas = [] # secondary
        self.terciarias = [] # terciaria
        self.pistas = [] # residential
        self.railes = []
        self.tipos_edificio = ['apartments', 'detached', 'dormitory', 'hotel', 'residential', 'house',
                                'commercial', 'industrial', 'office', 'retail', 'supermarket', 'warehouse',
                                'civic', 'college', 'government', 'hospital', 'public', 'school', 'university',
                                'yes']

    def atopase_na_rexion(self, nodos):
        for nodo in nodos:
            if (nodo[0] > c1[0] and nodo[0] < c2[0]) and (nodo[1] > c1[1] and nodo[1] < c2[1]):
                return True
        return False

    def way(self, w):
        nodos = []
        for nodo in w.nodes:
            nodos.append(t.transform(nodo.location.lat, nodo.location.lon))
        if self.atopase_na_rexion(nodos):
            if w.tags.get('building') in self.tipos_edificio:
                self.areas_residenciais.append(nodos)
            elif w.tags.get('highway') == 'motorway':
                self.autoestradas.append(nodos)
            elif (w.tags.get('highway') == 'trunk') or (w.tags.get('highway') == 'primary') \
                    or (w.tags.get('highway') == 'motorway_link'):
                self.nacionais.append(nodos)
            elif w.tags.get('highway') == 'secondary':
                self.autonomicas.append(nodos)
            elif w.tags.get('highway') == 'tertiary':
                self.terciarias.append(nodos)
            elif (w.tags.get('highway') == 'residential') or (w.tags.get('highway') == 'unclassified') \
                    or (w.tags.get('highway') == 'service'):
                self.pistas.append(nodos)
            elif (w.tags.get('highway') == 'railway') or (w.tags.get('landuse') == 'railway'):
                self.railes.append(nodos)


def a_coord(nodo, dim=(10000,10000)):
    anchura, altura = dim
    dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
    x = round( (nodo[0] - p1[0]) / dx * anchura )
    y = altura - round( (nodo[1] - p1[1]) / dy * altura )
    return (x,y)


def mkdir_se_non_existe(direccion):
    try:
        os.mkdir(direccion)
    except FileExistsError:
        pass

def atopase_na_rexion(nodos, p1, p2):
    for nodo in nodos:
        if (nodo[0] > p1[0] and nodo[0] < p2[0]) and (nodo[1] > p1[1] and nodo[1] < p2[1]):
                return True
    return False


def get_nodos(lista_nodos, p1, p2):
    n = []
    for nodos in lista_nodos:
        if atopase_na_rexion(nodos, p1, p2):
            n.append(nodos)
    return n

def mostrear2(imx, gt, gt_ed, gt_est, nome):
    imx_ = cv2.resize(imx, (1000, 1000),interpolation = cv2.INTER_NEAREST)
    gt_ = cv2.resize(gt, (1000, 1000),interpolation = cv2.INTER_NEAREST)
    visual = cv2.addWeighted(imx_, 0.85, gt_, 0.15, 0.0)
    cv2.imwrite(f"./DS_Estradas/debug/{nome}.png", visual)

def mostrear(imx, gt, gt_ed, gt_est, nome):
    tam = 2000; contador = 0
    for i in range(0, 10000, tam):
        for j in range(0, 10000, tam):
            #rexion = (j, i, j+tam, i+tam)
            imx_ = imx[j:j+tam,i:i+tam]; gt_ = gt[j:j+tam,i:i+tam]
            gt_ed_ = gt_ed[j:j+tam,i:i+tam]; gt_est_ = gt_est[j:j+tam,i:i+tam]
            imx_ = cv2.resize(imx_, (512, 512),interpolation = cv2.INTER_NEAREST)
            gt_ = cv2.resize(gt_, (512, 512),interpolation = cv2.INTER_NEAREST)
            gt_ed_ = cv2.resize(gt_ed_, (512, 512),interpolation = cv2.INTER_NEAREST)
            gt_est_ = cv2.resize(gt_est_, (512, 512),interpolation = cv2.INTER_NEAREST)
            visual = cv2.addWeighted(imx_, 0.85, gt_, 0.15, 0.0)
            #cv2.imwrite(f"./testeo/{nome}-{str(contador)}.png", visual)
            cv2.imwrite(f"./DS_Estradas/imx/{nome}-{str(contador)}.png", imx_)
            cv2.imwrite(f"./DS_Estradas/gt/{nome}-{str(contador)}.png", gt_est_)
            cv2.imwrite(f"./DS_Estradas/debug/{nome}-{str(contador)}.png", visual)
            
            cv2.imwrite(f"./DS_Edificios/imx/{nome}-{str(contador)}.png", imx_)
            cv2.imwrite(f"./DS_Edificios/gt/{nome}-{str(contador)}.png", gt_ed_)
            cv2.imwrite(f"./DS_Edificios/debug/{nome}-{str(contador)}.png", visual)
            contador += 1


if __name__=="__main__":
    t = pyproj.Transformer.from_crs("wgs 84", "epsg:25832")
    mkdir_se_non_existe("./DS_Estradas")
    mkdir_se_non_existe("./DS_Estradas/imx")
    mkdir_se_non_existe("./DS_Estradas/gt")
    mkdir_se_non_existe("./DS_Estradas/debug")
    mkdir_se_non_existe("./DS_Edificios")
    mkdir_se_non_existe("./DS_Edificios/imx")
    mkdir_se_non_existe("./DS_Edificios/gt")
    mkdir_se_non_existe("./DS_Edificios/debug")
    for f in os.listdir("./DS_Estradas/imx"):
        os.remove(os.path.join("./DS_Estradas/imx", f))
    for f in os.listdir("./DS_Estradas/gt"):
        os.remove(os.path.join("./DS_Estradas/gt", f))
    for f in os.listdir("./DS_Estradas/debug"):
        os.remove(os.path.join("./DS_Estradas/debug", f))

    for f in os.listdir("./DS_Edificios/imx"):
        os.remove(os.path.join("./DS_Edificios/imx", f))
    for f in os.listdir("./DS_Edificios/gt"):
        os.remove(os.path.join("./DS_Edificios/gt", f))
    for f in os.listdir("./DS_Edificios/debug"):
        os.remove(os.path.join("./DS_Edificios/debug", f))

    dir_principal = "/home/wouter/Desktop/fotos_proba2"
    fotos = os.listdir(dir_principal)
    c = []
    for foto in fotos:
        c.append((int(foto.split("_")[1]), int(foto.split("_")[3].split(".")[0])))
    c.sort(key=lambda y: y[0]+y[1])
    c1 = (c[0][0] - 500, c[0][1] - 500); c2 = (c[-1][0] + 500, c[-1][1] + 500)
    uavzr = UAVzrHandler()
    uavzr.apply_file("./muenster-regbez-latest.osm.pbf", locations=True)

    for k,foto in enumerate(fotos):
        coord = foto.split("_")
        cx = int(coord[1]); cy = int(coord[3].split(".")[0])
        p1 = (cx - 500, cy - 500); p2 = (cx + 500, cy + 500)
        
        ar = get_nodos(uavzr.areas_residenciais, p1, p2)
        autoestradas =  get_nodos(uavzr.autoestradas, p1, p2)
        nacionais = get_nodos(uavzr.nacionais, p1, p2)
        autonomicas = get_nodos(uavzr.autonomicas, p1, p2)
        terciarias = get_nodos(uavzr.terciarias, p1, p2)
        pistas = get_nodos(uavzr.pistas, p1, p2)
        railes = get_nodos(uavzr.railes, p1, p2)

        img = Image.new(mode="RGB", size=(10000, 10000))
        gt_estradas = Image.new("L", (10000, 10000), 0)
        gt_edificios = Image.new("L", (10000, 10000), 0)
        imgd = ImageDraw.Draw(img)
        gt_estradas_d = ImageDraw.Draw(gt_estradas)
        gt_edificios_d = ImageDraw.Draw(gt_edificios)
        for rexion in ar:
            lptos = []
            for nodo in rexion:
                lptos.append(a_coord(nodo))
            imgd.polygon(lptos, fill=(0,0,255))
            gt_edificios_d.polygon(lptos, fill=1)
         
        for rexion in autoestradas:
            lptos = ()
            for nodo in rexion:
                lptos += a_coord(nodo)
            imgd.line(lptos, fill=(255,0,0), width=160) 
            gt_estradas_d.line(lptos, fill=1, width=160)
       
        for rexion in nacionais:
            lptos = ()
            for nodo in rexion:
                lptos += a_coord(nodo)
            imgd.line(lptos, fill=(255,0,0), width=100) 
            gt_estradas_d.line(lptos, fill=1, width=100)
        
        for rexion in autonomicas:
            lptos = ()
            for nodo in rexion:
                lptos += a_coord(nodo)
            imgd.line(lptos, fill=(255,0,0), width=95) 
            gt_estradas_d.line(lptos, fill=1, width=95)
        
        for rexion in terciarias:
            lptos = ()
            for nodo in rexion:
                lptos += a_coord(nodo)
            imgd.line(lptos, fill=(255,0,0), width=90) 
            gt_estradas_d.line(lptos, fill=1, width=90)
       
        for rexion in pistas:
            lptos = ()
            for nodo in rexion:
                lptos += a_coord(nodo)
            imgd.line(lptos, fill=(255,0,0), width=45) 
            gt_estradas_d.line(lptos, fill=1, width=45)

        for rexion in railes:
            lptos = ()
            for nodo in rexion:
                lptos += a_coord(nodo)
            imgd.line(lptos, fill=(255,0,0), width=75) 
            gt_estradas_d.line(lptos, fill=1, width=75)

        gt = np.array(img); gt_estradas = np.array(gt_estradas); gt_edificios = np.array(gt_edificios) 
        gt = gt[:,:,::-1].copy(); #foto.split(".")[0] 
        mostrear2(cv2.imread(os.path.join(dir_principal, foto)), \
                gt, gt_edificios, gt_estradas, foto.split(".")[0])
