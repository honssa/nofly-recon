import requests
import sys
import os
import math
from pyproj import Transformer
from tqdm import tqdm

def wgs84_a_EPSG25832(x, y):
    transformer = Transformer.from_crs("wgs 84", "epsg:25832")
    x2, y2 = transformer.transform(x, y)
    x2 = math.floor(x2); y2 = math.floor(y2)
    x2 -= x2%1000; y2 -= y2%1000
    return (x2, y2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Erro ao parsear os argumentos. Uso do script:\n Uso: python descargar.py [LAT] [LON]")
        exit()
    
    x, y = wgs84_a_EPSG25832(float(sys.argv[1]), float(sys.argv[2]))
    xmin = x - 5000; ymin = y - 5000; xmax= x + 5000; ymax = y + 5000
    os.mkdir("fotos_descargadas")
    
    with tqdm(total=100) as pbar:
        for yc in range(ymin, ymax, 1000):
            for xc in range(xmin, xmax, 1000):
                url = "https://www.wcs.nrw.de/geobasis/wcs_nw_dop?" + \
                "REQUEST=GetCoverage&" + \
                "SERVICE=WCS&" + \
                "VERSION=2.0.1&" + \
                "COVERAGEID=nw_dop&" + \
                "FORMAT=image/png; mode=8bit&" + \
                "SUBSET=x({x_1},{x_2})&".format(x_1=xc, x_2=xc+1000) + \
                "SUBSET=y({y_1},{y_2})&".format(y_1=yc, y_2=yc+1000) + \
                "RANGESUBSET=1,2,3&" + \
                "OUTFILE=dop10rgb_32_408_5750_1_nw_220&" + \
                "APP=timonline"

                nome_imx = f"x_{xc+500}_y_{yc+500}.png"
                r = requests.get(url)
                with open(os.path.join("./fotos_descargadas",nome_imx), 'wb') as f:
                    f.write(r.content)
                pbar.update(1)
