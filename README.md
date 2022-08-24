# Reconecemento automatico de zonas prohibidas para o voo de UAV

Sistema intelixente para reconecer e clasificar areas como zonas 
residenciais ou estradas en fotografias aereas de alta resolucion.


## Instalacion

Para instalar esta aplicacion hai que descargar o repositorio:

$ git clone https://gitlab.com/t7276/nofly-recon.git

Dependencias: 
* [numpy](https://numpy.org)
* [Tensorflow](https://tensorflow.org)
* [Matplotlib](https://matplotlib.org)
* [opencv](https://opencv.org)
* [seaborn](https://seaborn.pydata.org)
* [pyosmium](https://osmcode.org/pyosmium)
* [seaborn](https://seaborn.pydata.org)
* [pyproj](https://pyproj4.github.io/pyproj/stable/)

Para descargar unha base de datos, usar o script "descargar.py" ou descargar o seguinte directorio que conten
unha base de datos de proba, que foi usada no desenvolvemento da ferramenta: 
https://udcgal.sharepoint.com/:f:/s/TFG7/EuBySTaSk8tIjPGf40ar\_7EBWdYGuE3Dv-LddzHymnkQ3Q?e=uucjdm

Esta base de datos, conten imaxes dunha area de 10x10 km da zona de Moers, Alemania.


## Uso
Unha vez descargado, deberanse seguir os seguintes pasos:

1: $python descargar.py coord\_x1 coord\_y1 coord\_x2 coord\_y2

Esto descargara fotos de alta resolucion 10000x10000px de https://www.wcs.nrw.de
estas fotografias cubriran un espazo de 1km cadrado. O numero de fotos que se descargara
dependera da dimension introducida nos argumentos do script, que tera a seguinte forma:

        +---------(x2,y2)
        |            |
        |            |
        |            |
    (x1,y1)----------+

2: $python crear\_ds <directorio_fotos> <ficheiro_osm>

Neste paso, crearase un conxunto de datos preparado para o adestramento nunha rede neuronal,
o ficheiro crear\_ds.py que se atopa no directorio de "segmentacion" creara un dataset coa 
seguinte estructura:

     - DS
    |
    |- Imaxes
    |
    |- Mascaras

As mascaras seran "ground truths" das fotografias aereas, en cambio no ficheiro crear\_ds.py do
directorio de clasificacion, crearase da seguinte forma:
    
    -DS
    |
    |- Edificios
    |
    |- Estradas
    |
    |- Libre

Cada carpeta definira unha clase na que se atopan imaxes coa maioria de pixels pertencentes
a esa clase.

3: $python adestrar\_clasificacion.py ou $python adestrar\_segmentacion.py

Neste paso adestraranse os modelos de clasificacion ou segmentacion en base a configuracion 
definida en Config.py 

4: $python programa/run.py

Esto e unha ferramenta cunha interfaz grafica
para poder visualizar os resultados dos modelos adestrados, en base a unha imaxe cargada
ou unha coordenada introducida.

## Autor
* Gonzalo Xoel Otero Gonzalez (Estudante)
 
## Titores:

* [Enrique Fernandez-Blanco](https://orcid.org/0000-0003-3260-8734)
* [Alejandro Puente-Castro](https://orcid.org/0000-0002-0134-6877)
