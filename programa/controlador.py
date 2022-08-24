from modeloApp import ModeloApp

class Controlador():
    def __init__(self):
        self.modelo = ModeloApp()
        #vista = ApMenu(self)

    def xenerarPredCoord(self, data):
        lat, lon, id_modelo = data
        imx = self.modelo.getImx(float(lat), float(lon))
        predicion = self.modelo.predecir(imx, id_modelo)
        return predicion

    def xenerarPredImx(self, data):
        imx, id_modelo = data
        predicion = self.modelo.predecir(imx, id_modelo)
        return predicion
        
    


#if __name__ == "__main__":
#    modeloSeg = ModeloSeg()
#    modeloClas = ModeloClas()
#    coord2Imx = Coord2Imx()
#    c = Controlador(modeloSeg, modeloClas, coord2Imx)
