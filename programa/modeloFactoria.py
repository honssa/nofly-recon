from  modelo import Modelo
from modeloConvolucional import ModeloConvolucional
from modeloResNet50 import ModeloResNet50
from modeloUnet import ModeloUnet


class ModeloFactoria:
    def __init__(self):
        self.modelo = None

    def getModelo(self, id_modelo):
        if id_modelo == 0:
            return ModeloConvolucional()
        elif id_modelo == 1:
            return ModeloResNet50()
        elif id_modelo == 2:
            return ModeloUnet()
        else:
            return None

