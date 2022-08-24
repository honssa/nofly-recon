from abc import ABC, abstractmethod

class Modelo(ABC):
    @abstractmethod
    def predecir(self, imx):
        pass
