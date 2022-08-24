from modelo import Modelo
import tensorflow as tf

class ModeloResNet50(Modelo):
    def __init__(self):
        modelo_dir = "./conv_simple.h5"
        self.modelo = tf.keras.models.load_model(modelo_dir, compile=False)

    def predecir(self, batch):
        return self.modelo.predict(batch)
