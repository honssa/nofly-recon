from modelo import Modelo
import tensorflow as tf

class ModeloUnet(Modelo):
    def __init__(self):
        modelo1_dir = "./seg_edificios_minLOSS.h5"
        modelo2_dir = "./seg_estradas_minLOSS.h5"
        self.modelo1 = tf.keras.models.load_model(modelo1_dir, compile=False)
        self.modelo2 = tf.keras.models.load_model(modelo2_dir, compile=False)

    def predecir(self, batch):
        return (self.modelo1.predict(batch, batch_size=1), self.modelo2.predict(batch, batch_size=1))
