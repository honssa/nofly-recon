import matplotlib
from matplotlib import pyplot as plt
import Config
import numpy as np
import os, random, shutil
from sklearn.metrics import cohen_kappa_score, confusion_matrix
matplotlib.use('TkAgg')


class Metricas:
    def __init__(self, clases):
        self.metricas = {"ACC": np.array([]), "TN": np.array([]), "FP": np.array([]),
                    "FN": np.array([]), "TP": np.array([]), "TPR": np.array([]), 
                    "TNR": np.array([]), "PPV": np.array([]), "F1": np.array([]),
                    "NPV": np.array([]), "FPR": np.array([]), "FNR": np.array([]), 
                    "FDR": np.array([]) } 
        self.resultados = [] # Metricas por cada clase
        self.kappas = np.array([])
        self.clases = clases
        for clase in self.clases:
            self.resultados.append(self.metricas.copy())
        self.y_orixinal = None
        self.y_predecida = None
        if os.path.exists("./resultados"):
            shutil.rmtree("./resultados")
        os.mkdir("resultados")


    def anotar_metricas(self, y_orixinal, y_predecida):
        self.y_orixinal = y_orixinal; self.y_predecida = y_predecida
        cnf_matrix = self.compute_confusion_matrix(y_orixinal, y_predecida)
        metrica = self.compute_total_classification_metrics(cnf_matrix)
        self.kappas = np.append(self.kappas, cohen_kappa_score(y_orixinal, y_predecida))
        for i,key in enumerate(self.metricas):
            for j,clase in enumerate(self.clases):
                self.resultados[j][key] = np.append(self.resultados[j][key], metrica[i][j])


    def calcular_media_dt(self):
        print(self.resultados)
        medias = []; dt = []
        for clase in self.clases:
            medias.append(np.array([])); dt.append(np.array([]));
        for metrica in self.metricas:
            for i,clase in enumerate(self.clases):
                medias[i] = np.append(medias[i], np.average(self.resultados[i][metrica]))
                dt[i] = np.append(dt[i], np.std(self.resultados[i][metrica]))
        
        f = open(os.path.join("./resultados","metricas"), "w")
        f.write("Metricas acadadas no adestramento (promedio de " + str(Config.NUM_ITERACIONS) + " iteracions)\n")
        for ic, clase in enumerate(self.clases): 
            f.write("\nClase (" + clase + "): \n\t\t\t media\t\t\t\tdesviacion tipica\n-------------------------------------------\n")
            for metrica in self.metricas:
                f.write("\t" + metrica + ":  " + str(medias[ic][list(self.metricas.keys()).index(metrica)]))
                f.write(",\t" + str(dt[ic][list(self.metricas.keys()).index(metrica)]) + "\n")
        
        f.write("\n\nKappa: media = " + str(np.average(self.kappas)) + ", desviacion tipica: " + str(np.std(self.kappas)))
        f.close()
        

    def compute_total_classification_metrics(self, cnf_matrix):
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float);FN = FN.astype(float);TP = TP.astype(float);TN = TN.astype(float)
        
        # Sensibilidade, "hit rate", "recall", ou "true positive rate"
        TPR = TP / (TP + FN)
        # Especificidade ou "true negative rate"
        TNR = TN / (TN + FP)
        # Precision ou "positive predictive value"
        PPV = TP / (TP + FP)         
        # F1
        F1 = 2 * (PPV * TPR) / (PPV + TPR)
        # "Negative predictive value"
        NPV = TN / (TN + FN)
        # "Fall out" ou "false positive rate"
        FPR = FP / (FP + TN)
        # "False negative rate"
        FNR = FN / (TP + FN)
        # "False discovery rate"
        FDR = FP / (TP + FP)
        # Precision promedia
        ACC = (TP + TN) / (TP + FP + FN + TN)
        return [ACC, TN, FP, FN, TP, TPR, TNR, PPV, F1, NPV, FPR, FNR, FDR]


    def compute_confusion_matrix(self, test_orig, test_predicted):
        print("test_orig: " + str(test_orig) + "   //   " + "test_predicted" + str(test_predicted))
        num_classes = len(np.unique(test_orig))
        matrix = np.zeros((num_classes,num_classes), int)
    
        for t1, t2 in zip(test_orig,test_predicted):
            matrix[t1,t2] += 1
        print(matrix)
        return matrix


    def inicializar_semente(self):
        if Config.SEMENTE == None:
            Config.SEMENTE = np.random.randint(1, 255)
        os.environ['PYTHONHASHSEED'] = str(Config.SEMENTE)
        random.seed(Config.SEMENTE)
        np.random.seed(Config.SEMENTE)
        import tensorflow as tf
        if tf.__version__ < '2.0.0':
            tf.set_random_seed(Config.SEMENTE)
        else:
            import tensorflow.compat.v1 as tf
            tf.set_random_seed(Config.SEMENTE)
        from tensorflow.python.keras import backend as K
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
        with open(os.path.join(Config.BASE_ROUTE,"session_seed.txt"), 'w') as seed_file:
            seed_file.write(str(Config.SEMENTE) + '\n')
            seed_file.close()


    def gardar_cm(self, test_orig, test_predicted):
    #def gardar_cm(self, cm):
        plt.figure(figsize=(10,6));
        import seaborn as sns
        cm = confusion_matrix(test_orig, test_predicted)
        #print(cm)
        graf = sns.heatmap(cm, annot=True, cmap="Blues", fmt='g')
        graf.set_title("Matriz de confusion\n")
        graf.set_xlabel("\nValores predecidos")
        graf.set_ylabel("\nValores actuais")
        graf.xaxis.set_ticklabels(self.clases)
        graf.yaxis.set_ticklabels(self.clases)

        plt.savefig(os.path.join("./resultados", "matriz_cm.png"))
        #plt.show()

    def grafica_adestramento(self, progresion):
        loss = progresion.history["loss"]
        val_loss = progresion.history["val_loss"]
        epochs = range(1, len(loss) + 1)
        acc = progresion.history['accuracy']
        val_acc = progresion.history['val_accuracy']

        plt.figure(figsize=(15,5));
        plot1 = plt.subplot2grid((1,2), (0,0), colspan=1)
        plot2 = plt.subplot2grid((1,2), (0,1), colspan=1)
        plot1.plot(epochs, loss, 'y', label='loss adestramento')
        plot1.plot(epochs, val_loss, 'r', label='loss validacion')
        plot1.set_xlabel('Epochs'); plot1.set_ylabel('Loss')
        plot1.legend()
        
        plot2.plot(epochs, acc, 'y', label='Precision adestramento')
        plot2.plot(epochs, val_acc, 'r', label='Precision validacion')
        plot2.set_xlabel('Epochs'); plot2.set_ylabel('Precision')
        plot2.legend()
        plt.savefig(os.path.join("./resultados", "graf.png"))
        plt.clf()
        self.gardar_cm(self.y_orixinal, self.y_predecida) 
