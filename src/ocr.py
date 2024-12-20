import math #https://docs.python.org/3/library/math.html
import os #https://docs.python.org/3/library/os.html
import time #https://docs.python.org/3/library/time.html
import datetime #https://docs.python.org/3/library/datetime.html

import cadenas

def crea_lista_1d(d1, valor=0):
    lista = [valor] * d1
    return lista

def crea_lista_2d(d1, d2, valor=0):
    lista = crea_lista_1d(d1)
    for i in range(d1):
        lista[i] = crea_lista_1d(d2, valor)
    return lista

def crea_lista_3d(d1, d2, d3, valor=0):
    lista = crea_lista_1d(d1)
    for i in range(d1):
        lista[i] = crea_lista_2d(d2, d3, valor)
    return lista

import cv2 #https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
import easyocr
import numpy as np #https://numpy.org/doc/stable/user/quickstart.html
import matplotlib.pyplot as plt #https://matplotlib.org/stable/users/explain/quick_start.html
import pytesseract
import scipy #https://docs.scipy.org/doc/scipy-1.13.1/tutorial/index.html#user-guide
import sklearn.model_selection as skl_ms #https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
import torch #https://pytorch.org/tutorials/beginner/basics/intro.html

C_EXPERIMENTOS = 0

if torch.cuda.is_available(): #https://pytorch.org/docs/stable/generated/torch.cuda.is_available.html#torch.cuda.is_available
    DISPOSITIVO = torch.device("cuda") #https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
elif torch.backends.mps.is_available(): #https://pytorch.org/docs/stable/backends.html#torch.backends.mps.is_available
    DISPOSITIVO = torch.device("mps") #https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
else:
    DISPOSITIVO = torch.device("cpu") #https://pytorch.org/docs/stable/tensor_attributes.html#torch.device

#https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html#define-the-class
class ClasificadorLetras(torch.nn.Module): #https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    def __init__(self):
        super().__init__()

        #https://www.ibm.com/es-es/topics/convolutional-neural-networks#%C2%BFC%C3%B3mo+funcionan+las+redes+neuronales+convolucionales%3F
        self.N_ENTRADA = 1 #imagen (1 x 28 x 28)

        self.N_CARACTERISTICAS_1 = 32 #32 filtros (1 x 3 x 3)
        self.N_FILTRO_1 = (3, 3)
        self.T_BORDE_1 = "same"
        self.capa1 = torch.nn.Conv2d(self.N_ENTRADA, self.N_CARACTERISTICAS_1, self.N_FILTRO_1, padding=self.T_BORDE_1) #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        self.activacion1 = torch.nn.ReLU() #https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
        self.N_SUBMUESTREO_1 = (2, 2) #filtro de submuestreo (32 x 2 x 2)
        self.agrupacion1 = torch.nn.MaxPool2d(self.N_SUBMUESTREO_1) #https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d

        self.N_CARACTERISTICAS_2 = 64 #64 filtros (32 x 3 x 3)
        self.N_FILTRO_2 = (3, 3)
        self.T_BORDE_2 = "same"
        self.capa2 = torch.nn.Conv2d(self.N_CARACTERISTICAS_1, self.N_CARACTERISTICAS_2, self.N_FILTRO_2, padding=self.T_BORDE_2) #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        self.activacion2 = torch.nn.ReLU() #https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
        self.N_SUBMUESTREO_2 = (2, 2) #filtro de submuestreo (64 x 2 x 2)
        self.agrupacion2 = torch.nn.MaxPool2d(self.N_SUBMUESTREO_2) #https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d

        self.N_CARACTERISTICAS_3 = 128 #128 filtros (64 x 3 x 3)
        self.N_FILTRO_3 = (3, 3)
        self.T_BORDE_3 = "same"
        self.capa3 = torch.nn.Conv2d(self.N_CARACTERISTICAS_2, self.N_CARACTERISTICAS_3, self.N_FILTRO_3, padding=self.T_BORDE_3) #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        self.activacion3 = torch.nn.ReLU() #https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
        self.N_SUBMUESTREO_3 = (2, 2) #filtro de submuestreo (128 x 2 x 2)
        self.agrupacion3 = torch.nn.MaxPool2d(self.N_SUBMUESTREO_3) #https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d

        #capa con entrada (128 x 3 x 3) y salida (1152)
        self.transicion = torch.nn.Flatten() #https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html#torch.nn.Flatten

        self.N_CAPA_4 = 64 #capa con entrada (1152) y salida (64)
        self.capa4 = torch.nn.Linear(self.N_CARACTERISTICAS_3 * 3 * 3, self.N_CAPA_4) #https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        self.activacion4 = torch.nn.ReLU() #https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU

        self.N_CAPA_5 = 128 #capa con entrada (64) y salida (128)
        self.capa5 = torch.nn.Linear(self.N_CAPA_4, self.N_CAPA_5) #https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        self.activacion5 = torch.nn.ReLU() #https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU

        self.N_CAPA_6 = 36 #capa con entrada (128) y salida (36)
        self.capa6 = torch.nn.Linear(self.N_CAPA_5, self.N_CAPA_6) #https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear

    def forward(self, entrada):
        #print(entrada.shape)
        #print(entrada)

        #x1 = x0
        x1 = entrada #(1 x 28 x 28)
        #h1 = x1 w1 + b1
        h1 = self.capa1(x1) #(32 x 28 x 28)
        #y1 = f1(h1)
        y1 = self.activacion1(h1) #(32 x 28 x 28)
        #y1 = p1(y1)
        y1 = self.agrupacion1(y1) #(32 x 14 x 14)

        #x2 = y1
        x2 = y1 #(32 x 14 x 14)
        #h2 = x2 w2 + b2
        h2 = self.capa2(x2) #(64 x 14 x 14)
        #y2 = f2(h2)
        y2 = self.activacion2(h2) #(64 x 14 x 14)
        #y2 = p2(y2)
        y2 = self.agrupacion2(y2) #(64 x 7 x 7)

        #x3 = y2
        x3 = y2 #(64 x 7 x 7)
        #h3 = x3 w3 + b3
        h3 = self.capa3(x3) #(128 x 7 x 7)
        #y3 = f3(h3)
        y3 = self.activacion3(h3) #(128 x 7 x 7)
        #y3 = p3(y3)
        y3 = self.agrupacion3(y3) #(128 x 3 x 3)

        #x4 = t(y3)
        x4 = self.transicion(y3) #(1152)
        #h4 = x4 w4 + b4
        h4 = self.capa4(x4) #(64)
        #y4 = f4(h4)
        y4 = self.activacion4(h4) #(64)
        #x5 = y4
        x5 = y4 #(64)

        #h1 = x1 w1 + b1
        h5 = self.capa5(x5) #(128)
        #y1 = f1(h1)
        y5 = self.activacion5(h5) #(128)
        #x5 = y4
        x6 = y5 #(128)

        #y6 = x6 w6 + b6
        y6 = self.capa6(x6) #(36)

        return y6
    
def muestra_imagen(imagen, titulo="", nFilas=1, nColumnas=1, iFigura=1, nuevaFigura=False):
    if nuevaFigura:
        plt.figure(figsize=(12, 8))
    plt.subplot(nFilas, nColumnas, iFigura)
    plt.title(titulo)
    if imagen.ndim == 2: #https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html
        plt.imshow(imagen, cmap="gray", vmin=0, vmax=255) #https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    else:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB) #https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
        plt.imshow(imagen) #https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html

def preprocesa_imagen_distancia_color(imagen, exponente=2/3, multiplo=1, vecindario=11, constante=2):
    imagen = imagen.tolist()
    nFilas = len(imagen)
    nColumnas = len(imagen[0])

    for i in range(nFilas):
        filaImagen = imagen[i]
        for j in range(nColumnas):
            pixel = filaImagen[j]
            pixel = pixel[:3]

            distanciaANegro = math.hypot(*pixel)
            maxDistancia = math.hypot(255, 255, 255)

            distanciaANegro **= exponente
            maxDistancia **= exponente

            pixel = 255 * multiplo * distanciaANegro / maxDistancia
            pixel = min(255, pixel)

            filaImagen[j] = pixel

    imagen = np.array(imagen, np.uint8)

    imagen = cv2.adaptiveThreshold(imagen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, vecindario, constante)

    return imagen

def preprocesa_imagen_transformacion_color(imagen, vecindario=11, constante=9):
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, vecindario, constante) #https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3
    invertion = 255 - adaptive
    dilation = cv2.dilate(invertion, np.ones((3, 3))) #https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c
    return dilation

def preprocesa_imagen_segmentacion_red_neuronal(imagen, nFilasEntrada=32*10, nColumnasEntrada=32*10, umbralDeteccion=0.7, umbralInterseccion=0.4, umbralLineas=0.5, umbralDeteccionLetras=0.4, umbralParticion=0.2):
    #https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray
    nFilas = imagen.shape[0] #https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html
    nColumnas = imagen.shape[1] #https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html
    imagenEntrada = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB) #https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    
    anchoEntrada = nColumnasEntrada
    altoEntrada = nFilasEntrada
    imagenEntrada = cv2.dnn.blobFromImage(imagenEntrada, 1.0, (anchoEntrada, altoEntrada), (123.68, 116.78, 103.94), True, False) #https://docs.opencv.org/3.4/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7

    detector = cv2.dnn.readNet("frozen_east_text_detection.pb") #https://docs.opencv.org/4.x/d6/d0f/group__dnn.html#ga138439da76f26266fdefec9723f6c5cd
    detector.setInput(imagenEntrada) #https://docs.opencv.org/4.x/db/d30/classcv_1_1dnn_1_1Net.html#a5e74adacffd6aa53d56046581de7fcbd
    capasResultados = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    probabilidades, coordenadas = detector.forward(capasResultados) #https://docs.opencv.org/4.x/db/d30/classcv_1_1dnn_1_1Net.html#a98ed94cb6ef7063d3697259566da310b

    #Convierte a coordenadas (centro, ancho, alto, angulo)
    nFilasSalida = probabilidades.shape[2]
    nColumnasSalida = probabilidades.shape[3]
    nueProbabilidades = []
    nueCoordenadas = []

    for i in range(nFilasSalida):
        filaProbabilidades = probabilidades[0, 0, i] #https://numpy.org/doc/stable/user/basics.indexing.html#single-element-indexing
        filaCoordenadasD1 = coordenadas[0, 0, i] #https://numpy.org/doc/stable/user/basics.indexing.html#single-element-indexing
        filaCoordenadasD2 = coordenadas[0, 1, i] #https://numpy.org/doc/stable/user/basics.indexing.html#single-element-indexing
        filaCoordenadasD3 = coordenadas[0, 2, i] #https://numpy.org/doc/stable/user/basics.indexing.html#single-element-indexing
        filaCoordenadasD4 = coordenadas[0, 3, i] #https://numpy.org/doc/stable/user/basics.indexing.html#single-element-indexing
        filaAngulos = coordenadas[0, 4, i] #https://numpy.org/doc/stable/user/basics.indexing.html#single-element-indexing

        for j in range(nColumnasSalida):
            probabilidad = filaProbabilidades[j]

            if probabilidad >= umbralDeteccion:
                d1 = filaCoordenadasD1[j]
                d2 = filaCoordenadasD2[j]
                d3 = filaCoordenadasD3[j]
                d4 = filaCoordenadasD4[j]
                angulo = filaAngulos[j]

                cosAngulo = math.cos(angulo)
                senAngulo = math.sin(angulo)

                alto = d1 + d3
                ancho = d2 + d4

                p2X = j * 4 + cosAngulo * d2 + senAngulo * d3
                p2Y = i * 4 - senAngulo * d2 + cosAngulo * d3

                p1X = p2X + -senAngulo * alto
                p1Y = p2Y + -cosAngulo * alto
                p3X = p2X + -cosAngulo * ancho
                p3Y = p2Y + senAngulo * ancho
                centro = (0.5 * (p1X + p3X), 0.5 * (p1Y + p3Y))

                nueProbabilidades.append(probabilidad)
                nueCoordenadas.append((centro, (ancho, alto), -math.degrees(angulo)))

    probabilidades = nueProbabilidades
    coordenadas = nueCoordenadas

    #Elimina la superposicion de recuadros
    indicesSeleccionados = cv2.dnn.NMSBoxesRotated(coordenadas, probabilidades, umbralDeteccion, umbralInterseccion) #https://docs.opencv.org/4.x/d6/d0f/group__dnn.html#ga9d118d70a1659af729d01b10233213ee

    #Obtiene las coordenadas (minI, maxI, minJ, maxJ) y los eventos de inicio y fin de palabra
    rFilasZFilasEntrada = nFilas / nFilasEntrada
    rColumnasZColumnasEntrada = nColumnas / nColumnasEntrada
    nueCoordenadas = []
    eventosPalabras = []

    for indice in indicesSeleccionados:
        coordenadasPalabra = coordenadas[indice]
        puntos = cv2.boxPoints(coordenadasPalabra) #https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gaf78d467e024b4d7936cf9397185d2f5c

        minCoordenadas = np.min(puntos, axis=0) #https://numpy.org/doc/stable//reference/generated/numpy.min.html
        maxCoordenadas = np.max(puntos, axis=0) #https://numpy.org/doc/stable//reference/generated/numpy.max.html

        minJ = minCoordenadas[0]
        minJ *= rColumnasZColumnasEntrada
        minJ = int(minJ)

        minI = minCoordenadas[1]
        minI *= rFilasZFilasEntrada
        minI = int(minI)

        maxJ = maxCoordenadas[0]
        maxJ *= rColumnasZColumnasEntrada
        maxJ = int(maxJ)

        maxI = maxCoordenadas[1]
        maxI *= rFilasZFilasEntrada
        maxI = int(maxI)

        recuadro = (minI, maxI, minJ, maxJ)
        nueCoordenadas.append(recuadro)

        alto = maxI - minI + 1
        iniRecuadro = (minI, 1, alto)
        finRecuadro = (maxI + 1, -1, -alto)
        eventosPalabras.append(iniRecuadro)
        eventosPalabras.append(finRecuadro)

    coordenadas = nueCoordenadas
    eventosPalabras.sort()

    #Procesa los eventos de inicio y fin para asignar prioridad a las posibles lineas de texto
    lineasTexto = []
    acNPalabras = 0
    acAlturas = 0
    filaAnt = -1

    for evento in eventosPalabras:
        (fila, dltNPalabras, dltAlturas) = evento

        if fila != filaAnt:
            nInterseccion = fila - filaAnt
            if acNPalabras != 0:
                prmAlturas = acAlturas / acNPalabras
            else:
                prmAlturas = 0
            esInterseccionMenor = (nInterseccion < umbralLineas * prmAlturas)

            linea = (esInterseccionMenor, -acNPalabras, filaAnt)
            lineasTexto.append(linea)
            
            filaAnt = fila

        acNPalabras += dltNPalabras
        acAlturas += dltAlturas

    nInterseccion = 1
    if acNPalabras != 0:
        prmAlturas = acAlturas / acNPalabras
    else:
        prmAlturas = 0
    esInterseccionMenor = (nInterseccion < umbralLineas * prmAlturas)

    linea = (esInterseccionMenor, -acNPalabras, filaAnt)
    lineasTexto.append(linea)
    lineasTexto.sort()

    #Asigna una linea a las palabras y las ordena
    nueCoordenadas = []

    for recuadro in coordenadas:
        (minI, maxI, minJ, maxJ) = recuadro

        lineaTexto = -1
        j = 0

        while lineaTexto == -1 and j < len(lineasTexto):
            lineaPotencial = lineasTexto[j]
            (esInterseccionMenor, acNPalabras, linea) = lineaPotencial

            if linea >= minI and linea <= maxI:
                lineaTexto = linea

            j += 1

        recuadro = (lineaTexto, minJ, maxJ, minI, maxI)
        nueCoordenadas.append(recuadro)

    coordenadas = nueCoordenadas
    coordenadas.sort()

    #Procesa las imagenes de las palabras
    imagenes = []
    for coordenadasPalabra in coordenadas:
        (lineaTexto, minJ, maxJ, minI, maxI) = coordenadasPalabra
        imagenPalabra = imagen[minI : maxI + 1, minJ : maxJ + 1]

        #https://docs.opencv.org/4.x/df/d0d/tutorial_find_contours.html
        #https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
        #FONDO NEGRO TEMPORAL
        contornos, jerarquias = cv2.findContours(255 - imagenPalabra, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
        if type(jerarquias) != type(None):
            jerarquias = jerarquias[0]
        else:
            jerarquias = tuple()

        #Convierte los contornos a recuadros, elimina los contornos internos y los ordena
        nueContornos = []

        for contorno, jerarquia in zip(contornos, jerarquias):
            contorno = np.reshape(contorno, (-1, 2))
            minCoordenadas = np.min(contorno, axis=0) #https://numpy.org/doc/stable//reference/generated/numpy.min.html
            maxCoordenadas = np.max(contorno, axis=0) #https://numpy.org/doc/stable//reference/generated/numpy.max.html

            minJ = int(minCoordenadas[0])
            minI = int(minCoordenadas[1])

            maxJ = int(maxCoordenadas[0])
            maxI = int(maxCoordenadas[1])

            recuadro = (minJ, maxJ, minI, maxI)
            if jerarquia[-1] < 0: #si no es un contorno interior
                nueContornos.append(recuadro)

        contornos = nueContornos
        contornos.sort()

        #Obtiene las imagenes de las letras
        for recuadro in contornos:
            (minJ, maxJ, minI, maxI) = recuadro
            
            altoImagen = imagenPalabra.shape[0]
            altoLetra = maxI - minI + 1

            if altoLetra > umbralDeteccionLetras * altoImagen:
                imagenLetra = imagenPalabra[minI : maxI + 1, minJ : maxJ + 1]
                
                anchoLetra = maxJ - minJ + 1
                
                if anchoLetra > (1 + umbralParticion) * altoLetra:
                    
                    anchoPalabra = anchoLetra
                    nLetras = math.ceil(anchoPalabra / altoLetra) #https://docs.python.org/3/library/math.html#math.ceil
                    anchoLetras = math.ceil(anchoPalabra / nLetras) #https://docs.python.org/3/library/math.html#math.ceil
                    iniLetra = minJ
                    finLetra = iniLetra + anchoLetras

                    for i in range(nLetras):
                        imagenLetra = imagenPalabra[minI : maxI + 1, iniLetra : finLetra + 1]
                        if type(imagenLetra) != type(None):
                            altoLetra = imagenLetra.shape[0]
                            anchoLetra = imagenLetra.shape[1]

                            if altoLetra > anchoLetra:
                                N_BORDE = math.ceil(0.1 * altoLetra)
                                bordeSup = N_BORDE
                                bordeInf = N_BORDE
                                bordeIzq = (bordeSup + altoLetra + bordeInf - anchoLetra) // 2
                                bordeDer = (bordeSup + altoLetra + bordeInf) - (bordeIzq + anchoLetra)
                            else:
                                N_BORDE = math.ceil(0.1 * anchoLetra)
                                bordeIzq = N_BORDE
                                bordeDer = N_BORDE
                                bordeSup = (bordeIzq + anchoLetra + bordeDer - altoLetra) // 2
                                bordeInf = (bordeIzq + anchoLetra + bordeDer) - (bordeSup + altoLetra)
                                
                            imagenLetra = cv2.copyMakeBorder(imagenLetra, bordeSup, bordeInf, bordeIzq, bordeDer, cv2.BORDER_CONSTANT, None, (255, 255, 255)) #https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga2ac1049c2c3dd25c2b41bffe17658a36
                            r28ZAlto = 28 / (bordeSup + altoLetra + bordeInf)
                            r28ZAncho = 28 / (bordeIzq + anchoLetra + bordeDer)
                            imagenLetra = cv2.resize(imagenLetra, None, None, r28ZAncho, r28ZAlto) #https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
                            imagenes.append(imagenLetra)

                            iniLetra = finLetra + 1
                            finLetra = iniLetra + anchoLetras
                            finLetra = min(finLetra, maxJ)
                
                else:
                    if altoLetra > anchoLetra:
                        N_BORDE = math.ceil(0.1 * altoLetra)
                        bordeSup = N_BORDE
                        bordeInf = N_BORDE
                        bordeIzq = (bordeSup + altoLetra + bordeInf - anchoLetra) // 2
                        bordeDer = (bordeSup + altoLetra + bordeInf) - (bordeIzq + anchoLetra)
                    else:
                        N_BORDE = math.ceil(0.1 * anchoLetra)
                        bordeIzq = N_BORDE
                        bordeDer = N_BORDE
                        bordeSup = (bordeIzq + anchoLetra + bordeDer - altoLetra) // 2
                        bordeInf = (bordeIzq + anchoLetra + bordeDer) - (bordeSup + altoLetra)

                    imagenLetra = cv2.copyMakeBorder(imagenLetra, bordeSup, bordeInf, bordeIzq, bordeDer, cv2.BORDER_CONSTANT, None, (255, 255, 255)) #https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga2ac1049c2c3dd25c2b41bffe17658a36
                    r28ZAlto = 28 / (bordeSup + altoLetra + bordeInf)
                    r28ZAncho = 28 / (bordeIzq + anchoLetra + bordeDer)
                    imagenLetra = cv2.resize(imagenLetra, None, None, r28ZAncho, r28ZAlto) #https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
                    imagenes.append(imagenLetra)

    return imagenes

def reconoce_caracteres_clasificador(imagenes):
    LETRAS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") #https://docs.python.org/3/library/stdtypes.html#list
    modelo = torch.load("clasificador.pt") #https://pytorch.org/docs/stable/generated/torch.load.html#torch.load
    modelo.to(DISPOSITIVO)

    with torch.inference_mode():
        modelo.eval() #https://pytorch.org/docs/2.3/generated/torch.nn.Module.html#torch.nn.Module.eval

        predicciones = []
        for imagen in imagenes:

            if imagen.ndim > 2: #https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) #https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
            imagen = imagen[:28, :28] #https://numpy.org/doc/stable/user/basics.indexing.html#slicing-and-striding
            bordeSup = 0
            bordeInf = 28 - imagen.shape[0] #https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html#numpy.ndarray.shape
            bordeIzq = 0
            bordeDer = 28 - imagen.shape[1] #https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html#numpy.ndarray.shape
            imagen = cv2.copyMakeBorder(imagen, bordeSup, bordeInf, bordeIzq, bordeDer, cv2.BORDER_CONSTANT, None, (255, 255, 255)) #https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga2ac1049c2c3dd25c2b41bffe17658a36
            
            imagen = np.reshape(imagen, (1, 1, 28, 28)) #https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
            imagen = torch.from_numpy(imagen) #https://pytorch.org/docs/stable/generated/torch.from_numpy.html
            imagen = imagen.float() #https://pytorch.org/docs/stable/generated/torch.Tensor.float.html
            imagen = imagen.to(DISPOSITIVO)

            prediccion = modelo(imagen)
            prediccion = torch.argmax(prediccion, dim=1) #https://pytorch.org/docs/stable/generated/torch.argmax.html
            predicciones += prediccion.tolist() #https://pytorch.org/docs/stable/generated/torch.Tensor.tolist.html

    print(predicciones)
    for i, prediccion in enumerate(predicciones):
        predicciones[i] = LETRAS[prediccion]

    texto = "".join(predicciones)
    return texto

def reconoce_caracteres_pytesseract(imagen):
    texto = pytesseract.image_to_string(imagen)
    texto = texto.replace("\n", " ")
    return texto

def reconoce_caracteres_easyocr(imagen): 
    lector = easyocr.Reader(["es"], True)
    texto = lector.readtext(imagen, detail=0)
    texto = " ".join(texto)
    return texto

def muestra_experimento(datos, funcionPreprocesamiento, funcionOcr, rDatos=""):
    global C_EXPERIMENTOS
    C_EXPERIMENTOS += 1

    print()
    print("Experimento:", C_EXPERIMENTOS)
    print("Preprocesamiento:", funcionPreprocesamiento)
    print("OCR:", funcionOcr)

    nImagenes = len(datos)
    tiempos = crea_lista_1d(nImagenes)
    longitudes = crea_lista_1d(nImagenes)

    for i, archivo in enumerate(datos):
        imagen = cv2.imread(f"{rDatos}/{archivo}", cv2.IMREAD_UNCHANGED) #https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gab32ee19e22660912565f8140d0f675a8

        t1 = time.time() #https://docs.python.org/3/library/time.html#time.time
        if funcionPreprocesamiento != None:
            imagen = funcionPreprocesamiento(imagen)
        texto = funcionOcr(imagen)
        t2 = time.time() #https://docs.python.org/3/library/time.html#time.time
        tiempos[i] = t2 - t1
        longitudes[i] = len(texto)

        print(f'[{archivo}] "{texto}"')

    tiempoPmd = sum(tiempos) / nImagenes #https://docs.python.org/3/library/functions.html#sum
    longitudPmd = sum(longitudes) / nImagenes #https://docs.python.org/3/library/functions.html#sum

    print(f"Tiempo promedio: {tiempoPmd} s")
    print(f"Longitud promedio: {longitudPmd}")
    print()

def pruebas_preprocesamiento_ocr():
    R_IMAGENES = "credenciales"
    imagenes = []
    for archivo in os.listdir(R_IMAGENES):
        if archivo.endswith(".jpg"):
            imagenes.append(archivo)

    nImagenes = len(imagenes)
    tpCredenciales = crea_lista_1d(nImagenes)
    for i in range(nImagenes):
        tpCredenciales[i] = imagenes[i][1]
    imagenesEnt, imagenesPrb = skl_ms.train_test_split(imagenes, test_size=0.2, random_state=0, stratify=tpCredenciales)

    print()
    print("Imagenes:", nImagenes)
    print("Imagenes entrenamiento:", len(imagenesEnt))
    print("Imagenes prueba:", len(imagenesPrb))
    print(imagenesPrb)
    print()

    muestra_experimento(imagenesPrb, None, reconoce_caracteres_pytesseract, R_IMAGENES)
    muestra_experimento(imagenesPrb, None, reconoce_caracteres_easyocr, R_IMAGENES)
    muestra_experimento(imagenesPrb, preprocesa_imagen_distancia_color, reconoce_caracteres_pytesseract, R_IMAGENES)
    muestra_experimento(imagenesPrb, preprocesa_imagen_distancia_color, reconoce_caracteres_easyocr, R_IMAGENES)
    muestra_experimento(imagenesPrb, preprocesa_imagen_transformacion_color, reconoce_caracteres_pytesseract, R_IMAGENES)
    muestra_experimento(imagenesPrb, preprocesa_imagen_transformacion_color, reconoce_caracteres_easyocr, R_IMAGENES)

def pruebas_ocr_clasificador():
    archivos = os.listdir()
    imagenes = []
    for archivo in archivos:
        if archivo.endswith(".jpg"):
            imagenes.append(archivo)

    for archivo in imagenes:
        imagen = cv2.imread(archivo, cv2.IMREAD_UNCHANGED)
        imagenPreprocesada = preprocesa_imagen_distancia_color(imagen)
        imagenes = preprocesa_imagen_segmentacion_red_neuronal(imagenPreprocesada)

        print()
        print(len(imagenes))
        texto = reconoce_caracteres_clasificador(imagenes)
        print(f'"{texto}"')
        print()

        muestra_imagen(imagen, "Original", 2, 2, 1, True)
        muestra_imagen(imagenPreprocesada, "Preprocesada", 2, 2, 2)

        if len(texto) > 100:
            for i1, i2 in zip(range(40), range(60, 60 + 40)):
                muestra_imagen(imagenes[i2], f"'{texto[i2]}'", 8, 10, 40 + i1 + 1)

        for i in range(3):
            imagen = cv2.rotate(imagen, cv2.ROTATE_90_CLOCKWISE)
            imagenPreprocesada = preprocesa_imagen_distancia_color(imagen)
            imagenes = preprocesa_imagen_segmentacion_red_neuronal(imagenPreprocesada)

            print()
            print(len(imagenes))
            texto = reconoce_caracteres_clasificador(imagenes)
            print(f'"{texto}"')
            print()

            muestra_imagen(imagen, "Original", 2, 2, 1, True)
            muestra_imagen(imagenPreprocesada, "Preprocesada", 2, 2, 2)

            if len(texto) > 100:
                for i1, i2 in zip(range(40), range(60, 60 + 40)):
                    muestra_imagen(imagenes[i2], f"'{texto[i2]}'", 8, 10, 40 + i1 + 1)

    plt.show()

def mide_resultados(datos, funcionOcr, preprocesamientos=[], rDatos="", nueEspacio="_"):
    print()
    print(f"Preprocesamientos: {preprocesamientos}")
    print(f"Ocr: {funcionOcr}")
    print("...")
    
    nDatos = len(datos)
    
    tiempos = crea_lista_1d(nDatos)
    longitudes = crea_lista_1d(nDatos)
    exactitudes = crea_lista_1d(nDatos)

    for i, archivo in enumerate(datos):
        imagen = cv2.imread(f"{rDatos}/{archivo}", cv2.IMREAD_UNCHANGED) #https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gab32ee19e22660912565f8140d0f675a8

        [nombre, extension] = archivo.rsplit(".", 1) #https://docs.python.org/3/library/stdtypes.html#str.rsplit
        with open(f"{rDatos}/{nombre}.txt", encoding="utf8") as finput: #https://docs.python.org/3/library/functions.html#open
            elementos = finput.readlines() #https://docs.python.org/3/tutorial/inputoutput.html#methods-of-file-objects        

        t1 = time.time() #https://docs.python.org/3/library/time.html#time.time

        textos = []

        for j in range(4):
            resultadoPreprocesamiento = imagen
            for funcionPreprocesamiento in preprocesamientos:
                resultadoPreprocesamiento = funcionPreprocesamiento(resultadoPreprocesamiento)

            texto = funcionOcr(resultadoPreprocesamiento)
            textos.append(texto)

            imagen = cv2.rotate(imagen, cv2.ROTATE_90_CLOCKWISE)

        texto = nueEspacio.join(textos)

        exactitud = cadenas.calcula_exactitud_caracteres(elementos, texto, nueEspacio)

        t2 = time.time() #https://docs.python.org/3/library/time.html#time.time

        tiempos[i] = t2 - t1
        longitudes[i] = len(texto)
        exactitudes[i] = exactitud

    tiempoPmd = sum(tiempos) / nDatos #https://docs.python.org/3/library/functions.html#sum
    longitudPmd = sum(longitudes) / nDatos #https://docs.python.org/3/library/functions.html#sum
    exactitudPmd = sum(exactitudes) / nDatos #https://docs.python.org/3/library/functions.html#sum

    print(f"Tiempo promedio: {tiempoPmd} s")
    print(f"Longitud promedio: {longitudPmd}")
    print(f"Exactitud promedio: {exactitudPmd}")
    print()

    with open("experimentos.txt", "a", encoding="utf8") as fprint: #https://docs.python.org/3/library/functions.html#open
        fecha = datetime.datetime.now()

        fprint.write("\n" * 10) #https://docs.python.org/3/library/io.html#io.TextIOBase.write
        fprint.write(f"{fecha}\n") #https://docs.python.org/3/library/io.html#io.TextIOBase.write
        fprint.write("EXPERIMENTO\n") #https://docs.python.org/3/library/io.html#io.TextIOBase.write
        fprint.write(f"Datos ({len(datos)}): {rDatos} {datos}\n") #https://docs.python.org/3/library/io.html#io.TextIOBase.write
        fprint.write(f"Preprocesamientos ({len(preprocesamientos)}): {preprocesamientos}\n") #https://docs.python.org/3/library/io.html#io.TextIOBase.write
        fprint.write(f"Ocr: {funcionOcr}\n") #https://docs.python.org/3/library/io.html#io.TextIOBase.write
        fprint.write("RESULTADOS\n") #https://docs.python.org/3/library/io.html#io.TextIOBase.write
        fprint.write(f"Tiempo promedio: {tiempoPmd} s {tiempos}\n") #https://docs.python.org/3/library/io.html#io.TextIOBase.write
        fprint.write(f"Longitud promedio: {longitudPmd} {longitudes}\n") #https://docs.python.org/3/library/io.html#io.TextIOBase.write
        fprint.write(f"Exactitud promedio: {exactitudPmd} {exactitudes}\n") #https://docs.python.org/3/library/io.html#io.TextIOBase.write
        fprint.write("\n" * 10) #https://docs.python.org/3/library/io.html#io.TextIOBase.write


def experimenta_exactitud():
    R_IMAGENES = "credenciales"
    imagenes = []
    for archivo in os.listdir(R_IMAGENES):
        if archivo.endswith(".jpg"):
            imagenes.append(archivo)

    nImagenes = len(imagenes)
    tpCredenciales = crea_lista_1d(nImagenes)
    for i in range(nImagenes):
        tpCredenciales[i] = imagenes[i][1]
    imagenesEnt, imagenesPrb = skl_ms.train_test_split(imagenes, test_size=0.2, random_state=0, stratify=tpCredenciales)

    print()
    print("Imagenes:", nImagenes)
    print("Imagenes entrenamiento:", len(imagenesEnt))
    print("Imagenes prueba:", len(imagenesPrb))
    print()

    mide_resultados(imagenesEnt, reconoce_caracteres_easyocr, [], R_IMAGENES)
    #mide_resultados(imagenesEnt, reconoce_caracteres_pytesseract, [preprocesa_imagen_transformacion_color], R_IMAGENES)
    mide_resultados(imagenesEnt, reconoce_caracteres_clasificador, [preprocesa_imagen_distancia_color, preprocesa_imagen_segmentacion_red_neuronal], R_IMAGENES, nueEspacio="")

    mide_resultados(imagenesPrb, reconoce_caracteres_easyocr, [], R_IMAGENES)
    #mide_resultados(imagenesPrb, reconoce_caracteres_pytesseract, [preprocesa_imagen_transformacion_color], R_IMAGENES)
    mide_resultados(imagenesPrb, reconoce_caracteres_clasificador, [preprocesa_imagen_distancia_color, preprocesa_imagen_segmentacion_red_neuronal], R_IMAGENES, nueEspacio="")

def procesa_ejemplo():
    nombreArchivo = "credencial-sin-foto"
    nombreArchivo = "foto-animada"
    imagen = cv2.imread(f"{nombreArchivo}.png", cv2.IMREAD_UNCHANGED) #https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gab32ee19e22660912565f8140d0f675a8
    imagenPreprocesada = preprocesa_imagen_distancia_color(imagen)
    cv2.imwrite(f"{nombreArchivo}-distancia-color.png", imagenPreprocesada) #https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga8ac397bd09e48851665edbe12aa28f25

def realiza_pruebas_hipotesis():
    resultadosExactitudEnt = [0.961038961038961, 0.39285714285714285, 0.974025974025974, 0.7078651685393258, 0.7701149425287356, 0.21875, 0.9710144927536232, 0.961038961038961, 0.8390804597701149, 0.90625, 0.7945205479452054, 0.6623376623376623, 0.5057471264367817, 0.974025974025974, 0.974025974025974, 0.8974358974358975, 0.875, 0.7678571428571429, 0.974025974025974, 0.9333333333333333, 0.36486486486486486, 0.8941176470588236, 0.49333333333333335, 0.95, 0.28378378378378377, 0.9726027397260274, 0.9722222222222222, 0.8961038961038961, 0.972972972972973, 0.9024390243902439, 0.922077922077922, 0.8857142857142857, 0.974025974025974, 0.8450704225352113, 0.9770114942528736, 0.9743589743589743, 0.6781609195402298, 0.975609756097561, 0.5540540540540541, 0.4588235294117647, 0.8823529411764706, 0.974025974025974, 0.918918918918919, 0.8875, 0.7402597402597403, 0.974025974025974, 0.234375, 0.828125, 0.9759036144578314, 0.975, 0.9620253164556962, 0.96875, 0.9767441860465116, 0.25675675675675674, 0.8941176470588236, 0.44, 0.974025974025974, 0.15625, 0.475, 0.9782608695652174, 0.7126436781609196, 0.9487179487179487, 0.5076923076923077, 0.935064935064935]
    resultadosExactitudPrb = [0.8024691358024691, 0.9058823529411765, 0.8941176470588236, 0.41333333333333333, 0.9625, 0.859375, 0.5057471264367817, 0.90625, 0.5076923076923077, 0.974025974025974, 0.974025974025974, 0.935064935064935, 0.971830985915493, 0.7790697674418605, 0.9743589743589743, 0.9692307692307692, 0.9761904761904762]
    #datos = [50, 61, 54, 45, 71, 65, 75, 35, 85, 71, 72, 70, 58, 50, 87, 51, 58, 63, 60, 60, 81, 38, 36, 64, 39, 60, 42, 74, 51, 68]
    #resultadosPrueba = scipy.stats.ttest_1samp(datos, 47.5, alternative="greater") #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html
    pruebaHipotesisExactitud = scipy.stats.ttest_1samp(resultadosExactitudEnt + resultadosExactitudPrb, 0.7, alternative="greater") #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html
    pruebaHipotesisExactitudEnt = scipy.stats.ttest_1samp(resultadosExactitudEnt, 0.7, alternative="greater") #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html
    pruebaHipotesisExactitudPrb = scipy.stats.ttest_1samp(resultadosExactitudPrb, 0.7, alternative="greater") #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html
    
    print()
    print("Exactitud")
    print("Conjunto de datos:", pruebaHipotesisExactitud)
    print("Conjunto de entrenamiento:", pruebaHipotesisExactitudEnt)
    print("Conjunto de prueba:", pruebaHipotesisExactitudPrb)
    print()

    resultadosTiempoEnt = [10.028615951538086, 12.504640579223633, 21.960879802703857, 19.678935050964355, 17.554142713546753, 13.809925317764282, 18.68346405029297, 17.371856451034546, 16.679930686950684, 16.10175633430481, 15.646751642227173, 13.619065761566162, 13.953370809555054, 22.03903102874756, 18.1879403591156, 18.754979372024536, 15.00941014289856, 15.501834392547607, 16.14695692062378, 17.544062614440918, 13.41160249710083, 15.762532949447632, 14.71546220779419, 16.509421825408936, 15.282813787460327, 15.596207618713379, 19.954705715179443, 22.626824617385864, 19.82835292816162, 15.11347484588623, 17.362520456314087, 17.1557195186615, 17.42752766609192, 16.835232734680176, 18.642253875732422, 16.282840251922607, 17.127564191818237, 15.746997356414795, 16.08846640586853, 15.526576280593872, 20.322158098220825, 16.392971515655518, 16.9985249042511, 16.483885526657104, 11.686841487884521, 14.929097652435303, 14.588787078857422, 16.136983156204224, 17.59822964668274, 17.12009334564209, 18.087865114212036, 17.953559398651123, 16.143086671829224, 14.356966495513916, 16.217118740081787, 14.8655366897583, 11.151258707046509, 11.939844608306885, 16.609525442123413, 50.19055461883545, 16.424735069274902, 16.218306064605713, 16.13312029838562, 17.08066177368164]
    resultadosTiempoPrb = [16.564479112625122, 16.697421312332153, 19.8209068775177, 17.220733642578125, 18.37381935119629, 18.022786855697632, 16.273658752441406, 17.197879552841187, 17.177860260009766, 18.398531913757324, 18.571319818496704, 18.698906898498535, 24.901489734649658, 17.811389684677124, 17.103251457214355, 18.989302158355713, 18.40819478034973]

    pruebaHipotesisTiempo = scipy.stats.ttest_1samp(resultadosTiempoEnt + resultadosTiempoPrb, 30, alternative="less") #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html
    pruebaHipotesisTiempoEnt = scipy.stats.ttest_1samp(resultadosTiempoEnt, 30, alternative="less") #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html
    pruebaHipotesisTiempoPrb = scipy.stats.ttest_1samp(resultadosTiempoPrb, 30, alternative="less") #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html
    
    print()
    print("Tiempo")
    print("Conjunto de datos:", pruebaHipotesisTiempo)
    print("Conjunto de entrenamiento:", pruebaHipotesisTiempoEnt)
    print("Conjunto de prueba:", pruebaHipotesisTiempoPrb)
    print()

    tpCredencialesEnt = ['h', 'h', 'h', 'v', 'v', 'v', 'h', 'h', 'h', 'v', 'v', 'v', 'v', 'h', 'h', 'v', 'h', 'v', 'h', 'h', 'h', 'h', 'h', 'v', 'h', 'h', 'v', 'h', 'h', 'v', 'h', 'h', 'v', 'h', 'v', 'h', 'v', 'v', 'h', 'h', 'h', 'v', 'h', 'v', 'v', 'h', 'v', 'v', 'v', 'v', 'h', 'v', 'v', 'h', 'h', 'h', 'h', 'v', 'v', 'h', 'v', 'h', 'h', 'h']
    tpCredencialesPrb = ['h', 'h', 'h', 'h', 'v', 'v', 'v', 'v', 'h', 'h', 'h', 'h', 'v', 'h', 'v', 'h', 'v']

    resultadosExactitudVEnt = []
    resultadosExactitudHEnt = []
    for exactitud, tipo in zip(resultadosExactitudEnt, tpCredencialesEnt):
        if tipo == 'h':
            resultadosExactitudHEnt.append(exactitud)
        else:
            resultadosExactitudVEnt.append(exactitud)
    
    resultadosExactitudVPrb = []
    resultadosExactitudHPrb = []
    for exactitud, tipo in zip(resultadosExactitudPrb, tpCredencialesPrb):
        if tipo == 'h':
            resultadosExactitudHPrb.append(exactitud)
        else:
            resultadosExactitudVPrb.append(exactitud)
    
    pruebaHipotesisExactitudHV = scipy.stats.ttest_ind((resultadosExactitudHEnt + resultadosExactitudHPrb), (resultadosExactitudVEnt + resultadosExactitudVPrb), equal_var=False) #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    pruebaHipotesisExactitudHVEnt = scipy.stats.ttest_ind(resultadosExactitudHEnt, resultadosExactitudVEnt, equal_var=False) #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    pruebaHipotesisExactitudHVPrb = scipy.stats.ttest_ind(resultadosExactitudHPrb, resultadosExactitudVPrb, equal_var=False) #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html

    print()
    print("Exactitud por tipo de credencial")
    print("Conjunto de datos:", pruebaHipotesisExactitudHV)
    print("Conjunto de entrenamiento:", pruebaHipotesisExactitudHVEnt)
    print("Conjunto de prueba:", pruebaHipotesisExactitudHVPrb)
    print()


def main():
    realiza_pruebas_hipotesis()

if __name__ == "__main__":
    main()