import os #https://docs.python.org/3/library/os.html
import time #https://docs.python.org/3/library/time.html

import cv2 #https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
import qreader #https://github.com/Eric-Canas/QReader?tab=readme-ov-file#usage

C_EXPERIMENTOS = 0

def crea_lista_1d(d1, valor=0):
    lista = [valor] * d1
    return lista

def decodifica_qr_opencv(imagen):
    decodificador = cv2.QRCodeDetector() #https://docs.opencv.org/4.x/de/dc3/classcv_1_1QRCodeDetector.html#a3becfe9df48966008179e1e6c39bf8f9

    texto, puntos, imagenQr = decodificador.detectAndDecode(imagen) #https://docs.opencv.org/4.x/d7/d90/classcv_1_1GraphicalCodeDetector.html#a878307029654e43ee8bee1c99efdbdc0
    i = 1
    while texto == "" and i <= 3:
        imagen = cv2.rotate(imagen, cv2.ROTATE_90_COUNTERCLOCKWISE) #https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga4ad01c0978b0ce64baa246811deeac24
        texto, puntos, imagenQr = decodificador.detectAndDecode(imagen) #https://docs.opencv.org/4.x/d7/d90/classcv_1_1GraphicalCodeDetector.html#a878307029654e43ee8bee1c99efdbdc0
        i += 1
    
    return texto

def decodifica_qr_qreader(imagen):
    decodificador = qreader.QReader() #https://github.com/Eric-Canas/QReader?tab=readme-ov-file#qreadermodel_size--s-min_confidence--05-reencode_to--shift-jis
    textos = decodificador.detect_and_decode(imagen, is_bgr=True) #https://github.com/Eric-Canas/QReader?tab=readme-ov-file#qreaderdetect_and_decodeimage-return_detections--false
    texto = textos[0]
    for resultado in textos:
        if resultado != None:
            resultado = texto
    if texto == None:
        texto = ""
    return texto

def muestra_experimento(datos, funcionQR, rDatos=""):
    global C_EXPERIMENTOS
    C_EXPERIMENTOS += 1

    print()
    print("Experimento:", C_EXPERIMENTOS)
    print("Funcion:", funcionQR)

    nImagenes = len(datos)
    tiempos = crea_lista_1d(nImagenes)

    for i, archivo in enumerate(datos):
        imagen = cv2.imread(f"{rDatos}/{archivo}", cv2.IMREAD_UNCHANGED) #https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gab32ee19e22660912565f8140d0f675a8

        t1 = time.time() #https://docs.python.org/3/library/time.html#time.time
        texto = funcionQR(imagen)
        t2 = time.time() #https://docs.python.org/3/library/time.html#time.time
        tiempos[i] = t2 - t1

        print(f'[{archivo}] "{texto}"')

    tiempoPmd = sum(tiempos) / nImagenes #https://docs.python.org/3/library/functions.html#sum

    print(f"Tiempo promedio: {tiempoPmd} s")
    print()

def main():
    R_IMAGENES = "qrs"
    imagenesPrb = os.listdir(R_IMAGENES)

    muestra_experimento(imagenesPrb, decodifica_qr_opencv, R_IMAGENES)
    muestra_experimento(imagenesPrb, decodifica_qr_qreader, R_IMAGENES)

if __name__ == "__main__":
    main()
