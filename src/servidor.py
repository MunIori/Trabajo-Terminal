import base64

import webscraping 
import cadenas
import qr
import ocr

import random #https://docs.python.org/3/library/random.html
import flask #https://flask.palletsprojects.com/en/3.0.x/quickstart/#quickstart

import numpy as np
import cv2

def decodifica_imagen(imagen):
    imagen = imagen.removeprefix("data:image/png;base64,")
    imagenBytes = base64.b64decode(imagen)
    imagenBytes = np.frombuffer(imagenBytes, np.uint8)
    imagen = cv2.imdecode(imagenBytes, cv2.IMREAD_COLOR)
    return imagen

servidor = flask.Flask(__name__) #https://flask.palletsprojects.com/en/3.0.x/api/#flask.Flask

@servidor.get("/") #https://flask.palletsprojects.com/en/3.0.x/api/#flask.Flask.get
def interfaz_indice():
    with open("indice.html", "r", encoding="utf8") as finput: #https://docs.python.org/3/library/functions.html#open
        pagina = finput.readlines() #https://docs.python.org/3/tutorial/inputoutput.html#methods-of-file-objects
        pagina = "\n".join(pagina) #https://docs.python.org/3/library/stdtypes.html#str.join
    return pagina

@servidor.get("/fotografia") #https://flask.palletsprojects.com/en/3.0.x/api/#flask.Flask.get
def interfaz_fotografia():
    with open("fotografia.html", "r", encoding="utf8") as finput: #https://docs.python.org/3/library/functions.html#open
        pagina = finput.readlines() #https://docs.python.org/3/tutorial/inputoutput.html#methods-of-file-objects
        pagina = "\n".join(pagina) #https://docs.python.org/3/library/stdtypes.html#str.join
    return pagina

@servidor.post("/fotografia") #https://flask.palletsprojects.com/en/3.0.x/api/#flask.Flask.post
def procesa_datos(): #https://flask.palletsprojects.com/en/3.0.x/patterns/javascript/#making-a-request-with-fetch
    peticion = flask.request #https://flask.palletsprojects.com/en/3.0.x/api/#flask.request
    datos = peticion.form #https://flask.palletsprojects.com/en/3.0.x/api/#flask.Request.form

    credencial = datos["credencial"] #https://werkzeug.palletsprojects.com/en/3.0.x/datastructures/#werkzeug.datastructures.MultiDict
    
    resultados = [random.random()] #https://docs.python.org/3/library/random.html#random.random
    
    print()
    print("---") 
    print(datos)
    print("---")
    print(credencial)
    print("---")
    print(resultados)
    print()
    
    return resultados

@servidor.get("/resultados") #https://flask.palletsprojects.com/en/3.0.x/api/#flask.Flask.get
def interfaz_resultados():
    with open("resultados.html", "r", encoding="utf8") as finput: #https://docs.python.org/3/library/functions.html#open
        pagina = finput.readlines() #https://docs.python.org/3/tutorial/inputoutput.html#methods-of-file-objects
        pagina = "\n".join(pagina) #https://docs.python.org/3/library/stdtypes.html#str.join
    return pagina

@servidor.get("/escaner") #https://flask.palletsprojects.com/en/3.0.x/api/#flask.Flask.get
def interfaz_escaner():
    with open("escaner.html", "r", encoding="utf8") as finput: #https://docs.python.org/3/library/functions.html#open
        pagina = finput.readlines() #https://docs.python.org/3/tutorial/inputoutput.html#methods-of-file-objects
        pagina = "\n".join(pagina) #https://docs.python.org/3/library/stdtypes.html#str.join
    return pagina

@servidor.get("/inicio") #https://flask.palletsprojects.com/en/3.0.x/api/#flask.Flask.get
def interfaz_inicio():
    with open("inicio.html", "r", encoding="utf8") as finput: #https://docs.python.org/3/library/functions.html#open
        pagina = finput.readlines() #https://docs.python.org/3/tutorial/inputoutput.html#methods-of-file-objects
        pagina = "\n".join(pagina) #https://docs.python.org/3/library/stdtypes.html#str.join
    return pagina

@servidor.get("/herramienta") #https://flask.palletsprojects.com/en/3.0.x/api/#flask.Flask.get
def interfaz_herramienta():
    with open("herramienta.html", "r", encoding="utf8") as finput: #https://docs.python.org/3/library/functions.html#open
        pagina = finput.readlines() #https://docs.python.org/3/tutorial/inputoutput.html#methods-of-file-objects
        pagina = "\n".join(pagina) #https://docs.python.org/3/library/stdtypes.html#str.join
    return pagina

@servidor.post("/herramienta") #https://flask.palletsprojects.com/en/3.0.x/api/#flask.Flask.post
def procesa_datos_herramienta(): #https://flask.palletsprojects.com/en/3.0.x/patterns/javascript/#making-a-request-with-fetch
    try:
        peticion = flask.request #https://flask.palletsprojects.com/en/3.0.x/api/#flask.request
        datos = peticion.form #https://flask.palletsprojects.com/en/3.0.x/api/#flask.Request.form

        indice = datos["indice"] #https://werkzeug.palletsprojects.com/en/3.0.x/datastructures/#werkzeug.datastructures.MultiDict
        credencial1 = datos["frente"] #https://werkzeug.palletsprojects.com/en/3.0.x/datastructures/#werkzeug.datastructures.MultiDict
        credencial2 = datos["reverso"] #https://werkzeug.palletsprojects.com/en/3.0.x/datastructures/#werkzeug.datastructures.MultiDict
        
        print()
        print("Datos recibidos:")
        print("Indice:", indice)
        print(f"Frente: {credencial1[:100]}...")
        print(f"Reverso: {credencial2[:100]}...")
        print()
        
        #resultados = [indice, random.randint(20200000, 20230000), "Selene Delgado López", random.random()] #https://docs.python.org/3/library/random.html#random.random
        
        print()
        print("Decodificando imagen 1...")
        credencial1 = decodifica_imagen(credencial1)
        print("Imagen decodificada:", credencial1.shape)
        print()

        print()
        print("Reconociendo caracteres...")
        textos = ocr.crea_lista_1d(4)
        for i in range(4):
            texto = ocr.reconoce_caracteres_easyocr(credencial1)
            textos[i] = texto
            credencial1 = cv2.rotate(credencial1, cv2.ROTATE_90_CLOCKWISE)
        textoCredencial = " ".join(textos)
        print("Caracteres reconocidos:", len(textoCredencial))
        print(f'"{textoCredencial}"')
        print()

        print()
        print("Decodificando imagen 2...")
        credencial2 = decodifica_imagen(credencial2)
        print("Imagen decodificada:", credencial2.shape)
        print()

        print()
        print("Decodificando QR...")
        url = qr.decodifica_qr_opencv(credencial2)
        if url == "":
            url = qr.decodifica_qr_qreader(credencial2)
        if url != "":
            print(f'QR decodificado: "{url}"')
            print()

            if url.startswith("https://www.dae.ipn.mx/vcred/?h="):
                
                print()
                print("Extrayendo la información de la página...")
                resultados = webscraping.extrae_informacion(url)
                print("Información extraida:", resultados)
                print()

                [nombre, boleta, programaAcademico, unidadAcademica] = resultados

                print()
                print("Calculando exactitud...")
                exactitud = cadenas.calcula_exactitud_caracteres(resultados, textoCredencial)
                exactitud = exactitud * 100
                if exactitud > 80:
                    exactitud = f"Admisible ({exactitud:.2f}% exactitud)"
                else:
                    exactitud = f"Insuficiente ({exactitud:.2f}% exactitud)"
                print("Exactitud calculada:", exactitud)
                print()

                resultados = [indice, boleta, nombre, exactitud]

            else:
                print()
                print("URL incorrecta")
                print()

                resultados = [indice, "-", "-", "URL incorrecta"]
        else:
            print("No se pudo decodificar el QR")
            print()

            resultados = [indice, "-", "-", "No se pudo decodificar el QR"]
            
    except:
        print()
        print("No se pudo procesar la imagen")
        print()

        resultados = [indice, "-", "-", "No se pudo procesar la imagen"]
    
    print()
    print("Respuesta:")
    print("Resultados:", resultados)
    print()

    return resultados

#servidor.run("0.0.0.0") #https://flask.palletsprojects.com/en/3.0.x/api/#flask.Flask.run
servidor.run("0.0.0.0", ssl_context="adhoc") #https://werkzeug.palletsprojects.com/en/3.0.x/serving/#werkzeug.serving.run_simple
