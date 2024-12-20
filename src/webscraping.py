import requests #https://docs.python-requests.org/en/latest/user/quickstart/#quickstart
import bs4 #https://www.crummy.com/software/BeautifulSoup/bs4/doc/#quick-start

def extrae_informacion(url):
    respuesta = requests.get(url, verify=False) #https://docs.python-requests.org/en/latest/api/#requests.get
    pagina = respuesta.text #https://docs.python-requests.org/en/latest/api/#requests.Response.text

    elementos = bs4.BeautifulSoup(pagina, "html.parser") #https://www.crummy.com/software/BeautifulSoup/bs4/doc/#making-the-soup
    
    nombre = elementos.find("div", class_="nombre") #https://www.crummy.com/software/BeautifulSoup/bs4/doc/#navigating-using-tag-names
    nombre = nombre.get_text() #https://www.crummy.com/software/BeautifulSoup/bs4/doc/#get-text
    
    boleta = elementos.find("div", class_="boleta")
    boleta = boleta.get_text()
    
    programaAcademico = elementos.find("div", class_="carrera")
    programaAcademico = programaAcademico.get_text()
    
    unidadAcademica = elementos.find("div", class_="escuela")
    unidadAcademica = unidadAcademica.get_text()

    datos = [nombre, boleta, programaAcademico, unidadAcademica]
    return datos

def main():
    datos = None
    #datos = extrae_informacion("https://servicios.dae.ipn.mx/vcred/?h=aa9f13071ad0a7dc32ce6863fbc8bc0760079d2ce4ef8f288a1e7cc93e0dd875")
    print(datos)

if __name__ == "__main__":
    main()
