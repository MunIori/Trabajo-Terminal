import math #https://docs.python.org/3/library/math.html#module-math

def crea_lista_1d(d1, valor=0):
    lista = [valor] * d1
    return lista

def crea_lista_2d(d1, d2, valor=0):
    lista = crea_lista_1d(d1)
    for i in range(d1):
        lista[i] = crea_lista_1d(d2, valor)
    return lista

def calcula_distancia(cadena, texto):
    relleno = "#" * len(cadena)
    texto = relleno + texto

    nCadena = len(cadena)
    nTexto = len(texto)
    distanciaMenorSubcadenaFin = crea_lista_2d(nCadena + 1, nTexto + 1)
    for i in range(1, nCadena + 1):
        iCadena = i - 1
        for j in range(1, nTexto + 1):
            iTexto = j - 1
            if cadena[iCadena] == texto[iTexto]:
                distanciaMenorSubcadenaFin[i][j] = distanciaMenorSubcadenaFin[i - 1][j - 1]
            else: 
                distanciaInsercion = distanciaMenorSubcadenaFin[i - 1][j]
                distanciaEliminacion = distanciaMenorSubcadenaFin[i][j - 1]
                distanciaSustitucion = distanciaMenorSubcadenaFin[i - 1][j - 1]
                distanciaMenorSubcadenaFin[i][j] = min(distanciaInsercion, distanciaEliminacion, distanciaSustitucion) + 1 #https://docs.python.org/3/library/functions.html#min
        #print(distanciaMenorSubcadenaFin[i])

    minDistancia = nTexto
    jResultadoTabla = -1
    for j in range(nCadena, nTexto + 1):
        if distanciaMenorSubcadenaFin[-1][j] < minDistancia:
            minDistancia = distanciaMenorSubcadenaFin[-1][j]
            jResultadoTabla = j
    iResultadoTexto = jResultadoTabla - 1

    print()
    print(minDistancia)
    print(f" {cadena}")
    print(f'"{texto[iResultadoTexto - nCadena + 1 : iResultadoTexto + 1]}"')
    print()

    return minDistancia
    
def preprocesa_texto(texto, nueEspacio="_"):
    textoPreprocesado = texto.strip() #https://docs.python.org/3/library/stdtypes.html#str.strip
    textoPreprocesado = texto.upper() #https://docs.python.org/3/library/stdtypes.html#str.upper
    textoPreprocesado = textoPreprocesado.replace(" ", nueEspacio) #https://docs.python.org/3/library/stdtypes.html#str.replace
    cambios = str.maketrans("ÁÉÍÓÚ", "AEIOU") #https://docs.python.org/3/library/stdtypes.html#str.maketrans
    textoPreprocesado = textoPreprocesado.translate(cambios) #https://docs.python.org/3/library/stdtypes.html#str.translate
    return textoPreprocesado

def preprocesa_textos(textos, nueEspacio="_"):
    nTextos = len(textos)
    textosPreprocesados = crea_lista_1d(nTextos)
    for i, texto in enumerate(textos):
        textoPreprocesado = preprocesa_texto(texto, nueEspacio)
        textosPreprocesados[i] = textoPreprocesado
    return textosPreprocesados

def calcula_exactitud_caracteres(datos, texto, nueEspacio="_"):
    nombre = datos[0]
    datos = datos[1:]
    componentesNombre = nombre.split() #https://docs.python.org/3/library/stdtypes.html#str.split

    texto = preprocesa_texto(texto, nueEspacio)
    componentesNombre = preprocesa_textos(componentesNombre, nueEspacio)
    datos = preprocesa_textos(datos, nueEspacio)

    minDistancia = math.inf #https://docs.python.org/3/library/math.html#math.inf
    nComponentes = len(componentesNombre)
    for i in range(nComponentes):
        nombre = componentesNombre[i:] + componentesNombre[:i]
        nombre = nueEspacio.join(nombre) #https://docs.python.org/3/library/stdtypes.html#str.join
        distancia = calcula_distancia(nombre, texto)
        minDistancia = min(minDistancia, distancia) #https://docs.python.org/3/library/functions.html#min

    acDistancias = minDistancia 
    acLongitudes = len(nombre)
    for elemento in datos:
        acLongitudes += len(elemento)
        acDistancias += calcula_distancia(elemento, texto)

    exactitud = (acLongitudes - acDistancias) / acLongitudes
    return exactitud

def main():
    exactitud = calcula_exactitud_caracteres(["RODRIGO"], "aaaaraodrigo")
    print(exactitud)
    exactitud = calcula_exactitud_caracteres(["RODRIGO OLARTE ASTUDILLO", "2020630602", "INGENIERÍA EN INTELIGENCIA ARTIFICIAL", "ESCOM"], 'Instituto Politecnico NAcional La Tecnica al servicio de la patria alumno: Munguia Poblano Erwin Programa academico: Ingenieria en inteligencia artificial unidad academica: 2021630463 escom escom semestre refrenado 2021/1 2021/2 2022/1 2022/2 2023/1 2023/2 2024/1 2024/2')
    print(exactitud)
    exactitud = calcula_exactitud_caracteres(["Erwin Munguia poblano", "2021630463", "INGENIERÍA EN INTELIGENCIA ARTIFICIAL", "ESCOM"], 'Instituto Politecnico NAcional La Tecnica al servicio de la patria alumno: Munguia Poblano Erwin Programa academico: Ingenieria en inteligencia artificial unidad academica: 2021630463 escom escom semestre refrenado 2021/1 2021/2 2022/1 2022/2 2023/1 2023/2 2024/1 2024/2')
    print(exactitud)
    exactitud = calcula_exactitud_caracteres(["Erwin Munguia poblano", "2021630463", "INGENIERÍA EN INTELIGENCIA ARTIFICIAL", "ESCOM"], 'InsItuto PolitecnIco Nacional La Tecnica al servicio de la patria alUmo: MungUia Poblano Ervin Programa academico: IngenIerIa en Inteligencia artificial unidad academica: 2021630463 escom escom Semestre refrenado 2021/1 2021/2 2022/1 2022/2 2023/1 2023/2 2024/1 2024/2')
    print(exactitud)
    exactitud = calcula_exactitud_caracteres(["Erwin Munguia poblano", "2021630463", "INGENIERÍA EN INTELIGENCIA ARTIFICIAL", "ESCOM"], 'Inztituto Polytekniko Nacional La Technika al servisio de la patria alumno: Munguia Poblano Ervin Programa akademiko: Ingenyeria en intelijensia artifisial unidad akademika: 2021630463 eskom eskom semestre refrenado')
    print(exactitud)
    
    exactitud = calcula_exactitud_caracteres(["RODRIGO OLARTE ASTUDILLO", "2020630602", "INGENIERÍA EN INTELIGENCIA ARTIFICIAL", "ESCOM"], "IPSLILULOP0IILECPICON4BCIOP3ITCCPIC7SCTVICIOCPJCTIJAIUPTPOOLARTEACRTEASTUDVTUDILLCCODRIGAPCOUTHV34PTICOINGENIEFENIERIAEINSISTTEMASTACIONALESZCOMPUUPI6HICH6QPTICH7020630602ESCOMSCOMS2PPCSLT2RFT2P6H8III20II070IZI2021I1Z021IZII2022I1IIZ022I2II23I1II")
    print(exactitud)
    exactitud = calcula_exactitud_caracteres(["Luis Gerardo Ortiz Cruz", "2021630033", "INGENIERÍA EN INTELIGENCIA ARTIFICIAL", "ESCOM"], "IPSLILHLOPOIILRCPICONHCIOPHITRCPICBISCTVIC1OQP7RTCJAIUPTPOORTIZCRUZLUISCGERARDOPTO9FJDTJTIJ66P9IC5INGENIERIARIAENINTEIGENCIAARTIIFICIALULA4UC6I2021I63QP33SCONESCOMSCPFRSLTQITRTTQH8D6O2021I1II20Z1I2ZI2I202J1II2023I2120Z4III2024QI")
    print(exactitud)
    exactitud = calcula_exactitud_caracteres(["Erwin Munguia Poblano", "2021630463", "INGENIERÍA EN INTELIGENCIA ARTIFICIAL", "ESCOM"], "PSRILHZOPO1ILQCP1CONZC9OP3ISKPTCCPICYJSC7TPICIOPIICTTJAILIPXPOTU1UNGUIAYBLANOERWINPTO9THGTJGLI1FCOIGENIERIAEAENINTELIGENCARTIFICUXL3CD8PP1C120214I63J0A63SCOMCLCISQHPQSLT4TCFTCPHA4OI20ZS41I3I20Z9CEIZO2IZO27J1M2I7023TSXIZQZ3JZIXCI")
    print(exactitud)

    exactitud = calcula_exactitud_caracteres(["RODRIGO OLARTE ASTUDILLO", "2020630602", "INGENIERÍA EN INTELIGENCIA ARTIFICIAL", "ESCOM"], "IPSLILULOP0IILECPICON4BCIOP3ITCCPIC7SCTVICIOCPJCTIJAIUPTPOOLARTEACRTEASTUDVTUDILLCCODRIGAPCOUTHV34PTICOINGENIEFENIERIAEINSISTTEMASTACIONALESZCOMPUUPI6HICH6QPTICH7020630602ESCOMSCOMS2PPCSLT2RFT2P6H8III20II070IZI2021I1Z021IZII2022I1IIZ022I2II23I1II", nueEspacio="")
    print(exactitud)
    exactitud = calcula_exactitud_caracteres(["Luis Gerardo Ortiz Cruz", "2021630033", "INGENIERÍA EN INTELIGENCIA ARTIFICIAL", "ESCOM"], "IPSLILHLOPOIILRCPICONHCIOPHITRCPICBISCTVIC1OQP7RTCJAIUPTPOORTIZCRUZLUISCGERARDOPTO9FJDTJTIJ66P9IC5INGENIERIARIAENINTEIGENCIAARTIIFICIALULA4UC6I2021I63QP33SCONESCOMSCPFRSLTQITRTTQH8D6O2021I1II20Z1I2ZI2I202J1II2023I2120Z4III2024QI", nueEspacio="")
    print(exactitud)
    exactitud = calcula_exactitud_caracteres(["Erwin Munguia Poblano", "2021630463", "INGENIERÍA EN INTELIGENCIA ARTIFICIAL", "ESCOM"], "PSRILHZOPO1ILQCP1CONZC9OP3ISKPTCCPICYJSC7TPICIOPIICTTJAILIPXPOTU1UNGUIAYBLANOERWINPTO9THGTJGLI1FCOIGENIERIAEAENINTELIGENCARTIFICUXL3CD8PP1C120214I63J0A63SCOMCLCISQHPQSLT4TCFTCPHA4OI20ZS41I3I20Z9CEIZO2IZO27J1M2I7023TSXIZQZ3JZIXCI", nueEspacio="")
    print(exactitud)


if __name__ == "__main__":
    main()
