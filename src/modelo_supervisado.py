import csv
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

CARACTERISTICAS = [
    "pasajeros",
    "numero_paradas",
    "transbordos",
    "tiempo_estimado",
    "hora_pico"
]

OBJETIVO = "retraso"

RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS = RUTA_PROYECTO / "datos" / "datos_transporte.csv"


# ============================================================
# 1. CARGAR EL DATASET
# ============================================================

def cargar_datos(ruta):
    datos = []

    with ruta.open("r", encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo)

        columnas_esperadas = {
            "id_viaje",
            "pasajeros",
            "numero_paradas",
            "transbordos",
            "tiempo_estimado",
            "hora_pico",
            "retraso"
        }

        if not lector.fieldnames:
            raise ValueError("El archivo CSV no contiene encabezados.")

        faltantes = columnas_esperadas - set(lector.fieldnames)

        if faltantes:
            raise ValueError(
                "Faltan columnas en el dataset: "
                + ", ".join(sorted(faltantes))
            )

        for fila in lector:
            datos.append(
                {
                    "id_viaje": int(fila["id_viaje"]),
                    "pasajeros": int(fila["pasajeros"]),
                    "numero_paradas": int(fila["numero_paradas"]),
                    "transbordos": int(fila["transbordos"]),
                    "tiempo_estimado": int(fila["tiempo_estimado"]),
                    "hora_pico": int(fila["hora_pico"]),
                    "retraso": fila["retraso"].strip()
                }
            )

    if len(datos) < 10:
        raise ValueError(
            "El dataset tiene muy pocos registros para realizar la práctica."
        )

    return datos


# ============================================================
# 2. FUNCIONES MATEMÁTICAS
# ============================================================

def entropia(etiquetas):
    """
    Calcula la entropía:
        H(S) = - sum(p_i * log2(p_i))
    """

    total = len(etiquetas)

    if total == 0:
        return 0.0

    frecuencias = Counter(etiquetas)
    resultado = 0.0

    for cantidad in frecuencias.values():
        proporcion = cantidad / total
        resultado -= proporcion * math.log2(proporcion)

    return resultado


def ganancia_informacion(datos, caracteristica, umbral):
    """
    Ganancia:
        IG = H(S) - (|Izq|/|S|)H(Izq) - (|Der|/|S|)H(Der)
    """

    izquierda = [
        fila for fila in datos
        if fila[caracteristica] <= umbral
    ]

    derecha = [
        fila for fila in datos
        if fila[caracteristica] > umbral
    ]

    if not izquierda or not derecha:
        return 0.0, izquierda, derecha

    etiquetas = [fila[OBJETIVO] for fila in datos]
    etiquetas_izq = [fila[OBJETIVO] for fila in izquierda]
    etiquetas_der = [fila[OBJETIVO] for fila in derecha]

    entropia_padre = entropia(etiquetas)

    entropia_hijos = (
        (len(izquierda) / len(datos)) * entropia(etiquetas_izq)
        + (len(derecha) / len(datos)) * entropia(etiquetas_der)
    )

    ganancia = entropia_padre - entropia_hijos

    return ganancia, izquierda, derecha


# ============================================================
# 3. NODO DEL ÁRBOL
# ============================================================

@dataclass
class Nodo:
    caracteristica: str = None
    umbral: float = None
    izquierda: object = None
    derecha: object = None
    clase: str = None
    muestras: int = 0
    ganancia: float = 0.0

    @property
    def es_hoja(self):
        return self.clase is not None


# ============================================================
# 4. MODELO DE ÁRBOL DE DECISIÓN
# ============================================================

class ArbolDecisionClasificador:
    """
    Árbol de decisión implementado con Python estándar.

    Aprende umbrales a partir de los datos utilizando entropía
    y ganancia de información.
    """

    def __init__(
        self,
        max_profundidad=4,
        min_muestras_division=4,
        min_muestras_hoja=2
    ):
        self.max_profundidad = max_profundidad
        self.min_muestras_division = min_muestras_division
        self.min_muestras_hoja = min_muestras_hoja
        self.raiz = None

    def ajustar(self, datos):
        if not datos:
            raise ValueError("No hay datos para entrenar el modelo.")

        self.raiz = self._construir(datos, profundidad=0)

    def _clase_mayoritaria(self, datos):
        conteo = Counter(
            fila[OBJETIVO]
            for fila in datos
        )

        return conteo.most_common(1)[0][0]

    def _mejor_division(self, datos):
        mejor = None
        mejor_ganancia = 0.0

        for caracteristica in CARACTERISTICAS:
            valores = sorted(
                set(
                    fila[caracteristica]
                    for fila in datos
                )
            )

            if len(valores) < 2:
                continue

            umbrales = [
                (a + b) / 2
                for a, b in zip(
                    valores,
                    valores[1:]
                )
            ]

            for umbral in umbrales:
                (
                    ganancia,
                    izquierda,
                    derecha
                ) = ganancia_informacion(
                    datos,
                    caracteristica,
                    umbral
                )

                if (
                    len(izquierda) < self.min_muestras_hoja
                    or len(derecha) < self.min_muestras_hoja
                ):
                    continue

                if ganancia > mejor_ganancia:
                    mejor_ganancia = ganancia
                    mejor = (
                        caracteristica,
                        umbral,
                        izquierda,
                        derecha,
                        ganancia
                    )

        return mejor

    def _construir(self, datos, profundidad):
        etiquetas = [
            fila[OBJETIVO]
            for fila in datos
        ]

        # Caso 1: todas las observaciones tienen la misma clase.
        if len(set(etiquetas)) == 1:
            return Nodo(
                clase=etiquetas[0],
                muestras=len(datos)
            )

        # Caso 2: se alcanzó la profundidad máxima.
        if profundidad >= self.max_profundidad:
            return Nodo(
                clase=self._clase_mayoritaria(datos),
                muestras=len(datos)
            )

        # Caso 3: no hay suficientes datos para seguir dividiendo.
        if len(datos) < self.min_muestras_division:
            return Nodo(
                clase=self._clase_mayoritaria(datos),
                muestras=len(datos)
            )

        mejor = self._mejor_division(datos)

        # Caso 4: ninguna división mejora la información.
        if mejor is None:
            return Nodo(
                clase=self._clase_mayoritaria(datos),
                muestras=len(datos)
            )

        (
            caracteristica,
            umbral,
            izquierda,
            derecha,
            ganancia
        ) = mejor

        return Nodo(
            caracteristica=caracteristica,
            umbral=umbral,
            izquierda=self._construir(
                izquierda,
                profundidad + 1
            ),
            derecha=self._construir(
                derecha,
                profundidad + 1
            ),
            muestras=len(datos),
            ganancia=ganancia
        )

    def predecir_uno(self, fila):
        if self.raiz is None:
            raise ValueError(
                "El modelo todavía no ha sido entrenado."
            )

        nodo = self.raiz

        while not nodo.es_hoja:
            if fila[nodo.caracteristica] <= nodo.umbral:
                nodo = nodo.izquierda
            else:
                nodo = nodo.derecha

        return nodo.clase

    def predecir(self, datos):
        return [
            self.predecir_uno(fila)
            for fila in datos
        ]

    def reglas(self):
        lineas = []

        def recorrer(nodo, nivel=0):
            sangria = "    " * nivel

            if nodo.es_hoja:
                lineas.append(
                    f"{sangria}=> CLASE = {nodo.clase} "
                    f"(muestras={nodo.muestras})"
                )
                return

            lineas.append(
                f"{sangria}SI {nodo.caracteristica} "
                f"<= {nodo.umbral:.2f}:"
            )

            recorrer(
                nodo.izquierda,
                nivel + 1
            )

            lineas.append(
                f"{sangria}SI {nodo.caracteristica} "
                f"> {nodo.umbral:.2f}:"
            )

            recorrer(
                nodo.derecha,
                nivel + 1
            )

        recorrer(self.raiz)

        return "\n".join(lineas)
        # ============================================================
# 5. DIVISIÓN ESTRATIFICADA ENTRENAMIENTO / PRUEBA
# ============================================================

def dividir_datos(
    datos,
    proporcion_prueba=0.30,
    semilla=42
):
    """
    Separa los registros conservando ejemplos de ambas clases
    en entrenamiento y prueba.
    """

    aleatorio = random.Random(semilla)

    por_clase = {}

    for fila in datos:
        por_clase.setdefault(
            fila[OBJETIVO],
            []
        ).append(fila)

    entrenamiento = []
    prueba = []

    for registros in por_clase.values():
        registros = registros.copy()
        aleatorio.shuffle(registros)

        cantidad_prueba = max(
            1,
            round(
                len(registros)
                * proporcion_prueba
            )
        )

        prueba.extend(
            registros[:cantidad_prueba]
        )

        entrenamiento.extend(
            registros[cantidad_prueba:]
        )

    aleatorio.shuffle(entrenamiento)
    aleatorio.shuffle(prueba)

    return entrenamiento, prueba


# ============================================================
# 6. MÉTRICAS
# ============================================================

def evaluar_modelo(reales, predichas):
    verdaderos_positivos = 0
    verdaderos_negativos = 0
    falsos_positivos = 0
    falsos_negativos = 0

    for real, predicha in zip(
        reales,
        predichas
    ):
        if real == "Si" and predicha == "Si":
            verdaderos_positivos += 1

        elif real == "No" and predicha == "No":
            verdaderos_negativos += 1

        elif real == "No" and predicha == "Si":
            falsos_positivos += 1

        elif real == "Si" and predicha == "No":
            falsos_negativos += 1

    total = len(reales)

    exactitud = (
        verdaderos_positivos
        + verdaderos_negativos
    ) / total

    precision = (
        verdaderos_positivos
        / (
            verdaderos_positivos
            + falsos_positivos
        )
        if (
            verdaderos_positivos
            + falsos_positivos
        ) > 0
        else 0.0
    )

    sensibilidad = (
        verdaderos_positivos
        / (
            verdaderos_positivos
            + falsos_negativos
        )
        if (
            verdaderos_positivos
            + falsos_negativos
        ) > 0
        else 0.0
    )

    f1 = (
        2
        * precision
        * sensibilidad
        / (precision + sensibilidad)
        if (precision + sensibilidad) > 0
        else 0.0
    )

    return {
        "vp": verdaderos_positivos,
        "vn": verdaderos_negativos,
        "fp": falsos_positivos,
        "fn": falsos_negativos,
        "exactitud": exactitud,
        "precision": precision,
        "sensibilidad": sensibilidad,
        "f1": f1
    }


# ============================================================
# 7. ENTRADAS SEGURAS
# ============================================================

def leer_entero(mensaje, minimo, maximo):
    while True:
        try:
            valor = int(
                input(mensaje)
            )

            if minimo <= valor <= maximo:
                return valor

            print(
                f"Ingrese un valor entre "
                f"{minimo} y {maximo}."
            )

        except ValueError:
            print(
                "Debe ingresar únicamente "
                "un número entero."
            )


def leer_hora_pico():
    while True:
        print(
            "\n¿El viaje ocurre en hora pico?"
        )
        print("0. No")
        print("1. Sí")

        opcion = leer_entero(
            "Seleccione una opción: ",
            0,
            1
        )

        return opcion


# ============================================================
# 8. PRESENTACIÓN DE RESULTADOS
# ============================================================

def mostrar_muestra(datos, cantidad=5):
    print(
        "\nPrimeros registros del dataset:"
    )

    encabezado = (
        f"{'ID':>3} "
        f"{'Pasaj.':>7} "
        f"{'Paradas':>7} "
        f"{'Transb.':>7} "
        f"{'Tiempo':>7} "
        f"{'Pico':>5} "
        f"{'Retraso':>8}"
    )

    print(encabezado)
    print("-" * len(encabezado))

    for fila in datos[:cantidad]:
        print(
            f"{fila['id_viaje']:>3} "
            f"{fila['pasajeros']:>7} "
            f"{fila['numero_paradas']:>7} "
            f"{fila['transbordos']:>7} "
            f"{fila['tiempo_estimado']:>7} "
            f"{fila['hora_pico']:>5} "
            f"{fila['retraso']:>8}"
        )


def mostrar_metricas(metricas):
    print(
        "\n========================================"
    )
    print(
        "EVALUACIÓN DEL MODELO"
    )
    print(
        "========================================"
    )

    print(
        f"Exactitud: "
        f"{metricas['exactitud'] * 100:.2f}%"
    )

    print(
        f"Precisión clase 'Si': "
        f"{metricas['precision'] * 100:.2f}%"
    )

    print(
        f"Sensibilidad clase 'Si': "
        f"{metricas['sensibilidad'] * 100:.2f}%"
    )

    print(
        f"F1 clase 'Si': "
        f"{metricas['f1'] * 100:.2f}%"
    )

    print(
        "\nMATRIZ DE CONFUSIÓN"
    )
    print(
        "                    Predicción"
    )
    print(
        "                 No          Si"
    )
    print(
        f"Real No      "
        f"{metricas['vn']:>5}       "
        f"{metricas['fp']:>5}"
    )
    print(
        f"Real Si      "
        f"{metricas['fn']:>5}       "
        f"{metricas['vp']:>5}"
    )


# ============================================================
# 9. PREDICCIÓN INTERACTIVA
# ============================================================

def solicitar_nuevo_viaje():
    print(
        "\n========================================"
    )
    print(
        "NUEVA PREDICCIÓN"
    )
    print(
        "========================================"
    )

    pasajeros = leer_entero(
        "Cantidad de pasajeros (1-120): ",
        1,
        120
    )

    numero_paradas = leer_entero(
        "Número de paradas (1-10): ",
        1,
        10
    )

    transbordos = leer_entero(
        "Número de transbordos (0-5): ",
        0,
        5
    )

    tiempo_estimado = leer_entero(
        "Tiempo estimado en minutos (1-120): ",
        1,
        120
    )

    hora_pico = leer_hora_pico()

    return {
        "pasajeros": pasajeros,
        "numero_paradas": numero_paradas,
        "transbordos": transbordos,
        "tiempo_estimado": tiempo_estimado,
        "hora_pico": hora_pico
    }


# ============================================================
# 10. PROGRAMA PRINCIPAL
# ============================================================

def main():
    print(
        "\n===================================================="
    )
    print(
        " MODELO SUPERVISADO - PREDICCIÓN DE RETRASOS"
    )
    print(
        " SISTEMA DE TRANSPORTE"
    )
    print(
        "===================================================="
    )

    try:
        datos = cargar_datos(
            RUTA_DATOS
        )
    except (
        FileNotFoundError,
        ValueError
    ) as error:
        print(
            "\nERROR AL CARGAR LOS DATOS:"
        )
        print(error)
        return

    conteo = Counter(
        fila[OBJETIVO]
        for fila in datos
    )

    print(
        f"\nArchivo de datos: {RUTA_DATOS}"
    )

    print(
        f"Registros cargados: {len(datos)}"
    )

    print(
        f"Viajes sin retraso: "
        f"{conteo.get('No', 0)}"
    )

    print(
        f"Viajes con retraso: "
        f"{conteo.get('Si', 0)}"
    )

    mostrar_muestra(
        datos
    )

    entrenamiento, prueba = dividir_datos(
        datos,
        proporcion_prueba=0.30,
        semilla=42
    )

    print(
        "\n========================================"
    )
    print(
        "DIVISIÓN DE LOS DATOS"
    )
    print(
        "========================================"
    )

    print(
        f"Entrenamiento: "
        f"{len(entrenamiento)} registros "
        f"(aprox. 70%)"
    )

    print(
        f"Prueba: "
        f"{len(prueba)} registros "
        f"(aprox. 30%)"
    )

    modelo = ArbolDecisionClasificador(
        max_profundidad=4,
        min_muestras_division=4,
        min_muestras_hoja=2
    )

    modelo.ajustar(
        entrenamiento
    )

    reales = [
        fila[OBJETIVO]
        for fila in prueba
    ]

    predichas = modelo.predecir(
        prueba
    )

    metricas = evaluar_modelo(
        reales,
        predichas
    )

    mostrar_metricas(
        metricas
    )

    print(
        "\n========================================"
    )
    print(
        "REGLAS APRENDIDAS POR EL ÁRBOL"
    )
    print(
        "========================================"
    )

    print(
        modelo.reglas()
    )

    print(
        "\n========================================"
    )
    print(
        "RESULTADOS DE LA MUESTRA DE PRUEBA"
    )
    print(
        "========================================"
    )

    print(
        f"{'ID':>3} "
        f"{'Real':>8} "
        f"{'Predicción':>12}"
    )

    print("-" * 27)

    for fila, prediccion in zip(
        prueba,
        predichas
    ):
        print(
            f"{fila['id_viaje']:>3} "
            f"{fila[OBJETIVO]:>8} "
            f"{prediccion:>12}"
        )

    while True:
        nuevo_viaje = solicitar_nuevo_viaje()

        resultado = modelo.predecir_uno(
            nuevo_viaje
        )

        print(
            "\n========================================"
        )
        print(
            "PREDICCIÓN DEL SISTEMA"
        )
        print(
            "========================================"
        )

        print(
            f"Resultado: {resultado}"
        )

        if resultado == "Si":
            print(
                "El modelo predice que el viaje "
                "puede presentar retraso."
            )
        else:
            print(
                "El modelo predice que el viaje "
                "llegará a tiempo."
            )

        print(
            "\n¿Desea realizar otra predicción?"
        )
        print("1. Sí")
        print("2. Salir")

        opcion = leer_entero(
            "Seleccione una opción: ",
            1,
            2
        )

        if opcion == 2:
            print(
                "\nGracias por utilizar "
                "el modelo supervisado."
            )
            break


if __name__ == "__main__":
    main()
