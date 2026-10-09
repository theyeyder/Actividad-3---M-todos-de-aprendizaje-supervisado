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