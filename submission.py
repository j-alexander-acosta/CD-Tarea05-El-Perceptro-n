#!/usr/bin/env python3
"""Tarea: El Perceptrón — este es el único archivo que debes completar.

Convenciones que se usan en toda la tarea (las mismas del enunciado):

* ``X`` es un arreglo de numpy de forma ``(n, d)``: una fila por ejemplo.
  Cuando el problema necesita sesgo, la última columna de ``X`` es una
  columna de unos y el sesgo es simplemente una coordenada más de ``w``.
* ``y`` es un arreglo de forma ``(n,)`` con valores en ``{-1, +1}``.
* ``w`` es un arreglo de forma ``(d,)``.
* Un ejemplo está **mal clasificado** cuando su margen
  ``(w . x) * y`` es **menor o igual que cero**. El caso de margen cero (el
  punto cae justo sobre la frontera) cuenta como error: por eso el algoritmo
  se mueve aunque parta de ``w = 0``.

El archivo trae dos cosas. Primero la **sección de respuestas**, donde anotas
los números que imprime ``python experimento.py`` una vez que tu código está
listo. Debajo van las cinco funciones de la tarea. Las dos cosas se corrigen
de forma automática y las dos viajan en este mismo archivo, el único que se
entrega.

Solo se permiten numpy y matplotlib. No modifiques las firmas de las
funciones ni los nombres de las constantes: el corrector automático los busca
tal como están escritos.
"""
from __future__ import annotations

import numpy as np

from util import plt


########################################################################
########################################################################
##
##  SECCIÓN DE RESPUESTAS
##
##  Acá van los números que produce tu propio código. No se calculan a
##  mano ni se copian del enunciado: se leen de lo que imprime
##
##      python experimento.py
##
##  cuando las cinco funciones ya están completas. Cada constante lleva
##  encima el experimento y la columna de la que sale.
##
##  Cómo se comparan las respuestas:
##
##  * los conteos (épocas y actualizaciones) son enteros exactos;
##  * la cota se compara con UN decimal y el radio R con TRES, la misma
##    precisión con la que los imprime experimento.py;
##  * las claves de cada diccionario ya vienen puestas y tienen que
##    quedar tal cual, ni una más ni una menos.
##
##  `grader.py` no revisa esta sección: sus puntos los pone el corrector
##  del curso al momento de entregar. Debajo empieza el código, y ese sí
##  lo revisan las pruebas de `grader.py`.
##
########################################################################
########################################################################


############################################################
# Problema 2(b). La tabla del primer experimento
#
# El primer experimento entrena sobre conjuntos separables de 120
# puntos con márgenes cada vez más pequeños e imprime una tabla de
# cuatro filas. Las tres constantes de abajo son tres columnas de esa
# tabla, con el margen de la fila como clave.
############################################################

# Columna «actualiz.»: cuántas actualizaciones aplicó tu entrenamiento
# antes de converger, en cada uno de los cuatro conjuntos. Enteros.
# BEGIN_TU_CODIGO
TABLA_ACTUALIZACIONES = {0.5: 1, 0.25: 3, 0.1: 10, 0.05: 16}
# END_TU_CODIGO

# Columna «cota»: el (R/gamma)^2 que devuelve tu cota_actualizaciones
# sobre esa misma fila, con un decimal.
# BEGIN_TU_CODIGO
TABLA_COTA = {0.5: 4.0, 0.25: 15.8, 0.1: 98.7, 0.05: 387.7}
# END_TU_CODIGO

# Columna «R»: la norma del ejemplo más largo del conjunto, con tres
# decimales.
# BEGIN_TU_CODIGO
TABLA_RADIO = {0.5: 0.997, 0.25: 0.993, 0.1: 0.993, 0.05: 0.985}
# END_TU_CODIGO


############################################################
# Problema 3(b). El experimento de XOR
#
# El tercer experimento entrena sobre los cuatro ejemplos de XOR en el
# espacio aumentado con tres topes de épocas distintos, y después
# repite el entrenamiento sobre la representación (h1, h2, 1) que
# produce la capa oculta fija.
############################################################

# Las actualizaciones que aplica sobre XOR con cada tope de épocas. La
# clave es el max_epocas y el valor, el total de actualizaciones.
# BEGIN_TU_CODIGO
XOR_ACTUALIZACIONES = {2000: 8000, 4000: 16000, 8000: 32000}
# END_TU_CODIGO

# Ahora el entrenamiento sobre la representación oculta, que sí
# converge: las épocas que recorre y las actualizaciones que aplica.
# BEGIN_TU_CODIGO
XOR_OCULTA_EPOCAS = 7
# END_TU_CODIGO

# BEGIN_TU_CODIGO
XOR_OCULTA_ACTUALIZACIONES = 11
# END_TU_CODIGO

# Las tres coordenadas del «w final» de ese mismo entrenamiento, en el
# orden en que las imprime el programa.
# BEGIN_TU_CODIGO
XOR_OCULTA_W = [2.0, -3.0, -1.0]
# END_TU_CODIGO


########################################################################
########################################################################
##
##  EL CÓDIGO DE LA TAREA
##
##  Las cinco funciones. No modifiques las firmas: el corrector
##  automático las llama tal como están escritas.
##
########################################################################
########################################################################


# ----------------------------------------------------------------------
# Parte 1: márgenes
# ----------------------------------------------------------------------
def margenes(X, y, w):
    """Margen de cada ejemplo respecto del vector de pesos ``w``.

    El margen del ejemplo ``i`` es ``(w . x_i) * y_i``: positivo cuando la
    predicción acierta, negativo cuando falla, y cero cuando el punto cae
    exactamente sobre la frontera.

    Devuelve un arreglo de forma ``(n,)`` en el mismo orden de las filas de
    ``X``. No modifica ninguno de los arreglos que recibe.

    Ejemplo::

        >>> margenes(np.array([[3.0, 1.0], [0.0, 2.0]]),
        ...          np.array([1, -1]), np.array([1.0, -3.0]))
        array([0., 6.])
    """
    # BEGIN_TU_CODIGO
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    return (X_arr @ w_arr) * y_arr
    # END_TU_CODIGO


# ----------------------------------------------------------------------
# Parte 2: la regla de actualización
# ----------------------------------------------------------------------
def paso_perceptron(x, y, w):
    """Aplica la regla del perceptrón a un único ejemplo ``(x, y)``.

    La regla es::

        si (w . x) * y <= 0:   w <- w + y * x
        si no:                 w queda igual

    Devuelve la tupla ``(w_nuevo, actualizo)``, donde ``w_nuevo`` es un
    arreglo **nuevo** de forma ``(d,)`` y ``actualizo`` es un ``bool`` que
    dice si hubo o no actualización.

    Importante: esta función **no** debe modificar ``w`` en el lugar. El
    corrector comprueba que el arreglo que recibiste siga intacto después de
    llamarla.
    """
    # BEGIN_TU_CODIGO
    x_arr = np.asarray(x, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    y_val = float(y)
    if float(np.dot(w_arr, x_arr) * y_val) <= 0.0:
        return w_arr + y_val * x_arr, True
    return w_arr.copy(), False
    # END_TU_CODIGO


# ----------------------------------------------------------------------
# Parte 3: el bucle de entrenamiento
# ----------------------------------------------------------------------
def entrenar_perceptron(X, y, max_epocas=100, w0=None):
    """Entrena un perceptrón recorriendo los datos por épocas.

    Una **época** es una pasada completa sobre las filas de ``X``, en el
    orden en que vienen, aplicando ``paso_perceptron`` a cada ejemplo (las
    actualizaciones se acumulan dentro de la misma época: el segundo ejemplo
    ya ve el ``w`` corregido por el primero).

    Criterio de parada: se detiene cuando una época completa termina **sin
    ninguna actualización** (ahí el conjunto quedó perfectamente clasificado)
    o cuando se agotan las ``max_epocas``.

    Argumentos:
        X, y: los datos, con las convenciones del módulo.
        max_epocas: tope de pasadas. Es lo que evita un bucle infinito cuando
            los datos no son separables.
        w0: vector inicial. Si es ``None`` se parte de ceros. Nunca se
            modifica: se trabaja sobre una copia.

    Devuelve un diccionario con exactamente estas cuatro claves:

    ==================  ==================================================
    ``"w"``             arreglo ``(d,)`` con los pesos finales
    ``"actualizaciones"``  entero: total de actualizaciones aplicadas
    ``"epocas"``        entero: épocas completas recorridas, **incluida**
                        la última, la que no produjo actualizaciones
    ``"convergio"``     ``bool``: ``True`` solo si una época terminó sin
                        actualizaciones
    ==================  ==================================================
    """
    # BEGIN_TU_CODIGO
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    if w0 is None:
        w = np.zeros(X_arr.shape[1], dtype=float)
    else:
        w = np.array(w0, dtype=float, copy=True)

    actualizaciones = 0
    epocas = 0
    convergio = False

    for _ in range(max_epocas):
        epocas += 1
        actualizaciones_epoca = 0
        for xi, yi in zip(X_arr, y_arr):
            w, actualizo = paso_perceptron(xi, yi, w)
            if actualizo:
                actualizaciones_epoca += 1
        actualizaciones += actualizaciones_epoca
        if actualizaciones_epoca == 0:
            convergio = True
            break

    return {
        "w": w,
        "actualizaciones": int(actualizaciones),
        "epocas": int(epocas),
        "convergio": bool(convergio),
    }
    # END_TU_CODIGO


# ----------------------------------------------------------------------
# Parte 4: la cota teórica de convergencia
# ----------------------------------------------------------------------
def cota_actualizaciones(X, y, w_ref):
    """Cota teórica del número de actualizaciones del perceptrón.

    Si existe un vector ``w_ref`` que separa las clases, el algoritmo
    (partiendo de ``w = 0``) hace a lo más ``(R / gamma) ** 2``
    actualizaciones, donde

    * ``R = max_i ||x_i||`` es la norma del ejemplo más largo, y
    * ``gamma = min_i (w_ref . x_i) * y_i / ||w_ref||`` es el margen
      geométrico que consigue ``w_ref``, es decir la distancia del punto más
      cercano a la frontera que define ``w_ref``.

    Devuelve la cota como ``float``.

    Lanza ``ValueError`` si ``w_ref`` es el vector cero o si no separa las
    clases (o sea, si el margen geométrico no resulta estrictamente
    positivo): en ese caso el teorema no dice nada y devolver un número
    sería mentir.
    """
    # BEGIN_TU_CODIGO
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    w_arr = np.asarray(w_ref, dtype=float)

    norma_w = float(np.linalg.norm(w_arr))
    if norma_w == 0.0:
        raise ValueError("w_ref no puede ser el vector cero")

    m = margenes(X_arr, y_arr, w_arr) / norma_w
    gamma = float(np.min(m))
    if gamma <= 0.0:
        raise ValueError("w_ref no separa estrictamente las clases")

    R = float(np.max(np.linalg.norm(X_arr, axis=1)))
    return float((R / gamma) ** 2)
    # END_TU_CODIGO


# ----------------------------------------------------------------------
# Parte 5: la frontera de decisión, en un PNG
# ----------------------------------------------------------------------
def graficar_frontera(X, y, w, ruta, titulo="Frontera de decisión"):
    """Guarda en ``ruta`` un PNG con los datos y la frontera de ``w``.

    El gráfico debe tener, como mínimo:

    * los puntos de la clase ``+1`` y los de la clase ``-1`` dibujados con
      marcadores distintos,
    * la frontera ``w . x = 0`` (o los dos semiplanos pintados de colores
      distintos, que es más fácil y se ve mejor),
    * ejes rotulados ``x_1`` y ``x_2``, el ``titulo`` recibido y una leyenda.

    ``X`` puede tener dos columnas (frontera por el origen) o tres, en cuyo
    caso la tercera es la columna de unos y ``w[2]`` es el sesgo. En ambos
    casos se grafica en el plano ``(x_1, x_2)``.

    La función **no devuelve nada**: su efecto es dejar el archivo escrito.
    Cierra la figura al terminar (``plt.close(fig)``) para no ir acumulando
    figuras abiertas; el corrector automático lo verifica.
    """
    # BEGIN_TU_CODIGO
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    w_arr = np.asarray(w, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5))

    x1_min, x1_max = float(np.min(X_arr[:, 0])) - 0.5, float(np.max(X_arr[:, 0])) + 0.5
    x2_min, x2_max = float(np.min(X_arr[:, 1])) - 0.5, float(np.max(X_arr[:, 1])) + 0.5

    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 200),
        np.linspace(x2_min, x2_max, 200),
    )

    sesgo = w_arr[2] if w_arr.shape[0] >= 3 else 0.0
    Z = w_arr[0] * xx1 + w_arr[1] * xx2 + sesgo

    ax.contourf(
        xx1, xx2, np.where(Z >= 0.0, 1.0, -1.0),
        levels=[-2.0, 0.0, 2.0],
        colors=["#ffcccc", "#cce5ff"],
        alpha=0.5,
    )
    if np.min(Z) <= 0.0 <= np.max(Z) and np.min(Z) < np.max(Z):
        ax.contour(xx1, xx2, Z, levels=[0.0], colors="k", linewidths=1.5)

    pos = y_arr == 1
    neg = y_arr == -1
    ax.scatter(X_arr[pos, 0], X_arr[pos, 1], c="tab:blue", marker="o", s=60, edgecolors="k", label="Clase +1")
    ax.scatter(X_arr[neg, 0], X_arr[neg, 1], c="tab:red", marker="x", s=60, label="Clase -1")

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(titulo)
    ax.legend(loc="best")

    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    # END_TU_CODIGO


if __name__ == "__main__":
    print("Este archivo solo define funciones.")
    print("Para revisar tu trabajo:      python grader.py")
    print("Para correr los experimentos: python experimento.py")
