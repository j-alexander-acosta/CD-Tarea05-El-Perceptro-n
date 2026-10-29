"""Utilidades de la tarea «El Perceptrón». No hay que modificar este archivo.

Aquí viven los conjuntos de datos (todos generados localmente: la tarea no
descarga nada) y un par de rutinas de apoyo. La única sutileza es el backend
de matplotlib: se fija en ``Agg`` antes de importar ``pyplot`` para que los
gráficos se guarden en archivos PNG y nunca se abra una ventana. Por eso
conviene importar ``plt`` desde este módulo::

    from util import plt
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # sin ventanas: los gráficos solo se guardan en disco

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = [
    "plt",
    "datos_tarea",
    "datos_con_margen",
    "datos_xor",
    "capa_oculta_xor",
    "direccion_max_margen",
]


def datos_tarea():
    """El conjunto pequeño de cinco ejemplos en 2D con que se ilustra la regla.

    Es lo bastante pequeño para seguirlo paso a paso con lápiz y papel, y el
    orden de las filas importa: recorrerlas al revés lleva a otra frontera,
    igual de válida. Las clases se separan con un hiperplano que pasa por el
    origen, de modo que no hace falta sesgo: ``X`` tiene solo dos columnas.

    Devuelve ``(X, y)`` con ``X`` de forma ``(5, 2)`` y ``y`` de forma ``(5,)``
    con valores en ``{-1, +1}``.
    """
    X = np.array(
        [
            [1.0, 1.0],
            [-1.0, 2.0],
            [2.0, -1.0],
            [-2.0, -1.0],
            [1.0, -2.0],
        ]
    )
    y = np.array([1, -1, 1, -1, 1])
    return X, y


def datos_con_margen(margen, n=120, semilla=0):
    """Conjunto separable en 2D con margen geométrico exactamente ``margen``.

    Los puntos caen dentro del disco unitario, así que ``R = max ||x_i||`` es
    prácticamente 1, y quedan a distancia al menos ``margen`` de la recta que
    pasa por el origen y es perpendicular a una dirección fija. La distancia
    se sortea concentrada cerca de esa banda, de modo que el problema sea
    genuinamente difícil y no un conjunto con dos nubes bien despegadas.

    Además se agregan dos puntos, uno por clase, a distancia *exactamente*
    ``margen``: eso fija el margen máximo del conjunto en ``margen`` y no más.

    Sirve para ver cómo cambia la cantidad de actualizaciones del perceptrón
    cuando el problema se vuelve más difícil (margen más pequeño).

    Devuelve ``(X, y)`` con ``X`` de forma ``(n, 2)``.
    """
    if not 0.0 < margen < 0.9:
        raise ValueError("el margen debe estar entre 0 y 0,9")
    if n < 4:
        raise ValueError("se necesitan al menos 4 puntos")

    rng = np.random.default_rng(semilla)
    normal = np.array([np.cos(0.6), np.sin(0.6)])  # perpendicular al separador
    tangente = np.array([-normal[1], normal[0]])

    puntos = [margen * normal, -margen * normal]  # los dos que fijan el margen
    while len(puntos) < n:
        distancia = margen * (1.0 + rng.exponential(0.35))
        if rng.random() < 0.5:
            distancia = -distancia
        candidato = distancia * normal + rng.uniform(-1.0, 1.0) * tangente
        if np.linalg.norm(candidato) > 1.0:
            continue
        puntos.append(candidato)

    X = np.array(puntos)
    X = X[rng.permutation(len(X))]  # que los dos puntos críticos no vayan primero
    y = np.where(X @ normal >= 0.0, 1, -1)
    return X, y


def datos_xor():
    """La tabla de verdad de XOR en el espacio aumentado.

    ``X`` tiene tres columnas: ``x1``, ``x2`` y una columna de unos (el truco
    del sesgo como un peso más que se vio en clases). Las etiquetas son
    ``+1`` para ``x1 XOR x2 = 1`` y ``-1`` en caso contrario.
    """
    X = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    y = np.array([-1, 1, 1, -1])
    return X, y


def capa_oculta_xor(X):
    """Aplica una capa oculta FIJA de dos unidades de umbral a los datos XOR.

    Las dos unidades no se aprenden: están escritas a mano y calculan

    * ``h1 = 1[x1 + x2 - 0,5 >= 0]``,
    * ``h2 = 1[x1 + x2 - 1,5 >= 0]``.

    Recibe ``X`` de forma ``(n, 2)`` o ``(n, 3)`` (en cuyo caso la tercera
    columna es la de unos y se ignora) y devuelve la representación oculta
    ``(h1, h2, 1)`` de forma ``(n, 3)``, lista para entrenar un perceptrón
    encima.
    """
    X = np.asarray(X, dtype=float)
    suma = X[:, 0] + X[:, 1]
    h1 = (suma - 0.5 >= 0.0).astype(float)
    h2 = (suma - 1.5 >= 0.0).astype(float)
    return np.column_stack([h1, h2, np.ones(len(X))])


def direccion_max_margen(X, y, iteraciones=200000):
    """Dirección de margen máximo del conjunto (si es linealmente separable).

    Resuelve el problema equivalente de encontrar el punto de norma mínima de
    la envoltura convexa de ``z_i = y_i x_i``, con el algoritmo de
    Mitchell-Dem'yanov-Malozemov (una variante de Frank-Wolfe que mueve peso
    de un vértice a otro y converge rápido). El punto óptimo ``p`` cumple que
    ``p / ||p||`` es la dirección que maximiza el margen geométrico, y ese
    margen máximo vale ``||p||``.

    Devuelve ``(w_ref, gamma)``. Si el conjunto no es separable, ``gamma``
    sale prácticamente en cero: no hay dirección que deje todo bien
    clasificado.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Z = X * y[:, None]

    pesos = np.zeros(len(Z))
    pesos[0] = 1.0
    p = Z[0].copy()

    for _ in range(iteraciones):
        productos = Z @ p
        menor = int(np.argmin(productos))
        activos = pesos > 1e-14
        mayor = int(np.argmax(np.where(activos, productos, -np.inf)))
        brecha = float(productos[mayor] - productos[menor])
        if brecha <= 1e-14:
            break
        direccion = Z[menor] - Z[mayor]
        norma2 = float(direccion @ direccion)
        if norma2 < 1e-18:
            break
        paso = float(np.clip(brecha / norma2, 0.0, pesos[mayor]))
        if paso <= 1e-16:
            break
        pesos[menor] += paso
        pesos[mayor] -= paso
        p = p + paso * direccion

    return p, float(np.linalg.norm(p))
