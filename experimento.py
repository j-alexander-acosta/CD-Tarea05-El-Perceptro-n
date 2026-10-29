#!/usr/bin/env python3
"""Cierre de la tarea: corre los tres experimentos y guarda los gráficos.

    python experimento.py

No hay que modificar este archivo. Usa las funciones que completaste en
``submission.py``, así que solo va a funcionar cuando el corrector ya te dé
puntaje en las partes 3, 4 y 5.

De su salida salen las constantes de la sección de respuestas de
``submission.py``: la tabla del primer experimento y los números de XOR del
tercero.

Deja además tres archivos PNG en esta misma carpeta:

* ``frontera-separable.png``   — los cinco ejemplos de ``datos_tarea()``,
* ``frontera-xor.png``         — XOR después del tope de épocas,
* ``frontera-xor-oculta.png``  — XOR visto desde la capa oculta fija.
"""
from __future__ import annotations

import sys

# Nada de .pyc en la carpeta de una entrega: el bytecode de submission
# es la solución de referencia en forma decompilable.
sys.dont_write_bytecode = True

import numpy as np

import submission as sub
from util import (
    capa_oculta_xor,
    datos_con_margen,
    datos_tarea,
    datos_xor,
    direccion_max_margen,
)

LINEA = "=" * 70


def experimento_separable():
    """Los cinco ejemplos del enunciado: converge y respeta la cota."""
    print(LINEA)
    print("2. Conjunto separable (los cinco ejemplos de datos_tarea)")
    print(LINEA)

    X, y = datos_tarea()
    resultado = sub.entrenar_perceptron(X, y, max_epocas=100)
    w_ref, gamma = direccion_max_margen(X, y)
    cota = sub.cota_actualizaciones(X, y, w_ref)
    radio = float(np.max(np.linalg.norm(X, axis=1)))

    print(f"  w final           = {np.round(resultado['w'], 4)}")
    print(f"  convergió         = {resultado['convergio']}")
    print(f"  épocas            = {resultado['epocas']}")
    print(f"  actualizaciones   = {resultado['actualizaciones']}")
    print(f"  R (ejemplo más largo) = {radio:.4f}")
    print(f"  gamma (margen máximo) = {gamma:.4f}")
    print(f"  cota (R/gamma)^2      = {cota:.2f}")
    print(f"  márgenes finales  = {np.round(sub.margenes(X, y, resultado['w']), 4) + 0.0}")

    sub.graficar_frontera(
        X, y, resultado["w"], "frontera-separable.png",
        titulo="Conjunto separable: frontera aprendida",
    )
    print("  -> frontera-separable.png")
    print()


def experimento_margen():
    """La cantidad de actualizaciones en función del margen."""
    print(LINEA)
    print("1. Actualizaciones frente a la cota, según el margen")
    print(LINEA)
    print(f"  {'margen':>8} {'R':>8} {'actualiz.':>10} {'cota':>12} {'cota/real':>10}")

    for margen in (0.5, 0.25, 0.1, 0.05):
        X, y = datos_con_margen(margen, n=120, semilla=2026)
        resultado = sub.entrenar_perceptron(X, y, max_epocas=5000)
        w_ref, _ = direccion_max_margen(X, y)
        cota = sub.cota_actualizaciones(X, y, w_ref)
        radio = float(np.max(np.linalg.norm(X, axis=1)))
        reales = resultado["actualizaciones"]
        razon = cota / reales if reales else float("inf")
        print(f"  {margen:>8.2f} {radio:>8.3f} {reales:>10d} {cota:>12.1f} {razon:>10.1f}")

    print()
    print("  Todas las corridas convergen; la cota nunca se viola.")
    print()


def experimento_xor():
    """XOR: el perceptrón no converge, pero sí lo hace tras la capa oculta."""
    print(LINEA)
    print("3. XOR: el límite del clasificador lineal")
    print(LINEA)

    X, y = datos_xor()
    tope = 2000
    resultado = sub.entrenar_perceptron(X, y, max_epocas=tope)
    print(f"  con {tope} épocas de tope:")
    print(f"    convergió       = {resultado['convergio']}")
    print(f"    épocas          = {resultado['epocas']}")
    print(f"    actualizaciones = {resultado['actualizaciones']}")
    print(f"    w final         = {np.round(resultado['w'], 4)}")
    print(f"    márgenes        = {np.round(sub.margenes(X, y, resultado['w']), 4) + 0.0}")
    print("    (al menos un margen sigue siendo <= 0: siempre queda un error)")

    print("  ¿y si le damos más épocas?")
    for tope_extra in (4000, 8000):
        otro = sub.entrenar_perceptron(X, y, max_epocas=tope_extra)
        print(f"    max_epocas={tope_extra:>5d} -> convergió = {otro['convergio']}, "
              f"actualizaciones = {otro['actualizaciones']}")

    print()
    print("  Los pesos dentro de una época, paso a paso:")
    w = np.zeros(3)
    for vuelta in range(2):
        for fila, etiqueta in zip(X, y):
            w, actualizo = sub.paso_perceptron(fila, etiqueta, w)
            marca = "actualiza" if actualizo else "  --     "
            print(f"    x = ({fila[0]:.0f}, {fila[1]:.0f}, 1)  y = {etiqueta:+d}   "
                  f"{marca}   w = {np.round(w, 2)}")
        print(f"    --- fin de la época {vuelta + 1} ---")

    print()
    ilustrativa = np.array([1.0, 1.0, -0.5])
    print(f"  La mejor recta posible acierta 3 de 4. Por ejemplo w = {ilustrativa}:")
    print(f"    márgenes = {np.round(sub.margenes(X, y, ilustrativa), 4)}")
    sub.graficar_frontera(
        X, y, ilustrativa, "frontera-xor.png",
        titulo="XOR: ninguna recta acierta los cuatro puntos",
    )
    print("  -> frontera-xor.png")

    print()
    print("  Ahora los mismos cuatro puntos, vistos desde una capa oculta fija")
    print("  de dos unidades de umbral con sesgos -0,5 y -1,5, escrita a mano:")
    H = capa_oculta_xor(X)
    for fila, salida, etiqueta in zip(X, H, y):
        print(f"    x = ({fila[0]:.0f}, {fila[1]:.0f})  ->  "
              f"h = ({salida[0]:.0f}, {salida[1]:.0f})   y = {etiqueta:+d}")

    oculto = sub.entrenar_perceptron(H, y, max_epocas=2000)
    print(f"    convergió       = {oculto['convergio']}")
    print(f"    épocas          = {oculto['epocas']}")
    print(f"    actualizaciones = {oculto['actualizaciones']}")
    print(f"    w final         = {np.round(oculto['w'], 4)}")

    sub.graficar_frontera(
        H, y, oculto["w"], "frontera-xor-oculta.png",
        titulo="XOR visto por la capa oculta: ejes $(h_1, h_2)$",
    )
    print("  -> frontera-xor-oculta.png")
    print()
    print("  El mismo algoritmo, los mismos datos, otra representación: converge.")
    print("  Las dos unidades ocultas las escribimos a mano. Aprenderlas a")
    print("  partir de los datos ya no es trabajo de la regla del perceptrón:")
    print("  esa es la puerta de entrada a la tarea siguiente.")
    print()


def main():
    # El orden importa: el experimento del margen produce la tabla de la
    # sección de respuestas y no depende del gráfico, así que va primero. Cada
    # experimento se aísla, de modo que una función todavía sin implementar no
    # se lleve por delante a los demás.
    experimentos = (
        ("actualizaciones frente a la cota", experimento_margen),
        ("conjunto separable", experimento_separable),
        ("XOR", experimento_xor),
    )
    for nombre, experimento in experimentos:
        try:
            experimento()
        except NotImplementedError:
            print()
            print("  [!] Falta implementar alguna de las funciones que usa el")
            print(f"      experimento «{nombre}»: se omite y el resto sigue")
            print("      corriendo.")
            print()

    print(LINEA)
    print("Listo. Los números de arriba van en la sección de respuestas")
    print("de submission.py.")
    print(LINEA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
