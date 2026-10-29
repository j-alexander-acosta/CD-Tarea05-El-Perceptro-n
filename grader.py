#!/usr/bin/env python3
"""Autograder de la tarea «El Perceptrón».

Se ejecuta sin conexión a internet y sin más dependencias que numpy y
matplotlib::

    python grader.py

Imprime el detalle de cada prueba, el subtotal por parte y el puntaje
final. Las pruebas que ves aquí son las mismas que puedes correr mientras
trabajas y suman 60 de los 100 puntos de la tarea. Los 40 restantes los
reparte el corrector del curso sobre esta misma entrega: pruebas adicionales
sobre las mismas cinco funciones y la revisión de la sección de respuestas
que abre `submission.py`. Toda la tarea se corrige de forma automática.
"""
from __future__ import annotations

import os
import signal
import sys

# Nada de .pyc en la carpeta de una entrega: el bytecode de submission
# es la solución de referencia en forma decompilable.
sys.dont_write_bytecode = True
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import util  # noqa: E402  (fija el backend Agg de matplotlib)
from util import (  # noqa: E402
    capa_oculta_xor,
    datos_con_margen,
    datos_tarea,
    datos_xor,
    direccion_max_margen,
    plt,
)

import submission as sub  # noqa: E402


# ----------------------------------------------------------------------
# Infraestructura de pruebas
# ----------------------------------------------------------------------
PARTES = [
    (1, "Márgenes", 8),
    (2, "La regla de actualización", 16),
    (3, "El bucle de entrenamiento", 23),
    (4, "La cota teórica de convergencia", 12),
    (5, "La frontera de decisión en un PNG", 10),
    (6, "La tabla del experimento de la cota", 16),
    (7, "El experimento de XOR", 15),
]
PUNTAJE_TOTAL = sum(puntos for _, _, puntos in PARTES)

PRUEBAS = []


def prueba(parte, descripcion, puntos, extra=False):
    """Decorador que registra una función de prueba."""

    def decorador(funcion):
        PRUEBAS.append(
            {
                "parte": parte,
                "descripcion": descripcion,
                "puntos": puntos,
                "funcion": funcion,
                "extra": extra,
            }
        )
        return funcion

    return decorador


def cerca(obtenido, esperado, tol=1e-9):
    """Compara arreglos o escalares con tolerancia absoluta."""
    return np.allclose(
        np.asarray(obtenido, dtype=float),
        np.asarray(esperado, dtype=float),
        rtol=0.0,
        atol=tol,
    )


def afirmar(condicion, mensaje):
    if not condicion:
        raise AssertionError(mensaje)


def carpeta_temporal():
    """Carpeta descartable donde las pruebas escriben sus PNG."""
    return tempfile.TemporaryDirectory(prefix="tarea-perceptron-")


def es_png(ruta):
    with open(ruta, "rb") as archivo:
        return archivo.read(8) == b"\x89PNG\r\n\x1a\n"


def capturar_figuras(llamada):
    """Ejecuta ``llamada`` y devuelve las figuras de matplotlib que creó.

    Se interceptan ``plt.figure`` y ``plt.subplots`` para poder revisar el
    contenido del gráfico incluso después de que la función bajo prueba haya
    cerrado la figura con ``plt.close(fig)``.
    """
    creadas = []
    original_figure = plt.figure
    original_subplots = plt.subplots

    def espia_figure(*args, **kwargs):
        figura = original_figure(*args, **kwargs)
        creadas.append(figura)
        return figura

    def espia_subplots(*args, **kwargs):
        resultado = original_subplots(*args, **kwargs)
        creadas.append(resultado[0])
        return resultado

    plt.figure = espia_figure
    plt.subplots = espia_subplots
    try:
        llamada()
    finally:
        plt.figure = original_figure
        plt.subplots = original_subplots
    return creadas


def artistas_de_datos(ax):
    """Cuántos conjuntos de puntos o líneas se dibujaron sobre los ejes."""
    return len(ax.collections) + len(ax.lines)


# ----------------------------------------------------------------------
# Parte 1: márgenes
# ----------------------------------------------------------------------
@prueba(1, "margenes: valores y forma del arreglo devuelto", 3)
def p1_valores():
    # Datos propios del corrector: no son los de util.py, así que esta prueba
    # comprueba la fórmula sin apoyarse en ningún conjunto de la tarea.
    X = np.array([[3.0, 1.0], [0.0, 2.0], [-2.0, 4.0]])
    y = np.array([1, -1, 1])
    obtenido = sub.margenes(X, y, np.array([1.0, -3.0]))
    afirmar(
        np.asarray(obtenido).shape == (3,),
        f"margenes debe devolver un arreglo de forma (3,); dio "
        f"{np.asarray(obtenido).shape}",
    )
    afirmar(
        cerca(obtenido, [0.0, 6.0, -14.0]),
        f"el margen del ejemplo i es (w . x_i) * y_i: sobre estos tres datos "
        f"debe dar [0, 6, -14] y dio {obtenido}",
    )


@prueba(1, "margenes: w = 0, datos con sesgo y arreglos intactos", 3)
def p1_bordes():
    X, y = datos_tarea()
    copia_X, copia_y = X.copy(), y.copy()
    obtenido = sub.margenes(X, y, np.zeros(2))
    afirmar(
        cerca(obtenido, np.zeros(5)),
        f"con w = 0 todos los márgenes valen 0; dio {obtenido}",
    )
    afirmar(cerca(X, copia_X, 0.0) and cerca(y, copia_y, 0.0),
            "margenes no debe modificar X ni y")

    Xx, yx = datos_xor()
    obtenido = sub.margenes(Xx, yx, np.array([1.0, 1.0, -0.5]))
    afirmar(
        cerca(obtenido, [0.5, 0.5, 0.5, -1.5]),
        f"los márgenes de XOR con w = [1, 1, -0,5] son [0,5, 0,5, 0,5, -1,5]; "
        f"dio {obtenido}",
    )


# ----------------------------------------------------------------------
# Parte 2: la regla de actualización
# ----------------------------------------------------------------------
@prueba(2, "paso_perceptron: positivo mal clasificado y w intacto", 4)
def p2_positivo():
    w = np.zeros(2)
    copia = w.copy()
    w_nuevo, actualizo = sub.paso_perceptron(np.array([3.0, 1.0]), 1, w)
    afirmar(bool(actualizo),
            "con margen 0 el ejemplo cuenta como error: debía actualizar")
    afirmar(
        cerca(w_nuevo, [3.0, 1.0]),
        f"w <- w + y*x debería dar [3, 1] y dio {w_nuevo}",
    )
    afirmar(
        cerca(w, copia, 0.0),
        "paso_perceptron no debe modificar w en el lugar, tampoco cuando "
        "actualiza (¿escribiste w += y * x?)",
    )


@prueba(2, "paso_perceptron: negativo mal clasificado", 4)
def p2_negativo():
    w = np.array([2.0, 3.0])
    copia = w.copy()
    w_nuevo, actualizo = sub.paso_perceptron(np.array([0.0, 2.0]), -1, w)
    afirmar(bool(actualizo), "el margen es -6 <= 0: debía actualizar")
    afirmar(
        cerca(w_nuevo, [2.0, 1.0]),
        f"w <- w - x debería dar [2, 1] y dio {w_nuevo}",
    )
    afirmar(cerca(w, copia, 0.0),
            "paso_perceptron no debe modificar w en el lugar")


@prueba(2, "paso_perceptron: ejemplo bien clasificado y w intacto", 4)
def p2_sin_actualizar():
    w = np.array([2.0, 1.0])
    copia = w.copy()
    w_nuevo, actualizo = sub.paso_perceptron(np.array([1.0, 3.0]), 1, w)
    afirmar(not actualizo, "el margen es 5 > 0: no correspondía actualizar")
    afirmar(cerca(w_nuevo, [2.0, 1.0]), f"w no debía cambiar; dio {w_nuevo}")
    afirmar(cerca(w, copia, 0.0), "paso_perceptron no debe modificar w en el lugar")
    w_nuevo[0] = 99.0
    afirmar(
        cerca(w, copia, 0.0),
        "el arreglo devuelto no debe ser el mismo objeto que w: devuelve una copia",
    )


@prueba(2, "paso_perceptron: margen cero negativo, listas y estabilidad", 4)
def p2_bordes():
    # Margen exactamente 0 con etiqueta -1: también es error.
    w_nuevo, actualizo = sub.paso_perceptron(
        np.array([1.0, -2.0]), -1, np.array([2.0, 1.0])
    )
    afirmar(bool(actualizo),
            "margen exactamente 0 con y = -1 también cuenta como error")
    afirmar(
        cerca(w_nuevo, [1.0, 3.0]),
        f"w - x debería dar [1, 3] y dio {w_nuevo}",
    )

    # Listas de Python.
    w_nuevo, actualizo = sub.paso_perceptron([1.0, 1.0], 1, [0.0, 0.0])
    afirmar(bool(actualizo) and cerca(w_nuevo, [1.0, 1.0]),
            "paso_perceptron debería aceptar listas de Python")

    # Aplicar dos veces sobre un ejemplo bien clasificado no cambia nada.
    w = np.array([3.0, 2.0])
    primero, _ = sub.paso_perceptron(np.array([3.0, 2.0]), 1, w)
    segundo, actualizo = sub.paso_perceptron(np.array([3.0, 2.0]), 1, primero)
    afirmar(not actualizo and cerca(segundo, [3.0, 2.0]),
            "un ejemplo bien clasificado nunca debería producir cambios")

    # El signo de la actualización: negativo mal clasificado resta.
    w_nuevo, _ = sub.paso_perceptron(np.array([3.0, 0.0]), -1, np.array([1.0, 0.0]))
    afirmar(
        cerca(w_nuevo, [-2.0, 0.0]),
        f"un negativo mal clasificado se resta: [-2, 0]; dio {w_nuevo}",
    )


# ----------------------------------------------------------------------
# Parte 3: el bucle de entrenamiento
# ----------------------------------------------------------------------
@prueba(3, "entrenar_perceptron: converge y cuenta bien sobre un conjunto separable", 7)
def p3_separable():
    # Conjunto propio del corrector, distinto del de datos_tarea(): ejercita
    # el margen 0, la acumulación dentro de la época y el conteo de épocas.
    X = np.array([[0.0, 3.0], [3.0, 0.0], [-1.0, -2.0]])
    y = np.array([1, -1, 1])
    r = sub.entrenar_perceptron(X, y, max_epocas=100)
    afirmar(bool(r["convergio"]), "el conjunto es separable: debía converger")
    afirmar(
        cerca(r["w"], [-4.0, 1.0]),
        f"recorriendo estos tres ejemplos en orden desde w = 0 el resultado es "
        f"[-4, 1]; dio {r['w']}",
    )
    afirmar(
        r["actualizaciones"] == 3,
        f"los tres ejemplos de la primera época actualizan; contó "
        f"{r['actualizaciones']}",
    )
    afirmar(
        r["epocas"] == 2,
        f"la última época, la que ya no actualiza, también cuenta; contó "
        f"{r['epocas']}",
    )
    m = sub.margenes(X, y, r["w"])
    afirmar(
        float(np.min(m)) > 0.0,
        f"tras converger ningún margen puede quedar en cero o negativo; dio {m}",
    )


@prueba(3, "entrenar_perceptron: contrato del diccionario y de w0", 4)
def p3_contrato():
    X, y = datos_tarea()
    r = sub.entrenar_perceptron(X, y, max_epocas=100)
    afirmar(
        isinstance(r, dict),
        f"entrenar_perceptron debe devolver un diccionario; devolvió {type(r).__name__}",
    )
    afirmar(
        set(r) == {"w", "actualizaciones", "epocas", "convergio"},
        f"las claves deben ser exactamente w, actualizaciones, epocas y "
        f"convergio; trajo {sorted(r)}",
    )
    afirmar(
        np.asarray(r["w"]).shape == (2,),
        f"w debe tener forma (2,); dio {np.asarray(r['w']).shape}",
    )

    w0 = np.array([-3.0, 4.0])
    copia = w0.copy()
    sub.entrenar_perceptron(X, y, max_epocas=100, w0=w0)
    afirmar(cerca(w0, copia, 0.0), "entrenar_perceptron no debe modificar w0")

    Xc = np.array([[0.0, 3.0], [3.0, 0.0], [-1.0, -2.0]])
    yc = np.array([1, -1, 1])
    ya_bueno = sub.entrenar_perceptron(Xc, yc, max_epocas=100,
                                       w0=np.array([-4.0, 1.0]))
    afirmar(
        ya_bueno["actualizaciones"] == 0 and ya_bueno["epocas"] == 1,
        "partiendo de una solución ya correcta debe terminar en 1 época sin "
        f"actualizaciones; dio {ya_bueno['epocas']} época(s) y "
        f"{ya_bueno['actualizaciones']} actualización(es)",
    )


@prueba(3, "entrenar_perceptron: XOR agota las épocas sin converger", 4)
def p3_xor():
    X, y = datos_xor()
    r = sub.entrenar_perceptron(X, y, max_epocas=300)
    afirmar(not r["convergio"], "XOR no es separable: no debía converger")
    afirmar(
        r["epocas"] == 300,
        f"sin converger debe agotar las 300 épocas; contó {r['epocas']}",
    )
    afirmar(
        r["actualizaciones"] >= 300,
        "sin converger cada época deja al menos una actualización; contó "
        f"{r['actualizaciones']}",
    )
    m = sub.margenes(X, y, r["w"])
    afirmar(
        float(np.min(m)) <= 0.0,
        f"al terminar debe quedar al menos un ejemplo mal clasificado; "
        f"márgenes {m}",
    )


@prueba(3, "entrenar_perceptron: XOR sí converge tras la capa oculta fija", 3)
def p3_xor_oculto():
    X, y = datos_xor()
    H = capa_oculta_xor(X)
    r = sub.entrenar_perceptron(H, y, max_epocas=500)
    afirmar(
        bool(r["convergio"]),
        "en el espacio oculto (h1, h2, 1) los datos sí son separables: "
        "debía converger",
    )
    m = sub.margenes(H, y, r["w"])
    afirmar(
        float(np.min(m)) > 0.0,
        f"tras converger todos los márgenes deben ser positivos; dio {m}",
    )


# ----------------------------------------------------------------------
# Parte 4: la cota teórica
# ----------------------------------------------------------------------
@prueba(4, "cota_actualizaciones: los cinco ejemplos de datos_tarea()", 4)
def p4_valor():
    X, y = datos_tarea()
    obtenido = sub.cota_actualizaciones(X, y, np.array([1.0, 0.0]))
    afirmar(
        cerca(obtenido, 5.0, 1e-9),
        "con w_ref = [1, 0] es R^2 = 5, gamma = 1 y la cota vale 5,0; "
        f"dio {obtenido}",
    )


@prueba(4, "cota_actualizaciones: no depende de la escala de w_ref", 3)
def p4_escala():
    X, y = datos_tarea()
    base = sub.cota_actualizaciones(X, y, np.array([1.0, 0.0]))
    for factor in (0.001, 7.0, 1000.0):
        otro = sub.cota_actualizaciones(X, y, factor * np.array([1.0, 0.0]))
        afirmar(
            cerca(otro, base, 1e-6),
            f"la cota debe ser la misma al escalar w_ref por {factor}: "
            f"{base} vs {otro} (falta dividir por ||w_ref||)",
        )
    w_ref, _ = direccion_max_margen(X, y)
    con_maxmargen = sub.cota_actualizaciones(X, y, w_ref)
    afirmar(
        cerca(con_maxmargen, 5.0, 1e-3),
        f"con la dirección de margen máximo la cota es 5,0; dio {con_maxmargen}",
    )


@prueba(4, "cota_actualizaciones: rechaza un w_ref que no separa", 3)
def p4_errores():
    X, y = datos_tarea()
    try:
        sub.cota_actualizaciones(X, y, np.zeros(2))
    except ValueError:
        pass
    else:
        afirmar(False, "con w_ref = 0 debía lanzar ValueError")

    try:
        sub.cota_actualizaciones(X, y, np.array([-1.0, 0.0]))
    except ValueError:
        pass
    else:
        afirmar(False, "con un w_ref que no separa debía lanzar ValueError")

    Xx, yx = datos_xor()
    try:
        sub.cota_actualizaciones(Xx, yx, np.array([1.0, 1.0, -0.5]))
    except ValueError:
        pass
    else:
        afirmar(False, "XOR no admite separador: debía lanzar ValueError")


# ----------------------------------------------------------------------
# Parte 5: el gráfico
# ----------------------------------------------------------------------
@prueba(5, "graficar_frontera: PNG válido, con puntos, ejes, título y leyenda", 4)
def p5_archivo():
    X, y = datos_tarea()
    titulo = "Prueba del corrector"
    with carpeta_temporal() as carpeta:
        ruta = os.path.join(carpeta, "frontera.png")
        devuelto = []
        figuras = capturar_figuras(
            lambda: devuelto.append(
                sub.graficar_frontera(X, y, np.array([5.0, -2.0]), ruta,
                                      titulo=titulo)
            )
        )
        afirmar(devuelto[0] is None, "graficar_frontera no debe devolver nada")
        afirmar(os.path.exists(ruta), f"no se creó el archivo {ruta}")
        afirmar(es_png(ruta), "el archivo creado no es un PNG")
        afirmar(
            os.path.getsize(ruta) > 3000,
            f"el PNG pesa solo {os.path.getsize(ruta)} bytes: parece vacío",
        )
        afirmar(
            bool(figuras),
            "no se creó ninguna figura de matplotlib: ¿de dónde salió el PNG?",
        )
        ejes = figuras[-1].axes
        afirmar(bool(ejes), "la figura quedó sin ejes: no se dibujó nada")
        ax = ejes[0]
        afirmar(
            any(titulo in eje.get_title() for eje in ejes),
            f"el gráfico debe llevar el título que recibe la función; el eje "
            f"trae «{ax.get_title()}»",
        )
        afirmar(
            bool(ax.get_xlabel().strip()) and bool(ax.get_ylabel().strip()),
            "faltan los rótulos de los ejes (x_1 y x_2)",
        )
        afirmar(
            ax.get_legend() is not None,
            "falta la leyenda que distingue las dos clases",
        )
        afirmar(
            artistas_de_datos(ax) >= 2,
            "el gráfico está prácticamente vacío: se esperan al menos los "
            "puntos de cada clase y la frontera, y se contaron "
            f"{artistas_de_datos(ax)} elemento(s) dibujado(s)",
        )


@prueba(5, "graficar_frontera: backend Agg y sin figuras abiertas", 2)
def p5_sin_ventanas():
    import matplotlib

    afirmar(
        matplotlib.get_backend().lower() == "agg",
        f"el backend debe ser Agg y es {matplotlib.get_backend()}",
    )
    plt.close("all")
    X, y = datos_tarea()
    with carpeta_temporal() as carpeta:
        for i in range(3):
            sub.graficar_frontera(X, y, np.array([5.0, -2.0]),
                                  os.path.join(carpeta, f"f{i}.png"))
    abiertas = plt.get_fignums()
    afirmar(
        not abiertas,
        f"quedaron {len(abiertas)} figuras abiertas: falta plt.close(fig)",
    )


@prueba(5, "graficar_frontera: datos con columna de sesgo", 2)
def p5_con_sesgo():
    X, y = datos_xor()
    with carpeta_temporal() as carpeta:
        ruta = os.path.join(carpeta, "xor.png")
        figuras = capturar_figuras(
            lambda: sub.graficar_frontera(X, y, np.array([1.0, 1.0, -0.5]),
                                          ruta, titulo="XOR")
        )
        afirmar(os.path.exists(ruta) and es_png(ruta),
                "con X de 3 columnas debe graficar igual, usando w[2] como sesgo")
        afirmar(bool(figuras) and bool(figuras[-1].axes),
                "la figura quedó sin ejes: no se dibujó nada")
        afirmar(
            artistas_de_datos(figuras[-1].axes[0]) >= 2,
            "con tres columnas el gráfico también debe traer los puntos de "
            "cada clase y la frontera",
        )

        H = capa_oculta_xor(X)
        ruta2 = os.path.join(carpeta, "oculto.png")
        sub.graficar_frontera(H, y, np.array([2.0, -2.0, -1.0]), ruta2)
        afirmar(os.path.exists(ruta2) and es_png(ruta2),
                "debe funcionar también sobre la representación oculta")


@prueba(5, "graficar_frontera: el gráfico depende de w y no se pisa", 2)
def p5_depende_de_w():
    plt.close("all")
    X, y = datos_tarea()
    with carpeta_temporal() as carpeta:
        uno = os.path.join(carpeta, "uno.png")
        otro = os.path.join(carpeta, "otro.png")
        # El mismo título en las dos llamadas: si los archivos difieren, la
        # diferencia solo puede venir de la frontera, que es justamente lo
        # que se quiere comprobar.
        sub.graficar_frontera(X, y, np.array([5.0, -2.0]), uno, titulo="igual")
        sub.graficar_frontera(X, y, np.array([-5.0, 2.0]), otro, titulo="igual")
        afirmar(os.path.exists(uno) and os.path.exists(otro),
                "dos llamadas deben dejar dos archivos distintos")
        afirmar(es_png(uno) and es_png(otro), "ambos deben ser PNG válidos")
        with open(uno, "rb") as a, open(otro, "rb") as b:
            afirmar(
                a.read() != b.read(),
                "el gráfico no cambia al invertir w: ¿se está dibujando la "
                "frontera o solo los puntos?",
            )

        # Frontera vertical: w[1] = 0 no debe romper el gráfico.
        vertical = os.path.join(carpeta, "vertical.png")
        sub.graficar_frontera(X, y, np.array([1.0, 0.0]), vertical)
        afirmar(os.path.exists(vertical) and es_png(vertical),
                "con w = [1, 0] la frontera es vertical y debe graficarse igual")

    afirmar(not plt.get_fignums(), "quedaron figuras abiertas")


# ----------------------------------------------------------------------
# Ejecución
# ----------------------------------------------------------------------
TOPE_SEGUNDOS = 60

# El temporizador se apoya en SIGALRM, que existe en Linux y macOS pero no en
# Windows. Si no está disponible, el corrector funciona igual, solo que sin
# tope de tiempo.
HAY_TEMPORIZADOR = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")


class TiempoAgotado(Exception):
    """Se levanta cuando una prueba supera el tope de tiempo."""


def _sono_la_alarma(_numero, _marco):
    raise TiempoAgotado


def ejecutar_una(caso):
    """Corre una prueba y devuelve (puntos obtenidos, detalle del error).

    Cada prueba tiene un tope de ``TOPE_SEGUNDOS``: un bucle de entrenamiento
    escrito como ``while True`` dejaría al corrector colgado para siempre, y en
    una corrección por lotes una sola entrega así bloquearía todo lo demás.
    """
    anterior = None
    if HAY_TEMPORIZADOR:
        anterior = signal.signal(signal.SIGALRM, _sono_la_alarma)
        signal.setitimer(signal.ITIMER_REAL, TOPE_SEGUNDOS)
    try:
        caso["funcion"]()
    except TiempoAgotado:
        return 0, (f"la prueba superó los {TOPE_SEGUNDOS} s: "
                   "¿el bucle de entrenamiento respeta max_epocas?")
    except NotImplementedError:
        return 0, "sin implementar todavía"
    except AssertionError as exc:
        return 0, str(exc)
    except Exception as exc:  # noqa: BLE001 — el grader nunca debe caerse
        return 0, f"{type(exc).__name__}: {exc}"
    finally:
        if HAY_TEMPORIZADOR:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, anterior)
    return caso["puntos"], ""


def main():
    print("=" * 70)
    print(" Autograder — Tarea: El Perceptrón")
    print("=" * 70)

    obtenido_total = 0
    maximo_total = 0
    hay_extra = any(c["extra"] for c in PRUEBAS)

    for numero, titulo, _ in PARTES:
        casos = [c for c in PRUEBAS if c["parte"] == numero]
        if not casos:
            continue
        print(f"\nParte {numero}: {titulo}")
        obtenido_parte = maximo_parte = 0
        for caso in casos:
            puntos, detalle = ejecutar_una(caso)
            obtenido_parte += puntos
            maximo_parte += caso["puntos"]
            marca = "OK   " if puntos == caso["puntos"] else "FALLA"
            sufijo = "  (adicional)" if caso["extra"] else ""
            print(f"  [{marca}] {caso['descripcion']}{sufijo}"
                  f"  ->  {puntos}/{caso['puntos']}")
            if detalle:
                print(f"           {detalle}")
        print(f"  Subtotal parte {numero}: {obtenido_parte}/{maximo_parte}")
        obtenido_total += obtenido_parte
        maximo_total += maximo_parte

    print("\n" + "=" * 70)
    if hay_extra:
        print(f" PUNTAJE DE LA TAREA: {obtenido_total}/{maximo_total}")
    else:
        print(f" PUNTAJE EN LAS PRUEBAS VISIBLES: {obtenido_total}/{maximo_total}")
        print(f" La tarea vale {PUNTAJE_TOTAL} puntos: los "
              f"{PUNTAJE_TOTAL - maximo_total} restantes los reparte el")
        print(" corrector del curso sobre esta misma entrega, entre pruebas")
        print(" adicionales sobre las mismas cinco funciones y la revisión de la")
        print(" sección de respuestas de submission.py.")
    print("=" * 70)
    return 0 if obtenido_total == maximo_total else 1


if __name__ == "__main__":
    sys.exit(main())
