# Tarea: El Perceptrón

Ciencia de Datos — Universidad Católica del Norte

## Archivos

| Archivo | Qué es |
|---|---|
| `enunciado.pdf` | El enunciado completo. Léelo primero. |
| `submission.py` | **El único archivo que debes editar y entregar.** Trae la sección de respuestas al comienzo y las cinco funciones debajo. |
| `util.py` | Conjuntos de datos y utilidades. No se modifica. |
| `experimento.py` | Cierre de la tarea: corre los tres experimentos, imprime los números de la sección de respuestas y guarda los gráficos. |
| `grader.py` | Corrector automático. Corre sin conexión a internet. |

## Antes de empezar

`COMO-TRABAJAR.pdf` trae las instrucciones generales: qué instalar, cómo abrir
una terminal en esta carpeta, qué significa el puntaje del corrector y los
errores más frecuentes. Sirve para todas las tareas del curso.

Un detalle que conviene saber de entrada: en Windows el intérprete se llama
`python` (o `py`), y en Linux y macOS suele ser `python3`. Los comandos de acá
usan `python`: si tu sistema responde «command not found», prueba con
`python3`.

## Cómo trabajar

Basta con Python 3, numpy y matplotlib:

```
pip install -r requirements.txt
```

Cada vez que completes una parte, corre el corrector desde esta carpeta:

```
python grader.py
```

El corrector que viene con la tarea comprueba una parte del puntaje: 60 de los
100 puntos, y con el código resuelto reporta 60 de 60. Llegar a ese total es
condición necesaria y no suficiente. La nota la pone el corrector del curso,
que corre esas mismas pruebas, otras sobre las mismas cinco funciones y la
revisión de la sección de respuestas.

Con el código listo, ejecuta el cierre:

```
python experimento.py
```

De su salida salen los números de la sección de respuestas. Deja además tres
archivos PNG en esta carpeta (`frontera-separable.png`, `frontera-xor.png` y
`frontera-xor-oculta.png`): son trabajo tuyo para entender la tarea y no se
entregan.

## Entrega

Un solo archivo: `submission.py`, con las cinco funciones completadas y la
sección de respuestas llena. Los 100 puntos de la tarea se corrigen de forma
automática, sin revisión a mano. No hay que subir datos ni gráficos: todo se
genera localmente con semillas fijas.
