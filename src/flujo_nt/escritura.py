#! /usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Escritura y lectura de resultados en archivos de texto.

Los resultados se escriben como tablas de texto plano con un encabezado de
líneas de comentario, legibles con ``numpy.loadtxt``. El formato y el
esquema de nombres son los de los scripts originales de análisis:

    <base>.<cálculo>.<código de especie>.dat

donde la base es, por defecto, la ruta de la trayectoria sin su extensión,
de modo que los resultados quedan junto a ella. Por ejemplo, el flujo de
oxígenos de agua de ``sistema.1.nc`` se escribe en
``sistema.1.flujo.inst.Oa.dat``.

El encabezado se arma a partir del resultado: el cálculo y la especie, las
unidades de cada columna, los parámetros del cálculo, el origen del archivo
y, en la última línea, los nombres de las columnas.

Los resultados por cuadro, cuyas magnitudes tienen forma
``(n_cuadros, n_bins)``, se escriben en formato largo: una fila por cada
cuadro y bin, como los archivos de densidad por bins de LAMMPS.

Examples
--------
>>> base = base_desde_trayectoria('sistema.1.nc')
>>> escribir_resultado(flujo, base, fuente='sistema.1.nc')
'sistema.1.flujo.inst.Oa.dat'

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: noviembre de 2026
"""

import datetime
import os
import re

import numpy as np

from . import __version__
from .analisis import Resultado
from .especies import Especie, buscar_especie


#/ ---- CONSTANTES DEL MÓDULO --------------------------------------------

FORMATO_PREDETERMINADO = '%.5e'
"""str: Formato numérico de las columnas, el mismo de LAMMPS y de los
scripts originales."""

EXTENSION_PREDETERMINADA = 'dat'
"""str: Extensión de los archivos de resultados."""

UNIDADES_METADATOS = {
    'limites_region': 'Å',
    'radio_region': 'Å',
    'radio_tubo': 'Å',
    'volumen_region': 'Å^3',
    'masa_molecular': 'g/mol',
    'ancho_bin': 'Å',
    'bordes_bins': 'Å',
    'centros_bins': 'Å',
    'largo': 'Å',
    'radio': 'Å',
    'radio_cilindro': 'Å',
    'origen': 'Å',
    'volumen_bin': 'Å^3',
    'umbral': 'Å',
    'sigma_ref': 'Å',
    'paso_integracion': 'fs'}
"""dict: Unidades de los metadatos que las tienen, para el encabezado."""


#/ ---- NOMBRES DE ARCHIVO ------------------------------------------------

def base_desde_trayectoria(ruta_trayectoria):
    """
    Base de los nombres de archivo de resultados de una trayectoria.

    Parameters
    ----------
    ruta_trayectoria : str
        Ruta a la trayectoria.

    Returns
    -------
    str
        Ruta de la trayectoria sin su extensión.

    Examples
    --------
    >>> base_desde_trayectoria('datos/sistema.1.nc')
    'datos/sistema.1'
    """
    return os.path.splitext(str(ruta_trayectoria))[0]


def nombre_archivo(base, resultado, extension=EXTENSION_PREDETERMINADA):
    """
    Ruta del archivo en que se escribe un resultado.

    Parameters
    ----------
    base : str
        Base de los nombres de archivo, típicamente la ruta de la
        trayectoria sin extensión.
    resultado : Resultado
        Resultado a escribir.
    extension : str, optional
        Extensión del archivo. Por defecto ``'dat'``.

    Returns
    -------
    str
        ``<base>.<cálculo>.<código de especie>.<extensión>``.
    """
    return (f'{base}.{resultado.calculo}.{resultado.especie.codigo}'
            f'.{extension}')


#/ ---- ESCRITURA --------------------------------------------------------

def escribir_resultado(resultado, base, extension=EXTENSION_PREDETERMINADA,
        formato=FORMATO_PREDETERMINADO, fuente=None):
    """
    Escritura de un resultado en un archivo de texto.

    Parameters
    ----------
    resultado : Resultado
        Resultado a escribir.
    base : str
        Base de los nombres de archivo; ver :func:`nombre_archivo`.
    extension : str, optional
        Extensión del archivo. Por defecto ``'dat'``.
    formato : str, optional
        Formato numérico de las columnas que no tengan uno propio en
        ``resultado.formatos``. Por defecto ``'%.5e'``.
    fuente : str, optional
        Trayectoria de la que proviene el resultado, para dejar constancia
        en el encabezado.

    Returns
    -------
    str
        Ruta del archivo escrito.

    Notes
    -----
    El archivo se escribe en UTF-8. Si el directorio de la base no existe,
    se crea.
    """
    nombres, unidades, datos = _tabla(resultado)
    formatos = [resultado.formatos.get(nombre, formato) for nombre in nombres]

    ruta = nombre_archivo(base, resultado, extension)
    directorio = os.path.dirname(ruta)
    if directorio:
        os.makedirs(directorio, exist_ok=True)

    np.savetxt(ruta, datos, fmt=formatos, encoding='utf-8',
               header=_encabezado(resultado, nombres, unidades, fuente))

    return ruta


def _tabla(resultado):
    """
    Columnas de un resultado como tabla bidimensional.

    Las columnas unidimensionales se escriben tal cual. Si hay columnas con
    forma ``(n_cuadros, n_bins)``, la tabla pasa a formato largo: las
    columnas por cuadro se repiten para cada bin, se agrega la columna
    ``centro_bin`` a partir de los metadatos, y las columnas por cuadro y
    bin se aplanan.

    Returns
    -------
    nombres : list of str
        Nombres de las columnas de la tabla.
    unidades : dict
        Unidades de cada columna.
    datos : numpy.ndarray
        Tabla, con una columna por nombre.
    """
    columnas = resultado.columnas
    unidades = dict(resultado.unidades)
    por_bin = [nombre for nombre, valores in columnas.items()
               if np.ndim(valores) == 2]

    if not por_bin:
        nombres = list(columnas)
        datos = [np.asarray(columnas[nombre], dtype=float)
                 for nombre in nombres]
        return nombres, unidades, np.column_stack(datos)

    n_cuadros, n_bins = np.shape(columnas[por_bin[0]])
    por_cuadro = [nombre for nombre in columnas if nombre not in por_bin]

    nombres = por_cuadro + ['centro_bin'] + por_bin
    unidades['centro_bin'] = 'Å'
    datos = (
        [np.repeat(np.asarray(columnas[nombre], dtype=float), n_bins)
         for nombre in por_cuadro]
        + [np.tile(resultado.metadatos['centros_bins'], n_cuadros)]
        + [np.asarray(columnas[nombre], dtype=float).ravel()
           for nombre in por_bin])

    return nombres, unidades, np.column_stack(datos)


def _encabezado(resultado, nombres, unidades, fuente):
    """
    Texto del encabezado de un archivo de resultados.

    La primera línea identifica el cálculo y la especie, la segunda las
    unidades, y la última los nombres de las columnas. Entre ellas van los
    metadatos del resultado y el origen del archivo.
    """
    especie = resultado.especie
    lineas = [
        f'Cálculo de {resultado.calculo} para {especie.nombre} '
        f'({especie.codigo})',
        'Unidades: ' + ', '.join(
            f'{nombre} [{unidades.get(nombre, "-")}]' for nombre in nombres)]

    metadatos = resultado.metadatos
    for clave, valor in metadatos.items():
        if valor is None:
            continue

        unidad = UNIDADES_METADATOS.get(clave)
        linea = f'{clave} = {_formatear(valor)}'
        if unidad:
            linea += f' {unidad}'

        #/ El ancho de bin se acompaña de los parámetros de que surge
        if (clave == 'ancho_bin' and 'fraccion_bin_r' in metadatos
                and 'sigma_ref' in metadatos):
            linea += (f' (fraccion_bin_r {metadatos["fraccion_bin_r"]:g}'
                      f' × sigma_ref {metadatos["sigma_ref"]:g} Å)')

        lineas.append(linea)

    fecha = datetime.datetime.now().isoformat(timespec='seconds')
    lineas.append(f'Generado por flujo_nt {__version__} el {fecha}')
    if fuente is not None:
        lineas.append(f'Trayectoria: {fuente}')

    lineas.append(' '.join(nombres))

    return '\n'.join(lineas)


def _formatear(valor):
    """
    Representación de un metadato para el encabezado.
    """
    if isinstance(valor, str):
        return valor

    arreglo = np.asarray(valor)

    if arreglo.dtype.kind in 'iub':
        return ' '.join(str(int(x)) for x in arreglo.ravel())

    return ' '.join(f'{x:.6f}' for x in arreglo.ravel())


#/ ---- LECTURA ----------------------------------------------------------

def leer_resultado(ruta):
    """
    Lectura de un archivo escrito con :func:`escribir_resultado`.

    Parameters
    ----------
    ruta : str
        Ruta del archivo.

    Returns
    -------
    Resultado
        Resultado con las columnas y unidades del archivo. Las columnas son
        siempre unidimensionales: un resultado por cuadro se lee en formato
        largo, con una fila por cada cuadro y bin. Las líneas intermedias
        del encabezado quedan, sin interpretar, en
        ``metadatos['encabezado']``.

    Raises
    ------
    ValueError
        Si el encabezado no tiene el formato esperado.
    """
    with open(ruta, encoding='utf-8') as f:
        encabezado = []
        for linea in f:
            if not linea.startswith('#'):
                break
            encabezado.append(linea[1:].strip())

    if len(encabezado) < 3:
        raise ValueError(f'Encabezado incompleto en {ruta}')

    identificacion = re.fullmatch(
        r'Cálculo de (\S+) para (.+) \((\S+)\)', encabezado[0])
    if identificacion is None or not encabezado[1].startswith('Unidades:'):
        raise ValueError(f'Encabezado con formato inesperado en {ruta}')

    calculo, nombre, codigo = identificacion.groups()
    try:
        especie = buscar_especie(codigo)
    except KeyError:
        especie = Especie(0, nombre, codigo)

    unidades = dict(re.findall(r'(\S+) \[([^\]]*)\]', encabezado[1]))
    nombres = encabezado[-1].split()
    datos = np.loadtxt(ruta, ndmin=2, encoding='utf-8')

    return Resultado(
        calculo, especie,
        columnas={nombre: datos[:, i] for i, nombre in enumerate(nombres)},
        unidades=unidades,
        metadatos={'encabezado': encabezado[2:-1]})


#/ ---- FIN MÓDULO --------------------------------------------------------
