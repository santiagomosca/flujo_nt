#! /usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Medición de tiempo y memoria de flujo_nt según el tamaño de bloque.

Para cada tamaño de bloque pedido se ejecutan sobre una trayectoria los
mismos análisis (flujo, densidad radial, perfil de velocidad y
pervaporación, con las especies por defecto del comando flujo-nt) y se
miden el tiempo real, el tiempo de CPU y el pico de memoria.

Cada medición corre en un proceso propio, porque el pico de memoria de un
proceso no baja nunca: medir varios tamaños en el mismo proceso daría
siempre el pico del mayor. Además se mide una base, un proceso que sólo
abre la trayectoria y mide el tubo, para separar el costo fijo (intérprete,
bibliotecas, metadatos) del que depende del tamaño de bloque.

El tamaño ``todo`` procesa la trayectoria en un solo bloque, es decir, con
todos los cuadros en memoria a la vez.

Sobre los tiempos: antes de medir se hace una ejecución de calentamiento que
no se registra, de modo que todas las mediciones encuentren la trayectoria
en el mismo estado del caché de archivos del sistema operativo. Si la
trayectoria cabe en ese caché, los tiempos medidos reflejan el cálculo y no
la lectura del disco.

Ejemplos de uso:
    ./medir_rendimiento.py sistema.1.nc
    ./medir_rendimiento.py sistema.1.nc -t 1,10,50,200,todo -r 3 \\
        -o rendimiento.dat
    ./medir_rendimiento.py sistema.1.nc -n 100 -t 10,50

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: noviembre de 2026
"""

import argparse
import datetime
import json
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np

import flujo_nt
from flujo_nt import (
    Configuracion,
    DensidadRadial,
    Flujo,
    PerfilVelocidad,
    Pervaporacion,
    Trayectoria,
    Tubo,
    buscar_especie,
    recorrer)


#/ ---- CONSTANTES DEL PROGRAMA ------------------------------------------

TAMANIOS_PREDETERMINADOS = '1,10,50,200,todo'
OXIGENOS = ('Oa', 'Om')
TODAS = ('Oa', 'Ha', 'Om', 'Cm', 'Hm')


#/ ---- MEDICIÓN DE UNA EJECUCIÓN (en un proceso propio) -----------------

def memoria_maxima_mb():
    """
    Pico de memoria residente del proceso actual, en MB.

    ``ru_maxrss`` está en kB en Linux y en bytes en macOS.
    """
    maximo = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**2 if sys.platform == 'darwin' else 1024
    return maximo / divisor


def construir_analisis(tubo):
    """
    Análisis de la medición: los cálculos por defecto de flujo-nt más la
    pervaporación, con las especies por defecto de cada uno.
    """
    oxigenos = [buscar_especie(codigo) for codigo in OXIGENOS]
    todas = [buscar_especie(codigo) for codigo in TODAS]

    return ([Flujo(especie, tubo) for especie in oxigenos]
            + [DensidadRadial(especie, tubo) for especie in todas]
            + [PerfilVelocidad(especie, tubo) for especie in oxigenos]
            + [Pervaporacion(especie, tubo) for especie in oxigenos])


def medir_una(ruta, ruta_configuracion, tamanio, cuadros):
    """
    Ejecución medida de los análisis con un tamaño de bloque.

    Parameters
    ----------
    ruta : str
        Trayectoria.
    ruta_configuracion : str or None
        Configuración.
    tamanio : int or None
        Cuadros por bloque. Si es None, un solo bloque con todos los
        cuadros. Si es 0, sólo se abre la trayectoria y se mide el tubo
        (medición base).
    cuadros : int or None
        Cantidad de cuadros a procesar desde el inicio. Si es None, todos.

    Returns
    -------
    dict
        Tiempo real y de CPU en s, pico de memoria en MB, cuadros
        procesados y átomos de la trayectoria.
    """
    inicio_real = time.perf_counter()
    inicio_cpu = time.process_time()

    configuracion = (None if ruta_configuracion is None
                     else Configuracion(ruta_configuracion))

    with Trayectoria(ruta, configuracion=configuracion) as tr:
        fin = tr.n_cuadros if cuadros is None else min(cuadros,
                                                       tr.n_cuadros)
        tubo = Tubo.desde_trayectoria(tr)

        if tamanio != 0:
            recorrer(tr, construir_analisis(tubo),
                     tamanio_bloque=fin if tamanio is None else tamanio,
                     fin=fin)

        n_atomos = tr.n_atomos

    return {'tiempo_real': time.perf_counter() - inicio_real,
            'tiempo_cpu': time.process_time() - inicio_cpu,
            'memoria_max': memoria_maxima_mb(),
            'cuadros': fin,
            'atomos': n_atomos}


#/ ---- ORQUESTACIÓN ------------------------------------------------------

def lanzar(ruta, ruta_configuracion, tamanio, cuadros):
    """
    Medición en un proceso nuevo.

    Returns
    -------
    dict or None
        Resultado de :func:`medir_una`, o None si el proceso falló (por
        ejemplo, si el sistema lo terminó por falta de memoria).
    str
        Mensaje de error, vacío si no hubo falla.
    """
    argumentos = [sys.executable, os.path.abspath(__file__), ruta,
                  '--medir-una', 'todo' if tamanio is None else str(tamanio)]
    if ruta_configuracion is not None:
        argumentos += ['--configuracion', ruta_configuracion]
    if cuadros is not None:
        argumentos += ['-n', str(cuadros)]

    proceso = subprocess.run(argumentos, capture_output=True, text=True)

    if proceso.returncode != 0:
        if proceso.returncode < 0:
            mensaje = (f'terminado por la señal {-proceso.returncode} '
                       f'(9: probablemente falta de memoria)')
        else:
            ultimas = proceso.stderr.strip().splitlines()[-1:]
            mensaje = f'código {proceso.returncode}: {" ".join(ultimas)}'
        return None, mensaje

    return json.loads(proceso.stdout.strip().splitlines()[-1]), ''


def leer_tamanios(texto):
    """Lista de tamaños de bloque; ``todo`` se representa con None."""
    tamanios = []
    for elemento in texto.split(','):
        elemento = elemento.strip()
        if elemento == 'todo':
            tamanios.append(None)
            continue
        try:
            valor = int(elemento)
        except ValueError:
            valor = 0
        if valor <= 0:
            raise argparse.ArgumentTypeError(
                f'tamaño de bloque inválido: {elemento!r}')
        tamanios.append(valor)
    return tamanios


def escribir_tabla(ruta_salida, filas, ruta, ruta_configuracion, base):
    """Escritura de las mediciones como tabla de texto con encabezado."""
    fecha = datetime.datetime.now().isoformat(timespec='seconds')
    encabezado = [
        'Rendimiento de flujo_nt según el tamaño de bloque',
        'Unidades: tamanio_bloque [cuadros], repeticion [-], '
        'tiempo_real [s], tiempo_cpu [s], memoria_max [MB]',
        'tamanio_bloque = -1 indica un solo bloque con todos los cuadros; '
        'NaN, una ejecución fallida',
        f'base: tiempo_real = {base["tiempo_real"]:.3f} s, memoria_max = '
        f'{base["memoria_max"]:.1f} MB (abrir la trayectoria y medir el '
        f'tubo)',
        f'cuadros = {base["cuadros"]}, atomos = {base["atomos"]}',
        f'Trayectoria: {ruta}',
        f'Configuración: {ruta_configuracion}',
        f'Máquina: {platform.node()} ({platform.platform()}), Python '
        f'{platform.python_version()}, numpy {np.__version__}',
        f'Generado por flujo_nt {flujo_nt.__version__} el {fecha}',
        'tamanio_bloque repeticion tiempo_real tiempo_cpu memoria_max']

    datos = np.array([[-1 if tamanio is None else tamanio, repeticion,
                       *(np.nan if medicion is None else medicion[clave]
                         for clave in ('tiempo_real', 'tiempo_cpu',
                                       'memoria_max'))]
                      for tamanio, repeticion, medicion in filas])

    np.savetxt(ruta_salida, datos, fmt=['%d', '%d', '%.4f', '%.4f', '%.2f'],
               header='\n'.join(encabezado), encoding='utf-8')


def opciones():
    parser = argparse.ArgumentParser(
        description='Medición de tiempo y memoria de flujo_nt según el '
                    'tamaño de bloque. Cada medición corre en un proceso '
                    'propio.')
    parser.add_argument('trayectoria', help='trayectoria netCDF')
    parser.add_argument(
        '--configuracion', default=None,
        help='archivo de configuración; por defecto <base>.final.data si '
             'existe')
    parser.add_argument(
        '-t', '--tamanios', type=leer_tamanios,
        default=leer_tamanios(TAMANIOS_PREDETERMINADOS),
        help='tamaños de bloque separados por comas; "todo" es un solo '
             f'bloque (por defecto: {TAMANIOS_PREDETERMINADOS})')
    parser.add_argument(
        '-r', '--repeticiones', type=int, default=3,
        help='mediciones por tamaño (por defecto: %(default)s)')
    parser.add_argument(
        '-n', '--cuadros', type=int, default=None,
        help='procesar sólo los primeros N cuadros (por defecto: todos)')
    parser.add_argument(
        '-o', '--salida', default=None,
        help='archivo de texto donde guardar la tabla de mediciones')
    parser.add_argument('--medir-una', default=None, help=argparse.SUPPRESS)

    return parser.parse_args()


def main():
    opts = opciones()

    ruta_configuracion = opts.configuracion
    if ruta_configuracion is None:
        candidata = os.path.splitext(opts.trayectoria)[0] + '.final.data'
        if os.path.isfile(candidata):
            ruta_configuracion = candidata

    #/ Modo interno: una sola medición, informada como JSON
    if opts.medir_una is not None:
        tamanio = None if opts.medir_una == 'todo' else int(opts.medir_una)
        print(json.dumps(medir_una(opts.trayectoria, ruta_configuracion,
                                   tamanio, opts.cuadros)))
        return 0

    print(f'Trayectoria: {opts.trayectoria}')
    print(f'Configuración: {ruta_configuracion}')

    #/ Calentamiento: deja la trayectoria en el mismo estado del caché de
    #/ archivos para todas las mediciones
    print('Calentamiento (no se registra)...', flush=True)
    calentamiento, error = lanzar(opts.trayectoria, ruta_configuracion, 50,
                                  opts.cuadros)
    if calentamiento is None:
        print(f'error: {error}', file=sys.stderr)
        return 1

    base, error = lanzar(opts.trayectoria, ruta_configuracion, 0,
                         opts.cuadros)
    if base is None:
        print(f'error: {error}', file=sys.stderr)
        return 1

    print(f'{base["cuadros"]} cuadros, {base["atomos"]} átomos')
    print(f'Base (abrir y medir el tubo): {base["tiempo_real"]:.2f} s, '
          f'{base["memoria_max"]:.1f} MB\n')
    print(f'{"bloque":>8s} {"rep":>4s} {"real [s]":>10s} {"CPU [s]":>10s} '
          f'{"memoria [MB]":>13s}')

    filas = []
    for tamanio in opts.tamanios:
        etiqueta = 'todo' if tamanio is None else str(tamanio)
        for repeticion in range(1, opts.repeticiones + 1):
            medicion, error = lanzar(opts.trayectoria, ruta_configuracion,
                                     tamanio, opts.cuadros)
            filas.append((tamanio, repeticion, medicion))

            if medicion is None:
                print(f'{etiqueta:>8s} {repeticion:4d}  falló: {error}',
                      flush=True)
                break
            print(f'{etiqueta:>8s} {repeticion:4d} '
                  f'{medicion["tiempo_real"]:10.2f} '
                  f'{medicion["tiempo_cpu"]:10.2f} '
                  f'{medicion["memoria_max"]:13.1f}', flush=True)

    if opts.salida is not None:
        escribir_tabla(opts.salida, filas, opts.trayectoria,
                       ruta_configuracion, base)
        print(f'\nEscrito {opts.salida}')

    return 0


#/ ---- EJECUCIÓN DEL SCRIPT ----------------------------------------------
if __name__ == '__main__':
    sys.exit(main())


#/ ---- FIN SCRIPT --------------------------------------------------------
