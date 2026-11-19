#! /usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Extracción de valores de referencia calculados por LAMMPS durante la corrida.

Lee los archivos que escriben los comandos fix ave/chunk (densidad numérica
por bins: *.numdens) y fix ave/time (conteo de pervaporación: *.npervap) del
input in.reservorios.nemd.campo_electrico, y guarda en un archivo JSON sólo
los pasos que coinciden con los cuadros de los datos de ejemplo. Los tests de
flujo_nt comparan contra ese archivo, de modo que no hace falta versionar los
archivos completos de LAMMPS, que pesan decenas de megabytes.

Sólo hace falta volver a ejecutarlo si cambian los datos de ejemplo.

Uso, desde la raíz del repositorio:
    python legado/extraer_referencia_lammps.py DIRECTORIO PREFIJO \\
        --largo 130.64 --radio 5.491724

donde DIRECTORIO contiene los archivos PREFIJO.Oa.izq.numdens,
PREFIJO.Oa.tubo.numdens, PREFIJO.Oa.der.numdens y PREFIJO.Oa.npervap, y
--largo y --radio son los valores de v__l y v__r con que corrió LAMMPS.

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: noviembre de 2026
"""

import argparse
import json
import os
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(RAIZ, 'tests', 'referencia_lammps.json')

#/ Pasos de integración de los cuadros de los datos de ejemplo
PASOS = (0, 500)
REGIONES = ('izq', 'tubo', 'der')


def leer_ave_chunk(ruta, pasos):
    """
    Bloques de un archivo de fix ave/chunk para los pasos pedidos.

    Cada bloque empieza con una línea 'paso cantidad_de_bins total' seguida
    de una fila por bin. Devuelve un diccionario de paso a arreglo de filas.
    """
    bloques = {}

    with open(ruta) as f:
        for linea in f:
            if linea.startswith('#'):
                continue

            paso, n_bins = (int(c) for c in linea.split()[:2])
            filas = [next(f).split() for _ in range(n_bins)]

            if paso in pasos:
                bloques[paso] = np.array(filas, dtype=float)

            if len(bloques) == len(pasos):
                break

    faltantes = set(pasos) - set(bloques)
    if faltantes:
        raise ValueError(f'Faltan los pasos {sorted(faltantes)} en {ruta}')

    return bloques


def leer_ave_time(ruta, pasos):
    """
    Valores de un archivo de fix ave/time para los pasos pedidos.
    """
    datos = np.loadtxt(ruta, comments='#')
    valores = {int(paso): valor for paso, valor in datos[:, :2]}

    return [float(valores[paso]) for paso in pasos]


def main():
    parser = argparse.ArgumentParser(
        description='Extracción de referencias de LAMMPS para los tests.')
    parser.add_argument('directorio',
        help='directorio con los archivos de LAMMPS')
    parser.add_argument('prefijo',
        help='prefijo común de los archivos (sin el código de especie)')
    parser.add_argument('--codigo', default='Oa',
        help="código de la especie (predeterminado: 'Oa')")
    parser.add_argument('--largo', type=float, required=True,
        help='largo del tubo con que corrió LAMMPS (v__l), en Å')
    parser.add_argument('--radio', type=float, required=True,
        help='radio del tubo con que corrió LAMMPS (v__r), en Å')
    opciones = parser.parse_args()

    base = os.path.join(opciones.directorio,
                        f'{opciones.prefijo}.{opciones.codigo}')

    especie = {}
    for region in REGIONES:
        bloques = leer_ave_chunk(f'{base}.{region}.numdens', PASOS)

        #/ En el cilindro del tubo, la primera coordenada es la radial y la
        #/ segunda la longitudinal
        columna = 2 if region == 'tubo' else 1

        especie[region] = {
            'centro': bloques[PASOS[0]][:, columna].tolist(),
            'conteo': [bloques[paso][:, -2].tolist() for paso in PASOS],
            'densidad': [bloques[paso][:, -1].tolist() for paso in PASOS]}

    especie['pervaporacion'] = leer_ave_time(f'{base}.npervap', PASOS)

    referencia = {
        'descripcion': (
            'Valores escritos por LAMMPS durante la corrida, con el input '
            'in.reservorios.nemd.campo_electrico, para los pasos de los '
            'cuadros de los datos de ejemplo. Generado por '
            'legado/extraer_referencia_lammps.py.'),
        'prefijo': opciones.prefijo,
        'pasos': list(PASOS),
        'largo': opciones.largo,
        'radio': opciones.radio,
        'factor_radio': 1.5,
        'numero_bins': 32,
        opciones.codigo: especie}

    with open(SALIDA, 'w') as f:
        json.dump(referencia, f, indent=2)

    print(f'Escrito {SALIDA}')
    for region in REGIONES:
        print(f'\t{region}: {len(especie[region]["centro"])} bins')
    print(f'\tpervaporación: {especie["pervaporacion"]}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
