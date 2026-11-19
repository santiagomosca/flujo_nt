#! /usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Generación de los valores de referencia de la biblioteca original.

Ejecuta las funciones de la biblioteca original (versión de diciembre de 2025)
sobre los datos de ejemplo, siguiendo los mismos pasos que los scripts
originales de flujo, densidad radial, perfil de velocidad y pervaporación, y
guarda los resultados en un archivo JSON. Los tests
de flujo_nt comparan contra ese archivo, de modo que no dependen de la
biblioteca original para ejecutarse.

Sólo hace falta volver a ejecutarlo si cambian los datos de ejemplo.

Uso, desde la raíz del repositorio:
    python legado/generar_referencia.py

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: noviembre de 2026
"""

import contextlib
import io
import json
import os
import sys

import numpy as np

DIRECTORIO = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(DIRECTORIO)
sys.path.insert(0, DIRECTORIO)

import analisis_netcdf_depurado_diciembre_2025 as anc  # noqa: E402

#/ Alias de las funciones originales cuyo nombre no entra en una línea
densidad_radial_original = \
    anc.calcular_densidad_numerica_radial_tubo_curvo_interpolado_por_cuadro
perfil_velocidad_original = \
    anc.calcular_perfil_radial_velocidad_tubo_curvo_interpolado_por_cuadro_v2

NOMBRE_EJEMPLO = 'reservorios_R1-5_L37-5_X0-115_Elong-1eneg5_398K'
TRAYECTORIA = os.path.join(
    RAIZ, 'datos', 'ejemplo', f'{NOMBRE_EJEMPLO}_2cuadros.nc')
CONFIGURACION = os.path.join(
    RAIZ, 'datos', 'ejemplo', f'{NOMBRE_EJEMPLO}_final.data')
SALIDA = os.path.join(RAIZ, 'tests', 'referencia_original.json')

FRACCION_REGION = 0.9
FRACCION_RADIO = 1.0
FRACCION_BIN_R = 0.25
SIGMA_REF = 3.5
TIPOS_FLUJO = {1: 18.0154, 3: 32.0416}
TIPOS_DENSIDAD = (1, 2, 3, 4, 5)


def lista(arreglo):
    """Conversión a lista de flotantes, apta para JSON."""
    return np.asarray(arreglo, dtype=float).tolist()


def main():
    #/ Se silencia la salida de la biblioteca original, que imprime cada paso
    with contextlib.redirect_stdout(io.StringIO()):
        trayectoria = anc.leer_trayectoria(TRAYECTORIA)
        variables = trayectoria['variables']

        longitudes = anc.obtener_longitudes_celda_desde_configuracion(
            CONFIGURACION)
        for indice, direccion in enumerate('xyz'):
            variables['cell_lengths']['data'][:, indice] = \
                longitudes[direccion]

        eje = anc.detectar_eje_tubo(variables['cell_lengths'])

        radio, largo, radios_anillos, centros_anillos, bordes_anillos = \
            anc.obtener_dimensiones_tubo(
                variables['type'],
                variables['coordinates'],
                variables['cell_origin'],
                variables['cell_lengths'],
                None,
                None,
                1.5,
                37.5,
                eje_tubo=eje,
                tubo_curvo=True)

        posiciones = variables['coordinates']
        velocidades = variables['velocities']
        tipos = variables['type']
        limites = [-largo / 2, largo / 2]
        comunes = dict(fraccion_region=FRACCION_REGION,
                       fraccion_radio=FRACCION_RADIO, eje_tubo=eje)
        radiales = dict(comunes, fraccion_bin_r=FRACCION_BIN_R,
                        sigma_ref=SIGMA_REF, volumenes_iguales=False)

        #/ Pasos de flujo_sin_rotacion.py
        conteos, flujo = {}, {}
        for tipo, masa in TIPOS_FLUJO.items():
            conteo, numerico, masico = anc.calcular_flujo(
                posiciones, velocidades, tipos, tipo, masa, limites, radio,
                centros_anillos, bordes_anillos, **comunes)[:3]
            conteos[str(tipo)] = conteo.tolist()
            flujo[str(tipo)] = {
                'conteo': conteo.tolist(),
                'flujo_numerico': lista(numerico),
                'flujo_masico': lista(masico)}

        #/ Pasos de densidad_radial_curvo_por_cuadro_sin_rotacion.py
        densidad_radial = {}
        for tipo in TIPOS_DENSIDAD:
            total, media, desvio, bordes_r = densidad_radial_original(
                posiciones, tipos, tipo, limites, radio,
                centros_anillos, bordes_anillos, **radiales)[:4]
            densidad_radial[str(tipo)] = {
                'conteo': lista(total), 'media': lista(media),
                'desvio': lista(desvio), 'bordes': lista(bordes_r)}

        #/ Pasos de perfil_velocidad_curvo_por_cuadro.py, con la variante
        #/ de acumulación rápida
        perfil = {}
        for tipo in TIPOS_FLUJO:
            conteo, media, desvio = perfil_velocidad_original(
                posiciones, tipos, velocidades, tipo, limites, radio,
                centros_anillos, bordes_anillos, **radiales)[:3]
            perfil[str(tipo)] = {
                'conteo': lista(conteo), 'media': lista(media),
                'desvio': lista(desvio)}

        #/ Pasos de conteo_pervaporacion.py
        pervaporacion = {
            str(tipo): anc.contar_particulas_pervaporadas(
                posiciones, tipos, tipo, largo,
                omitir_primer_cuadro=False, eje_tubo=eje).tolist()
            for tipo in TIPOS_FLUJO}

    referencia = {
        'descripcion': (
            'Valores obtenidos con analisis_netcdf_depurado_diciembre_2025.py '
            'sobre los datos de ejemplo, con los parámetros predeterminados '
            'de obtener_dimensiones_tubo y los pasos de '
            'flujo_sin_rotacion.py. '
            'Generado por legado/generar_referencia.py.'),
        'trayectoria': os.path.basename(TRAYECTORIA),
        'eje': eje,
        'largo': float(largo),
        'radio': float(radio),
        'bordes_anillos': np.asarray(bordes_anillos, dtype=float).tolist(),
        'radios_anillos': np.asarray(radios_anillos, dtype=float).tolist(),
        'centros_anillos': np.asarray(centros_anillos, dtype=float).tolist(),
        'fraccion_region': FRACCION_REGION,
        'fraccion_radio': FRACCION_RADIO,
        'fraccion_bin_r': FRACCION_BIN_R,
        'sigma_ref': SIGMA_REF,
        'conteos_en_tubo': conteos,
        'flujo': flujo,
        'densidad_radial': densidad_radial,
        'perfil_velocidad': perfil,
        'pervaporacion': pervaporacion}

    with open(SALIDA, 'w') as f:
        json.dump(referencia, f, indent=2)

    print(f'Escrito {SALIDA}')
    print(f'\teje {eje}, largo {largo:.6f} Å, radio {radio:.6f} Å, '
          f'{len(radios_anillos)} anillos')
    print(f'\tconteos en tubo: {conteos}')
    print(f'\tpervaporación: {pervaporacion}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
