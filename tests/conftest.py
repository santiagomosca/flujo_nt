# -*- coding:utf-8 -*-

"""
Piezas compartidas por los tests de flujo_nt.

Hay dos fuentes de datos:

- Los datos de ejemplo de ``datos/ejemplo``: dos cuadros de una trayectoria
  real con su archivo de configuración. Los tests que los usan se omiten si
  los archivos no están. Los valores de referencia de la biblioteca
  original y de LAMMPS para esos dos cuadros están en ``tests/*.json``.
- Sistemas sintéticos construidos en el momento: trayectorias netCDF
  pequeñas y tubos de geometría conocida, para los casos que los datos reales
  no ejercitan.

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: noviembre de 2026
"""

import json
import pathlib

import numpy as np
import pytest
from netCDF4 import Dataset

from flujo_nt import Celda, Configuracion, Trayectoria, Tubo


RAIZ = pathlib.Path(__file__).resolve().parents[1]
DIRECTORIO_EJEMPLO = RAIZ / 'datos' / 'ejemplo'
NOMBRE_EJEMPLO = 'reservorios_R1-5_L37-5_X0-115_Elong-1eneg5_398K'
TRAYECTORIA_EJEMPLO = DIRECTORIO_EJEMPLO / f'{NOMBRE_EJEMPLO}_2cuadros.nc'
CONFIGURACION_EJEMPLO = DIRECTORIO_EJEMPLO / f'{NOMBRE_EJEMPLO}_final.data'
REFERENCIA_ORIGINAL = (
    pathlib.Path(__file__).parent / 'referencia_original.json')
REFERENCIA_LAMMPS = pathlib.Path(__file__).parent / 'referencia_lammps.json'


#/ ---- DATOS DE EJEMPLO --------------------------------------------------

@pytest.fixture(scope='session')
def rutas_ejemplo():
    """Rutas a la trayectoria y la configuración de ejemplo."""
    faltantes = [ruta.name for ruta in (TRAYECTORIA_EJEMPLO,
                                        CONFIGURACION_EJEMPLO)
                 if not ruta.is_file()]
    if faltantes:
        pytest.skip(f'Faltan datos de ejemplo en {DIRECTORIO_EJEMPLO}: '
                    f'{faltantes}')

    return str(TRAYECTORIA_EJEMPLO), str(CONFIGURACION_EJEMPLO)


@pytest.fixture(scope='session')
def configuracion(rutas_ejemplo):
    """Configuración de ejemplo."""
    return Configuracion(rutas_ejemplo[1])


@pytest.fixture
def trayectoria(rutas_ejemplo, configuracion):
    """Trayectoria de ejemplo abierta, con la celda corregida."""
    with Trayectoria(rutas_ejemplo[0], configuracion=configuracion) as tr:
        yield tr


@pytest.fixture(scope='session')
def tubo_ejemplo(rutas_ejemplo, configuracion):
    """Tubo medido en el primer cuadro de la trayectoria de ejemplo."""
    with Trayectoria(rutas_ejemplo[0], configuracion=configuracion) as tr:
        return Tubo.desde_trayectoria(tr)


@pytest.fixture(scope='session')
def referencia():
    """Valores obtenidos con la biblioteca original."""
    return json.loads(REFERENCIA_ORIGINAL.read_text())


@pytest.fixture(scope='session')
def referencia_lammps():
    """Valores escritos por LAMMPS durante la corrida."""
    return json.loads(REFERENCIA_LAMMPS.read_text())


#/ ---- TRAYECTORIAS SINTÉTICAS -------------------------------------------

def escribir_netcdf(ruta, n_cuadros=3, n_atomos=4, programa='LAMMPS',
        tipos=None, identificadores=None, longitudes=None, origen=None,
        posiciones=None, velocidades=None,
        nombre_tipos='type', nombre_identificadores='id',
        con_fuerzas=False, variables_extra=()):
    """
    Escritura de una trayectoria netCDF mínima con convención AMBER.

    Los argumentos no dados toman valores válidos: tipos y orden constantes,
    identificadores crecientes y celda constante no degenerada. Las
    velocidades sólo se escriben si se dan.
    """
    if tipos is None:
        tipos = np.tile(np.arange(n_atomos) % 2 + 1, (n_cuadros, 1))
    if identificadores is None:
        identificadores = np.tile(np.arange(1, n_atomos + 1), (n_cuadros, 1))
    if longitudes is None:
        longitudes = np.tile([10.0, 10.0, 20.0], (n_cuadros, 1))
    if origen is None:
        origen = np.zeros((n_cuadros, 3))
    if posiciones is None:
        posiciones = np.arange(n_cuadros * n_atomos * 3).reshape(
            n_cuadros, n_atomos, 3)

    with Dataset(ruta, 'w', format='NETCDF3_64BIT_DATA') as ds:
        ds.Conventions = 'AMBER'
        if programa is not None:
            ds.program = programa

        ds.createDimension('frame', None)
        ds.createDimension('atom', n_atomos)
        ds.createDimension('spatial', 3)
        ds.createDimension('cell_spatial', 3)

        tiempo = ds.createVariable('time', 'f4', ('frame',))
        tiempo[:] = np.arange(n_cuadros) * 500.0

        ds.createVariable(
            'coordinates', 'f4', ('frame', 'atom', 'spatial'))[:] = \
            posiciones

        if velocidades is not None:
            ds.createVariable(
                'velocities', 'f4', ('frame', 'atom', 'spatial'))[:] = \
                velocidades

        ds.createVariable(nombre_tipos, 'i4', ('frame', 'atom'))[:] = tipos
        ds.createVariable(
            nombre_identificadores, 'i4', ('frame', 'atom'))[:] = \
            identificadores
        ds.createVariable(
            'cell_lengths', 'f4', ('frame', 'cell_spatial'))[:] = longitudes
        ds.createVariable(
            'cell_origin', 'f4', ('frame', 'cell_spatial'))[:] = origen

        if con_fuerzas:
            ds.createVariable(
                'forces', 'f4', ('frame', 'atom', 'spatial'))[:] = 0.0

        for nombre in variables_extra:
            ds.createVariable(nombre, 'f4', ('frame', 'atom'))[:] = 0.0

    return str(ruta)


@pytest.fixture
def crear_netcdf(tmp_path):
    """Fábrica de trayectorias sintéticas en un directorio temporal."""
    def crear(nombre='sintetica.nc', **opciones):
        return escribir_netcdf(tmp_path / nombre, **opciones)

    return crear


#/ ---- TUBOS SINTÉTICOS --------------------------------------------------

def tubo_sintetico(radio=5.5, largo=130.0, flecha=1.5, ruido=0.05,
        semilla=0):
    """
    Átomos de un tubo curvo de geometría conocida, con dos paredes cosidas.

    El tubo es una red regular sobre una superficie cilíndrica de eje Z,
    cuyo eje se curva como una parábola en X con desplazamiento ``flecha``
    en los extremos. Las paredes son planos en z = ±largo/2 que cubren la
    sección de la celda.

    Returns
    -------
    atomos : numpy.ndarray
        Posiciones, con forma ``(n, 3)``.
    celda : Celda
        Celda de 64 x 64 x 500 Å, centrada en el origen.
    eje_x : callable
        Posición en X del eje verdadero en función de z.
    """
    generador = np.random.default_rng(semilla)

    def eje_x(z):
        return flecha * (2 * z / largo) ** 2

    #/ Un átomo por distancia de enlace sobre la circunferencia, filas
    #/ alternadas para imitar la red del grafeno
    n_angulos = int(round(2 * np.pi * radio / 1.42))
    filas = np.arange(-largo / 2 * 0.98, largo / 2 * 0.98, 0.71)
    angulos = np.concatenate([
        2 * np.pi * (np.arange(n_angulos) + 0.5 * (i % 2)) / n_angulos
        for i in range(filas.size)])
    z = np.repeat(filas, n_angulos)

    tubo = np.stack([eje_x(z) + radio * np.cos(angulos),
                     radio * np.sin(angulos),
                     z], axis=1)
    tubo += generador.normal(0, ruido, tubo.shape)

    grilla = np.linspace(-31.5, 31.5, 45)
    x, y = np.meshgrid(grilla, grilla)
    plano = np.stack([x.ravel(), y.ravel()], axis=1)
    plano = plano[np.hypot(plano[:, 0], plano[:, 1]) > radio + 1]
    paredes = np.vstack([
        np.column_stack([plano, np.full(len(plano), signo * largo / 2)])
        for signo in (-1, 1)])

    celda = Celda([64.0, 64.0, 500.0], [-32.0, -32.0, -250.0])

    return np.vstack([tubo, paredes]), celda, eje_x


@pytest.fixture
def fabrica_tubo():
    """Fábrica de tubos sintéticos de geometría conocida."""
    return tubo_sintetico
