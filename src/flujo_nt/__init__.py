"""
flujo_nt: análisis de trayectorias de flujo a través de nanotubos.

Biblioteca para analizar trayectorias de dinámica molecular de mezclas que
fluyen a través de nanotubos de carbono, generadas con LAMMPS en formato
netCDF (convención AMBER).

Módulos
-------
trayectoria
    Lectura perezosa de trayectorias y archivos de configuración.
geometria
    Marco de referencia del tubo, medición del tubo curvo y corrección de
    posiciones por su curvatura.

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: noviembre de 2026
"""

from .trayectoria import (
    Bloque,
    Celda,
    Configuracion,
    Dialecto,
    ErrorTrayectoria,
    Trayectoria)
from .geometria import ErrorGeometria, MarcoTubo, Tubo

__version__ = '0.1.0'

__all__ = [
    'Bloque',
    'Celda',
    'Configuracion',
    'Dialecto',
    'ErrorGeometria',
    'ErrorTrayectoria',
    'MarcoTubo',
    'Trayectoria',
    'Tubo']
