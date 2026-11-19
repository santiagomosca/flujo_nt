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
especies
    Especies de partículas del sistema.
analisis
    Análisis de flujo, densidad, perfil de velocidad y pervaporación.

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
from .especies import ESPECIES_AGUA_METANOL, Especie, buscar_especie
from .analisis import (
    Analisis,
    DensidadLongitudinal,
    DensidadRadial,
    ErrorAnalisis,
    Flujo,
    PerfilVelocidad,
    Pervaporacion,
    Resultado,
    recorrer)

__version__ = '0.1.0'

__all__ = [
    'Analisis',
    'Bloque',
    'Celda',
    'Configuracion',
    'DensidadLongitudinal',
    'DensidadRadial',
    'Dialecto',
    'ESPECIES_AGUA_METANOL',
    'ErrorAnalisis',
    'ErrorGeometria',
    'ErrorTrayectoria',
    'Especie',
    'Flujo',
    'MarcoTubo',
    'PerfilVelocidad',
    'Pervaporacion',
    'Resultado',
    'Trayectoria',
    'Tubo',
    'buscar_especie',
    'recorrer']
