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
escritura
    Escritura y lectura de resultados en archivos de texto.

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: noviembre de 2026
"""

#/ La versión se define antes de importar los submódulos, porque el módulo de
#/ escritura la registra en los encabezados de los archivos
__version__ = '0.1.0'

from .trayectoria import (  # noqa: E402
    Bloque,
    Celda,
    Configuracion,
    Dialecto,
    ErrorTrayectoria,
    Trayectoria)
from .geometria import ErrorGeometria, MarcoTubo, Tubo  # noqa: E402
from .especies import (  # noqa: E402
    ESPECIES_AGUA_METANOL, Especie, buscar_especie)
from .analisis import (  # noqa: E402
    Analisis,
    DensidadLongitudinal,
    DensidadRadial,
    ErrorAnalisis,
    Flujo,
    PerfilVelocidad,
    Pervaporacion,
    Resultado,
    recorrer)
from .escritura import (  # noqa: E402
    base_desde_trayectoria,
    escribir_resultado,
    leer_resultado,
    nombre_archivo)


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
    'base_desde_trayectoria',
    'buscar_especie',
    'escribir_resultado',
    'leer_resultado',
    'nombre_archivo',
    'recorrer']
