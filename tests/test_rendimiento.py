# -*- coding:utf-8 -*-

"""
Test de humo del script de medición de rendimiento.

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: noviembre de 2026
"""

import subprocess
import sys

import numpy as np

from conftest import RAIZ


SCRIPT = RAIZ / 'rendimiento' / 'medir_rendimiento.py'


def test_medicion_de_rendimiento(tmp_path, rutas_ejemplo):
    salida = tmp_path / 'rendimiento.dat'
    proceso = subprocess.run(
        [sys.executable, str(SCRIPT), rutas_ejemplo[0],
         '--configuracion', rutas_ejemplo[1], '-t', '1,todo', '-r', '1',
         '-o', str(salida)],
        capture_output=True, text=True)

    assert proceso.returncode == 0, proceso.stderr

    tabla = np.loadtxt(salida, ndmin=2)
    np.testing.assert_array_equal(tabla[:, 0], [1, -1])
    assert np.all(tabla[:, 2:] > 0)
