# -*- coding:utf-8 -*-

"""
Tests del recorte de trayectorias (módulo y subcomando recortar).

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: noviembre de 2026
"""

import numpy as np
import pytest
from netCDF4 import Dataset

from flujo_nt import Trayectoria
from flujo_nt.cli import main
from flujo_nt.recorte import recortar_trayectoria


def test_recorte_conserva_datos_y_escala(tmp_path, rutas_ejemplo,
        configuracion):
    """El recorte copia los datos crudos: el tiempo conserva su factor de
    escala y no se divide dos veces por él."""
    salida = tmp_path / 'recortada.nc'
    recortar_trayectoria(rutas_ejemplo[0], str(salida), 1)

    with Trayectoria(rutas_ejemplo[0], configuracion=configuracion) as orig, \
            Trayectoria(str(salida), configuracion=configuracion) as nueva:
        assert nueva.n_cuadros == 1
        assert nueva.paso_integracion == orig.paso_integracion
        np.testing.assert_array_equal(nueva.tiempos, orig.tiempos[:1])
        np.testing.assert_array_equal(nueva.cuadro(0).posiciones,
                                      orig.cuadro(0).posiciones)


def test_recorte_excluye_variables(tmp_path, rutas_ejemplo):
    salida = tmp_path / 'recortada.nc'
    inexistentes = recortar_trayectoria(
        rutas_ejemplo[0], str(salida), 1,
        excluir=['velocities', 'Velocity Magnitude'])

    assert inexistentes == ['Velocity Magnitude']
    with Dataset(salida) as ds:
        assert 'velocities' not in ds.variables
        assert 'coordinates' in ds.variables


@pytest.mark.parametrize('cuadros', [0, 3])
def test_recorte_rechaza_cantidad_invalida(tmp_path, rutas_ejemplo, cuadros):
    with pytest.raises(ValueError):
        recortar_trayectoria(rutas_ejemplo[0], str(tmp_path / 'x.nc'),
                             cuadros)


def test_recorte_rechaza_sobrescribir_la_entrada(rutas_ejemplo):
    with pytest.raises(ValueError):
        recortar_trayectoria(rutas_ejemplo[0], rutas_ejemplo[0], 1)


def test_subcomando_recortar(capsys, tmp_path, rutas_ejemplo):
    salida = tmp_path / 'recortada.nc'

    assert main(['recortar', rutas_ejemplo[0], str(salida), '-n', '1',
                 '-x', 'forces', 'Velocity Magnitude', '-q']) == 0

    capturado = capsys.readouterr()
    assert capturado.out == ''
    assert "'Velocity Magnitude'" in capturado.err
    with Dataset(salida) as ds:
        assert ds.dimensions['frame'].size == 1
        assert 'forces' not in ds.variables


def test_subcomando_recortar_listar(capsys, rutas_ejemplo):
    assert main(['recortar', rutas_ejemplo[0], '-l']) == 0
    assert 'Dimensiones:' in capsys.readouterr().out


def test_subcomando_recortar_sin_salida(rutas_ejemplo):
    with pytest.raises(SystemExit) as salida:
        main(['recortar', rutas_ejemplo[0]])
    assert salida.value.code == 2
