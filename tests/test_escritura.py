# -*- coding:utf-8 -*-

"""
Tests del módulo flujo_nt.escritura.

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: noviembre de 2026
"""

import numpy as np
import pytest

from flujo_nt import (
    DensidadRadial,
    Flujo,
    PerfilVelocidad,
    Pervaporacion,
    base_desde_trayectoria,
    buscar_especie,
    escribir_resultado,
    leer_resultado,
    nombre_archivo,
    recorrer)


@pytest.fixture
def resultados(trayectoria, tubo_ejemplo):
    """Resultados de los datos de ejemplo, uno de cada forma."""
    agua = buscar_especie('Oa')

    flujo, densidad, serie, pervaporacion = recorrer(trayectoria, [
        Flujo(agua, tubo_ejemplo),
        DensidadRadial(agua, tubo_ejemplo),
        PerfilVelocidad(agua, tubo_ejemplo, por_cuadro=True),
        Pervaporacion(agua, tubo_ejemplo)])

    return {'flujo': flujo, 'densidad': densidad, 'serie': serie,
            'pervaporacion': pervaporacion}


def leer_encabezado(ruta):
    with open(ruta, encoding='utf-8') as f:
        return [linea for linea in f if linea.startswith('#')]


#/ ---- NOMBRES ----------------------------------------------------------

def test_nombres_de_archivo(resultados):
    """El esquema de nombres es el de los scripts originales."""
    base = base_desde_trayectoria('datos/sistema.R1-5.1.nc')

    assert base == 'datos/sistema.R1-5.1'
    assert (nombre_archivo(base, resultados['flujo'])
            == 'datos/sistema.R1-5.1.flujo.inst.Oa.dat')
    assert (nombre_archivo(base, resultados['serie'])
            == 'datos/sistema.R1-5.1.perfil.velocidad.serie.Oa.dat')


#/ ---- ESCRITURA Y LECTURA ----------------------------------------------

def test_ida_y_vuelta(resultados, tmp_path):
    flujo = resultados['flujo']
    ruta = escribir_resultado(flujo, str(tmp_path / 'sistema'),
                              fuente='sistema.nc')

    leido = leer_resultado(ruta)

    assert leido.calculo == 'flujo.inst'
    assert leido.especie is buscar_especie('Oa')
    assert leido.nombres == ['tiempo', 'n', 'Jn', 'Jm']
    assert leido.unidades == flujo.unidades
    for nombre in flujo.nombres:
        #/ El formato %.5e conserva seis cifras significativas
        np.testing.assert_allclose(leido[nombre], flujo[nombre], rtol=1e-5)
    assert 'Trayectoria: sistema.nc' in leido.metadatos['encabezado']


def test_encabezado_documenta_ancho_de_bin(resultados, tmp_path):
    ruta = escribir_resultado(resultados['densidad'], str(tmp_path / 's'))

    encabezado = ''.join(leer_encabezado(ruta))

    assert ('ancho_bin = 0.875000 Å (fraccion_bin_r 0.25 × sigma_ref 3.5 Å)'
            in encabezado)
    assert encabezado.splitlines()[-1] == (
        '# centro_bin n rho_n_media rho_n_desvio')


def test_pervaporacion_con_formato_de_lammps(resultados, tmp_path):
    """
    Dos columnas, como el archivo de fix ave/time de LAMMPS: el número de
    paso como entero y el conteo en notación exponencial.
    """
    ruta = escribir_resultado(resultados['pervaporacion'], str(tmp_path / 's'))

    with open(ruta, encoding='utf-8') as f:
        filas = [linea.split() for linea in f if not linea.startswith('#')]

    assert filas == [['0', '3.00000e+00'], ['500', '3.00000e+00']]


def test_serie_en_formato_largo(resultados, tmp_path):
    """Una fila por cuadro y bin, que se reacomoda con un reshape."""
    serie = resultados['serie']
    n_cuadros, n_bins = serie['v_media'].shape

    leido = leer_resultado(escribir_resultado(serie, str(tmp_path / 's')))

    assert leido.nombres == ['tiempo', 'centro_bin', 'n', 'v_media']
    assert leido['tiempo'].size == n_cuadros * n_bins
    np.testing.assert_allclose(
        leido['centro_bin'].reshape(n_cuadros, n_bins)[0],
        serie.metadatos['centros_bins'])
    np.testing.assert_allclose(
        leido['v_media'].reshape(n_cuadros, n_bins), serie['v_media'],
        rtol=1e-5, equal_nan=True)
    np.testing.assert_array_equal(
        leido['tiempo'].reshape(n_cuadros, n_bins)[:, 0], serie['tiempo'])


def test_crea_el_directorio_de_salida(resultados, tmp_path):
    ruta = escribir_resultado(resultados['flujo'],
                              str(tmp_path / 'nuevo' / 'sistema'))

    assert ruta == str(tmp_path / 'nuevo' / 'sistema.flujo.inst.Oa.dat')


def test_leer_archivo_ajeno_falla(tmp_path):
    ruta = tmp_path / 'ajeno.dat'
    ruta.write_text('# otra cosa\n# más\n# a b\n1 2\n', encoding='utf-8')

    with pytest.raises(ValueError, match='formato inesperado'):
        leer_resultado(str(ruta))
