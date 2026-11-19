# -*- coding:utf-8 -*-

"""
Tests del módulo flujo_nt.trayectoria.

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: noviembre de 2026
"""

import numpy as np
import pytest
from netCDF4 import Dataset

from flujo_nt import Configuracion, ErrorTrayectoria, Trayectoria


#/ ---- CONFIGURACIÓN -----------------------------------------------------

def test_configuracion_lee_encabezado(configuracion):
    assert configuracion.n_atomos == 101112
    assert configuracion.n_tipos == 7
    assert configuracion.masas[1] == pytest.approx(15.9994)
    assert configuracion.masas[6] == pytest.approx(12.0106)
    assert configuracion.celda.longitudes == pytest.approx(
        [63.947316, 63.9, 522.66])
    assert configuracion.celda.origen[2] == pytest.approx(-326.65)


def test_configuracion_inexistente_falla(tmp_path):
    with pytest.raises(ErrorTrayectoria):
        Configuracion(str(tmp_path / 'no_existe.data'))


#/ ---- CELDA -------------------------------------------------------------

def test_celda_degenerada_sin_configuracion_falla(rutas_ejemplo):
    """LAMMPS escribe longitud nula en z con kspace_modify slab."""
    with pytest.raises(ErrorTrayectoria, match='slab'):
        Trayectoria(rutas_ejemplo[0])


def test_celda_corregida_desde_configuracion(trayectoria):
    assert not trayectoria.celda.degenerada
    assert trayectoria.celda.longitudes[2] == pytest.approx(522.66)
    #/ El origen se conserva de la trayectoria
    assert trayectoria.celda.origen[2] == pytest.approx(-326.65, abs=1e-4)


#/ ---- METADATOS --------------------------------------------------------

def test_dialecto_y_dimensiones(trayectoria):
    assert trayectoria.dialecto.nombre == 'LAMMPS'
    assert trayectoria.n_cuadros == 2
    assert trayectoria.n_atomos == 101112


def test_tiempos_aplican_scale_factor(trayectoria):
    """
    Regresión: la variable time de LAMMPS guarda el número de paso y trae
    scale_factor igual al paso de integración (2 fs). Con un volcado cada
    500 pasos, sin aplicar el factor se obtendrían números de paso (0, 500)
    en lugar de tiempos (0, 1000 fs).
    """
    assert trayectoria.tiempos == pytest.approx([0.0, 1000.0])


def test_pasos_de_integracion(trayectoria, crear_netcdf):
    """El número de paso es el que usa LAMMPS en sus archivos de salida."""
    assert trayectoria.paso_integracion == pytest.approx(2.0)
    np.testing.assert_array_equal(trayectoria.pasos, [0, 500])

    with Trayectoria(crear_netcdf()) as tr:
        np.testing.assert_array_equal(tr.pasos, [0, 250, 500])


def test_tipos_presentes(trayectoria):
    assert trayectoria.tipos_presentes == {
        1: 27493, 2: 54986, 3: 3575, 4: 3575, 5: 3575, 6: 4788, 7: 3120}


def test_memoria_por_cuadro(trayectoria):
    memoria = trayectoria.memoria_por_cuadro()
    assert memoria['posiciones'] == 101112 * 3 * 4
    assert sum(memoria.values()) / 1024**2 == pytest.approx(5.79, abs=0.01)


#/ ---- LECTURA POR BLOQUES ------------------------------------------------

def test_posiciones_son_ndarray_sin_mascara(trayectoria):
    posiciones = trayectoria.cuadro(0).posiciones
    assert type(posiciones) is np.ndarray
    assert posiciones.dtype == np.float32
    assert posiciones.shape == (1, 101112, 3)


def test_bloque_lee_solo_lo_pedido(trayectoria):
    """La pereza se verifica sobre el caché interno del bloque."""
    bloque = next(trayectoria.bloques())
    assert bloque._cache == {}

    bloque.posiciones
    assert set(bloque._cache) == {'posiciones'}

    #/ El segundo acceso devuelve el mismo objeto, sin releer
    assert bloque.posiciones is bloque.posiciones


def test_imagenes_se_ensamblan_desde_ix_iy_iz(rutas_ejemplo, trayectoria):
    imagenes = trayectoria.cuadro(1).imagenes

    with Dataset(rutas_ejemplo[0]) as ds:
        esperadas = np.stack(
            [ds.variables[n][1] for n in ('ix', 'iy', 'iz')], axis=-1)

    assert imagenes.shape == (1, 101112, 3)
    np.testing.assert_array_equal(imagenes[0], esperadas)


@pytest.mark.parametrize('tamanio', [1, 2, 3, 7, 50])
def test_bloques_cubren_sin_huecos(crear_netcdf, tamanio):
    with Trayectoria(crear_netcdf(n_cuadros=7)) as tr:
        bloques = list(tr.bloques(tamanio=tamanio))
        indices = np.concatenate([b.indices for b in bloques])

    np.testing.assert_array_equal(indices, np.arange(7))
    assert all(len(b) <= tamanio for b in bloques)


@pytest.mark.parametrize('opciones', [
    {'tamanio': 0},
    {'inicio': 5, 'fin': 3},
    {'fin': 99}])
def test_bloques_rango_invalido_falla(crear_netcdf, opciones):
    with Trayectoria(crear_netcdf(n_cuadros=7)) as tr:
        with pytest.raises(ErrorTrayectoria):
            list(tr.bloques(**opciones))


def test_cuadro_fuera_de_rango_falla(trayectoria):
    with pytest.raises(ErrorTrayectoria):
        trayectoria.cuadro(2)


#/ ---- SELECCIÓN DE PARTÍCULAS -------------------------------------------

def test_mascara_tipo(trayectoria):
    mascara = trayectoria.mascara_tipo(1)
    assert mascara.shape == (101112,)
    assert mascara.sum() == 27493
    #/ Queda cacheada
    assert trayectoria.mascara_tipo(1) is mascara


def test_mascara_tipos(trayectoria):
    assert trayectoria.mascara_tipos((6, 7)).sum() == 4788 + 3120


def test_mascara_tipo_ausente_falla(trayectoria):
    with pytest.raises(ErrorTrayectoria, match='99'):
        trayectoria.mascara_tipo(99)


#/ ---- VARIABLES Y DIALECTOS ---------------------------------------------

def test_variable_ausente(crear_netcdf):
    with Trayectoria(crear_netcdf(con_fuerzas=False)) as tr:
        assert 'fuerzas' not in tr.variables_disponibles
        with pytest.raises(ErrorTrayectoria, match='fuerzas'):
            tr.cuadro(0).fuerzas


def test_dialecto_ovito(crear_netcdf):
    ruta = crear_netcdf(
        programa='OVITO',
        paso_integracion=None,
        nombre_tipos='atom_types',
        nombre_identificadores='identifier',
        variables_extra=('Velocity Magnitude',))

    with Trayectoria(ruta) as tr:
        assert tr.dialecto.nombre == 'OVITO'
        np.testing.assert_array_equal(tr.tipos, [1, 2, 1, 2])
        assert 'identificadores' in tr.variables_disponibles
        assert tr.paso_integracion is None and tr.pasos is None


@pytest.mark.parametrize('nombre_tipos, dialecto', [
    ('type', 'LAMMPS'),
    ('atom_types', 'OVITO')])
def test_dialecto_por_nombres_de_variables(crear_netcdf, nombre_tipos,
        dialecto):
    """Sin atributo program, el dialecto se deduce de las variables."""
    ruta = crear_netcdf(programa=None, nombre_tipos=nombre_tipos)

    with Trayectoria(ruta, verificar=False) as tr:
        assert tr.dialecto.nombre == dialecto


def test_dialecto_desconocido_falla(crear_netcdf):
    ruta = crear_netcdf(programa=None, nombre_tipos='especies')

    with pytest.raises(ErrorTrayectoria, match='dialecto'):
        Trayectoria(ruta)


#/ ---- INVARIANTES -------------------------------------------------------

@pytest.mark.parametrize('opciones, mensaje', [
    ({'tipos': [[1, 1, 6, 6], [1, 1, 6, 6], [1, 6, 1, 6]]},
     'tipos de partícula cambian'),
    ({'identificadores': np.tile([3, 1, 4, 2], (3, 1))},
     'no están ordenados'),
    ({'identificadores': [[1, 2, 3, 4], [1, 2, 3, 4], [2, 1, 3, 4]]},
     'orden de las partículas cambia'),
    ({'longitudes': [[10, 10, 20], [10, 10, 21], [10, 10, 22]]},
     'no es constante')],
    ids=['tipos_cambian', 'ids_desordenados', 'orden_cambia', 'celda_npt'])
def test_invariantes_detectan_trayectorias_patologicas(crear_netcdf,
        opciones, mensaje):
    with pytest.raises(ErrorTrayectoria, match=mensaje):
        Trayectoria(crear_netcdf(**opciones))


#/ ---- ARCHIVO ----------------------------------------------------------

def test_trayectoria_inexistente_falla(tmp_path):
    with pytest.raises(ErrorTrayectoria):
        Trayectoria(str(tmp_path / 'no_existe.nc'))


def test_cerrar_es_idempotente(crear_netcdf):
    tr = Trayectoria(crear_netcdf())
    tr.cerrar()
    tr.cerrar()
