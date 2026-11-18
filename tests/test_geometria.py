# -*- coding:utf-8 -*-

"""
Tests del módulo flujo_nt.geometria.

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: noviembre de 2026
"""

import numpy as np
import pytest

from flujo_nt import Celda, ErrorGeometria, MarcoTubo, Tubo


#/ ---- MARCO DEL TUBO ----------------------------------------------------

@pytest.mark.parametrize('eje, longitudinal, transversales', [
    ('X', 0, (1, 2)),
    ('Y', 1, (2, 0)),
    ('Z', 2, (0, 1))])
def test_marco_convencion_ciclica(eje, longitudinal, transversales):
    marco = MarcoTubo(eje)
    assert marco.indice_longitudinal == longitudinal
    assert marco.indices_transversales == transversales


def test_marco_eje_invalido_falla():
    with pytest.raises(ErrorGeometria):
        MarcoTubo('W')


def test_marco_desde_celda_corregida(trayectoria):
    assert MarcoTubo.desde_celda(trayectoria.celda).eje == 'Z'


@pytest.mark.parametrize('longitudes', [
    [63.947, 63.9, 0.0],
    [64.0, 65.0, 20.0]],
    ids=['degenerada', 'ambigua'])
def test_marco_desde_celda_rechaza(longitudes):
    """La biblioteca original devolvía 'X' en silencio en ambos casos."""
    with pytest.raises(ErrorGeometria):
        MarcoTubo.desde_celda(Celda(longitudes, [0, 0, 0]))


def test_marco_componentes():
    marco = MarcoTubo('Z')
    vectores = np.array([[3.0, 4.0, 7.0]])

    assert np.shares_memory(marco.longitudinal(vectores), vectores)
    np.testing.assert_array_equal(marco.longitudinal(vectores), [7.0])
    np.testing.assert_array_equal(marco.transversales(vectores), [[3.0, 4.0]])
    np.testing.assert_allclose(marco.radial(vectores), [5.0])


#/ ---- MEDICIÓN FRENTE A LA BIBLIOTECA ORIGINAL --------------------------

def test_largo_reproduce_original(tubo_ejemplo, referencia):
    assert tubo_ejemplo.largo == pytest.approx(referencia['largo'], abs=1e-4)
    assert abs(tubo_ejemplo.centro) < 0.01


def test_radio_cercano_al_original(tubo_ejemplo, referencia):
    """
    El radio difiere levemente del original porque los anillos se
    construyen con bordes corregidos (ancho uniforme, sin átomos fuera de
    los anillos).
    """
    assert tubo_ejemplo.radio == pytest.approx(referencia['radio'], abs=5e-3)


def test_anillos_reproducen_original_con_mismos_bordes(trayectoria,
        referencia):
    """Con los bordes del original, el cálculo por anillo es idéntico."""
    carbonos = trayectoria.cuadro(0).posiciones[0][
        trayectoria.mascara_tipo(6)]

    centros, radios, conteos = Tubo._medir_anillos(
        carbonos[:, 2], carbonos[:, [0, 1]],
        np.array(referencia['bordes_anillos']))

    np.testing.assert_allclose(radios, referencia['radios_anillos'],
                               atol=1e-6)
    np.testing.assert_allclose(centros, referencia['centros_anillos'],
                               atol=1e-5)


@pytest.mark.parametrize('tipo', [1, 3], ids=['agua', 'metanol'])
def test_conteo_en_tubo_reproduce_original(trayectoria, tubo_ejemplo,
        referencia, tipo):
    seleccion = trayectoria.mascara_tipo(tipo)
    conteos = []
    for bloque in trayectoria.bloques():
        corregidas = tubo_ejemplo.corregir(bloque.posiciones[:, seleccion])
        en_tubo = tubo_ejemplo.en_region(
            corregidas, fraccion_largo=referencia['fraccion_region'])
        conteos.extend(en_tubo.sum(axis=1).tolist())

    assert conteos == referencia['conteos_en_tubo'][str(tipo)]


#/ ---- MEDICIÓN SOBRE GEOMETRÍA CONOCIDA ---------------------------------

def test_tubo_sintetico_medido_con_exactitud(fabrica_tubo):
    atomos, celda, eje_x = fabrica_tubo()
    tubo = Tubo.medir(atomos, MarcoTubo.desde_celda(celda), celda)

    assert tubo.largo == pytest.approx(130.0, abs=1e-6)
    assert tubo.centro == pytest.approx(0.0, abs=1e-6)
    assert tubo.radio == pytest.approx(5.5, abs=0.01)
    np.testing.assert_allclose(
        tubo.centros_anillos[:, 0], eje_x(tubo.medios_anillos), atol=0.05)


def test_tubo_recto_es_caso_degenerado(fabrica_tubo):
    """En un tubo recto todos los centros de anillo coinciden."""
    atomos, celda, _ = fabrica_tubo(flecha=0.0)
    tubo = Tubo.medir(atomos, MarcoTubo.desde_celda(celda), celda)

    np.testing.assert_allclose(tubo.centros_anillos, 0.0, atol=0.03)


@pytest.mark.parametrize('distancia', [0.5, 3.0, 5.0])
def test_corregir_recupera_distancia_al_eje_curvo(fabrica_tubo, distancia):
    atomos, celda, eje_x = fabrica_tubo()
    marco = MarcoTubo.desde_celda(celda)
    tubo = Tubo.medir(atomos, marco, celda)

    generador = np.random.default_rng(1)
    z = generador.uniform(-58, 58, 5000)
    angulo = generador.uniform(0, 2 * np.pi, z.size)
    posiciones = np.stack([eje_x(z) + distancia * np.cos(angulo),
                           distancia * np.sin(angulo),
                           z], axis=1)

    radial = marco.radial(tubo.corregir(posiciones))

    np.testing.assert_allclose(radial, distancia, atol=0.03)


#/ ---- ERRORES DE MEDICIÓN -----------------------------------------------

@pytest.mark.parametrize('tipos', [(6, 7), (1,)],
                         ids=['con_paredes_sueltas', 'sin_tubo'])
def test_tipos_que_no_son_tubo_se_rechazan(trayectoria, tipos):
    with pytest.raises(ErrorGeometria, match='pared'):
        Tubo.desde_trayectoria(trayectoria, tipos=tipos)


def test_anillo_vacio_se_rechaza(fabrica_tubo):
    atomos, celda, _ = fabrica_tubo()
    radial = np.hypot(atomos[:, 0], atomos[:, 1])
    hueco = (atomos[:, 2] > 10) & (atomos[:, 2] < 12.5) & (radial < 8)

    with pytest.raises(ErrorGeometria, match='Anillos'):
        Tubo.medir(atomos[~hueco], MarcoTubo.desde_celda(celda), celda)


def test_tubo_sin_paredes_se_rechaza(fabrica_tubo):
    atomos, celda, _ = fabrica_tubo()
    radial = np.hypot(atomos[:, 0], atomos[:, 1])

    with pytest.raises(ErrorGeometria, match='borde'):
        Tubo.medir(atomos[radial < 8], MarcoTubo.desde_celda(celda), celda)


@pytest.mark.parametrize('opciones', [
    {'centros_anillos': np.zeros((3, 2))},
    {'radios_anillos': np.zeros(3)},
    {'bordes_anillos': [0.0, 2.0, 1.0, 3.0, 4.0]}],
    ids=['centros', 'radios', 'bordes'])
def test_constructor_valida_formas(opciones):
    argumentos = dict(
        marco=MarcoTubo('Z'), largo=10.0, centro=0.0,
        bordes_anillos=[0.0, 1.0, 2.0, 3.0, 4.0],
        centros_anillos=np.zeros((4, 2)), radios_anillos=np.ones(4))
    argumentos.update(opciones)

    with pytest.raises(ErrorGeometria):
        Tubo(**argumentos)


#/ ---- USO --------------------------------------------------------------

def test_corregir_no_modifica_entrada(trayectoria, tubo_ejemplo):
    posiciones = trayectoria.cuadro(0).posiciones[
        :, trayectoria.mascara_tipo(1)]
    copia = posiciones.copy()

    corregidas = tubo_ejemplo.corregir(posiciones)

    np.testing.assert_array_equal(posiciones, copia)
    assert corregidas.shape == posiciones.shape
    assert corregidas.dtype == posiciones.dtype


def test_limites_y_region():
    tubo = Tubo(MarcoTubo('Z'), largo=100.0, centro=0.0,
                bordes_anillos=[-40.0, 0.0, 40.0],
                centros_anillos=np.zeros((2, 2)), radios_anillos=[5.0, 5.0])

    np.testing.assert_allclose(tubo.limites(0.9), [-45.0, 45.0])

    puntos = np.array([[0.0, 0.0, 0.0],     # en el eje
                       [4.9, 0.0, 44.0],    # dentro
                       [5.1, 0.0, 0.0],     # fuera en radio
                       [0.0, 0.0, 46.0]])   # fuera en largo
    np.testing.assert_array_equal(
        tubo.en_region(puntos, fraccion_largo=0.9),
        [True, True, False, False])
