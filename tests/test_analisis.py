# -*- coding:utf-8 -*-

"""
Tests de los módulos flujo_nt.especies y flujo_nt.analisis.

Hay tres tipos de verificación:

- Contra la biblioteca original y contra LAMMPS, sobre los datos de ejemplo.
- Exactas, sobre un sistema sintético cuya respuesta se conoce de antemano.
- De propiedades del diseño, como que el resultado no dependa del tamaño de
  bloque.

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: noviembre de 2026
"""

import numpy as np
import pytest

from flujo_nt import (
    ESPECIES_AGUA_METANOL,
    DensidadLongitudinal,
    DensidadRadial,
    ErrorAnalisis,
    Flujo,
    MarcoTubo,
    PerfilVelocidad,
    Pervaporacion,
    Trayectoria,
    Tubo,
    buscar_especie,
    recorrer)


#/ ---- ESPECIES ---------------------------------------------------------

def test_masas_moleculares_coinciden_con_configuracion(configuracion):
    """Las masas de la tabla son la suma de las masas atómicas del .data."""
    masas = configuracion.masas

    assert buscar_especie('Oa').masa_molecular == pytest.approx(
        masas[1] + 2 * masas[2])
    assert buscar_especie('Om').masa_molecular == pytest.approx(
        masas[3] + masas[4] + masas[5])


def test_tipos_de_la_tabla_coinciden_con_la_trayectoria(trayectoria):
    """Hay el doble de hidrógenos que de oxígenos en el agua, y uno de cada
    tipo por molécula de metanol."""
    presentes = trayectoria.tipos_presentes

    assert presentes[2] == 2 * presentes[1]
    assert presentes[3] == presentes[4] == presentes[5]
    assert {e.tipo for e in ESPECIES_AGUA_METANOL} == {1, 2, 3, 4, 5}


def test_buscar_especie():
    assert buscar_especie('Om') is buscar_especie(3)

    with pytest.raises(KeyError):
        buscar_especie('Xx')


#/ ---- FRENTE A LA BIBLIOTECA ORIGINAL ----------------------------------

@pytest.mark.parametrize('codigo', ['Oa', 'Om'])
def test_flujo_reproduce_original(trayectoria, tubo_ejemplo, referencia,
        codigo):
    """
    Mismas partículas en la región y misma suma de velocidades. El flujo
    difiere sólo por el volumen de la región, porque el radio del tubo se
    mide ahora con anillos de ancho uniforme.
    """
    especie = buscar_especie(codigo)
    original = referencia['flujo'][str(especie.tipo)]

    resultado, = recorrer(trayectoria, [Flujo(especie, tubo_ejemplo)])

    volumen_original = (np.pi * referencia['radio']**2 * referencia['largo']
                        * referencia['fraccion_region'])

    np.testing.assert_array_equal(resultado['n'], original['conteo'])
    np.testing.assert_allclose(
        resultado['Jn'] * resultado.metadatos['volumen_region'],
        np.array(original['flujo_numerico']) * volumen_original,
        rtol=1e-5)


@pytest.mark.parametrize('tipo', [1, 2, 3, 4, 5])
def test_densidad_radial_cercana_a_original(trayectoria, tubo_ejemplo,
        referencia, tipo):
    """
    Mismos bins y mismo total de partículas. Algunas partículas cerca de un
    borde cambian de bin, porque la coordenada radial difiere en centésimas
    de Å con la geometría medida por el método revisado.
    """
    original = referencia['densidad_radial'][str(tipo)]

    resultado, = recorrer(
        trayectoria, [DensidadRadial(buscar_especie(tipo), tubo_ejemplo)])

    np.testing.assert_allclose(resultado.metadatos['bordes_bins'],
                               original['bordes'])
    assert resultado['n'].sum() == sum(original['conteo'])
    assert np.abs(resultado['n'] - original['conteo']).max() <= 2


@pytest.mark.parametrize('codigo', ['Oa', 'Om'])
def test_perfil_velocidad_conserva_sumas(trayectoria, tubo_ejemplo,
        referencia, codigo):
    """
    El total de muestras y las sumas de velocidades y de sus cuadrados sobre
    todos los bins coinciden con el original, aunque alguna partícula cambie
    de bin.
    """
    especie = buscar_especie(codigo)
    original = referencia['perfil_velocidad'][str(especie.tipo)]

    resultado, = recorrer(trayectoria,
                          [PerfilVelocidad(especie, tubo_ejemplo)])

    def sumas(conteo, media, desvio):
        conteo, media, desvio = (np.asarray(x, dtype=float)
                                 for x in (conteo, media, desvio))
        hay = conteo > 0
        return (conteo.sum(),
                (conteo * media)[hay].sum(),
                (conteo * (desvio**2 + media**2))[hay].sum())

    nuevas = sumas(resultado['n'], resultado['v_media'],
                   resultado['v_desvio'])
    viejas = sumas(original['conteo'], original['media'], original['desvio'])

    assert nuevas[0] == viejas[0]
    np.testing.assert_allclose(nuevas[1:], viejas[1:], rtol=1e-5)


@pytest.mark.parametrize('codigo', ['Oa', 'Om'])
def test_pervaporacion_reproduce_original(trayectoria, tubo_ejemplo,
        referencia, codigo):
    especie = buscar_especie(codigo)

    resultado, = recorrer(trayectoria, [Pervaporacion(especie, tubo_ejemplo)])

    np.testing.assert_array_equal(
        resultado['n'], referencia['pervaporacion'][str(especie.tipo)])


#/ ---- FRENTE A LAMMPS --------------------------------------------------

@pytest.mark.parametrize('region', ['izq', 'tubo', 'der'])
def test_densidad_longitudinal_reproduce_lammps(trayectoria,
        referencia_lammps, region):
    """Con el largo y el radio de la corrida, coincide bin a bin."""
    lammps = referencia_lammps['Oa'][region]

    resultado, = recorrer(trayectoria, [DensidadLongitudinal(
        buscar_especie('Oa'), region,
        largo=referencia_lammps['largo'],
        radio=referencia_lammps['radio'],
        factor_radio=referencia_lammps['factor_radio'],
        numero_bins=referencia_lammps['numero_bins'],
        por_cuadro=True)])

    np.testing.assert_array_equal(resultado['n'], lammps['conteo'])
    np.testing.assert_allclose(resultado['rho_n'], lammps['densidad'],
                               rtol=1e-5)
    #/ LAMMPS escribe las coordenadas con seis cifras significativas
    np.testing.assert_allclose(resultado.metadatos['centros_bins'],
                               lammps['centro'], atol=1e-3)


def test_pervaporacion_reproduce_lammps(trayectoria, referencia_lammps):
    resultado, = recorrer(trayectoria, [Pervaporacion(
        buscar_especie('Oa'), largo=referencia_lammps['largo'])])

    np.testing.assert_array_equal(resultado['n'],
                                  referencia_lammps['Oa']['pervaporacion'])


#/ ---- SISTEMA SINTÉTICO ------------------------------------------------

@pytest.fixture
def sistema_sintetico(crear_netcdf):
    """
    Trayectoria de dos cuadros con partículas en posiciones elegidas, y un
    tubo recto de radio 5.5 Å y largo 100 Å sobre el eje Z.

    Las partículas de tipo 1 son:

    - tres dentro de la región, a distancias 0.5, 1 y 2 Å del eje, con
      velocidades longitudinales (0.1, 0.2, 0.3) Å/fs en el primer cuadro
      y (0.3, 0.2, 0.1) Å/fs en el segundo;
    - una fuera del radio (7 Å) y otra fuera del largo de la región (z=48);
    - una en cada reservorio, en z = ±60.

    Hay además una partícula de tipo 2 dentro de la región, que ningún
    análisis de tipo 1 debe contar.
    """
    posiciones = np.array([
        [0.5, 0.0, 0.0], [0.0, 1.0, 10.0], [2.0, 0.0, -20.0],
        [7.0, 0.0, 0.0], [1.0, 0.0, 48.0],
        [0.0, 0.0, 60.0], [0.0, 0.0, -60.0],
        [0.5, 0.0, 0.0]])
    tipos = [1, 1, 1, 1, 1, 1, 1, 2]

    velocidades = np.zeros((2, 8, 3))
    velocidades[:, :, 2] = 9.0
    velocidades[0, :3, 2] = [0.1, 0.2, 0.3]
    velocidades[1, :3, 2] = [0.3, 0.2, 0.1]

    ruta = crear_netcdf(
        n_cuadros=2, n_atomos=8,
        tipos=np.tile(tipos, (2, 1)),
        posiciones=np.tile(posiciones, (2, 1, 1)),
        velocidades=velocidades,
        longitudes=np.tile([64.0, 64.0, 200.0], (2, 1)),
        origen=np.tile([-32.0, -32.0, -100.0], (2, 1)))

    tubo = Tubo(MarcoTubo('Z'), largo=100.0, centro=0.0,
                bordes_anillos=[-40.0, 0.0, 40.0],
                centros_anillos=np.zeros((2, 2)),
                radios_anillos=[5.5, 5.5])

    return ruta, tubo


def test_analisis_exactos_en_sistema_sintetico(sistema_sintetico):
    ruta, tubo = sistema_sintetico
    agua = buscar_especie('Oa')

    with Trayectoria(ruta) as tr:
        flujo, densidad, perfil, serie, adelante, atras = recorrer(tr, [
            Flujo(agua, tubo),
            DensidadRadial(agua, tubo),
            PerfilVelocidad(agua, tubo),
            PerfilVelocidad(agua, tubo, por_cuadro=True),
            Pervaporacion(agua, tubo),
            Pervaporacion(agua, tubo, lado=-1)])

    volumen = np.pi * 5.5**2 * 90.0
    np.testing.assert_array_equal(flujo['n'], [3, 3])
    np.testing.assert_allclose(flujo['Jn'], [0.6 / volumen] * 2, rtol=1e-6)
    np.testing.assert_allclose(flujo['Jm'], flujo['Jn'] * 18.0154)

    #/ Bins de 0.875 Å hasta cubrir 5.5 Å: siete bins
    np.testing.assert_array_equal(densidad['n'], [2, 2, 2, 0, 0, 0, 0])
    bordes = densidad.metadatos['bordes_bins']
    np.testing.assert_allclose(
        densidad['rho_n_media'][:3],
        1.0 / (90.0 * np.pi * np.diff(bordes[:4]**2)))
    np.testing.assert_allclose(densidad['rho_n_desvio'], 0.0)

    np.testing.assert_allclose(perfil['v_media'][:3], [0.2, 0.2, 0.2],
                               rtol=1e-6)
    np.testing.assert_allclose(perfil['v_desvio'][:3], [0.1, 0.0, 0.1],
                               atol=1e-6)
    assert np.isnan(perfil['v_media'][3:]).all()

    np.testing.assert_allclose(serie['v_media'][:, :3],
                               [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]],
                               rtol=1e-6)

    np.testing.assert_array_equal(adelante['n'], [1, 1])
    np.testing.assert_array_equal(atras['n'], [1, 1])


def test_densidad_longitudinal_exacta_en_sistema_sintetico(
        sistema_sintetico):
    """
    Bins de 100/32 = 3.125 Å. En el tubo, el cilindro tiene radio
    1.5 × 5.5 = 8.25 Å, así que la partícula a 7 Å del eje se cuenta.
    """
    ruta, tubo = sistema_sintetico
    agua = buscar_especie('Oa')

    with Trayectoria(ruta) as tr:
        izquierda, centro, derecha = recorrer(tr, [
            DensidadLongitudinal(agua, region, tubo)
            for region in ('izq', 'tubo', 'der')])

    def ocupados(resultado):
        return {int(i): resultado['n_media'][i]
                for i in np.flatnonzero(resultado['n_media'])}

    assert len(centro['n_media']) == 32
    assert ocupados(centro) == {9: 1, 16: 2, 19: 1, 31: 1}
    assert ocupados(izquierda) == {12: 1}
    assert ocupados(derecha) == {3: 1}
    np.testing.assert_allclose(
        centro['rho_n_media'][16], 2 / (3.125 * np.pi * 8.25**2))


#/ ---- PROPIEDADES DEL DISEÑO -------------------------------------------

def test_resultado_no_depende_del_tamanio_de_bloque(rutas_ejemplo,
        configuracion, tubo_ejemplo):
    """Procesar de a un cuadro o de a dos da exactamente lo mismo."""
    agua, metanol = buscar_especie('Oa'), buscar_especie('Om')

    def analisis():
        return [Flujo(agua, tubo_ejemplo),
                DensidadRadial(metanol, tubo_ejemplo),
                PerfilVelocidad(agua, tubo_ejemplo),
                PerfilVelocidad(agua, tubo_ejemplo, por_cuadro=True),
                DensidadLongitudinal(agua, 'izq', tubo_ejemplo),
                Pervaporacion(metanol, tubo_ejemplo)]

    with Trayectoria(rutas_ejemplo[0], configuracion=configuracion) as tr:
        de_a_uno = recorrer(tr, analisis(), tamanio_bloque=1)
        de_a_dos = recorrer(tr, analisis(), tamanio_bloque=2)

    for uno, dos in zip(de_a_uno, de_a_dos):
        assert uno.nombres == dos.nombres
        for nombre in uno.nombres:
            np.testing.assert_allclose(uno[nombre], dos[nombre],
                                       rtol=1e-12, equal_nan=True)


def test_rango_de_cuadros(trayectoria, tubo_ejemplo):
    resultado, = recorrer(trayectoria,
                          [Pervaporacion(buscar_especie('Oa'), tubo_ejemplo)],
                          inicio=1)

    np.testing.assert_array_equal(resultado['tiempo'], [1000.0])


def test_calculos_tienen_nombres_distintos(tubo_ejemplo):
    """Cada resultado se escribe en un archivo distinto."""
    agua = buscar_especie('Oa')
    calculos = [
        Flujo(agua, tubo_ejemplo).calculo,
        DensidadRadial(agua, tubo_ejemplo).calculo,
        PerfilVelocidad(agua, tubo_ejemplo).calculo,
        PerfilVelocidad(agua, tubo_ejemplo, por_cuadro=True).calculo,
        Pervaporacion(agua, tubo_ejemplo).calculo] + [
        DensidadLongitudinal(agua, region, tubo_ejemplo,
                             por_cuadro=por_cuadro).calculo
        for region in ('izq', 'tubo', 'der')
        for por_cuadro in (False, True)]

    assert len(set(calculos)) == len(calculos)


#/ ---- ERRORES ----------------------------------------------------------

def test_flujo_sin_masa_molecular_falla(tubo_ejemplo):
    with pytest.raises(ErrorAnalisis, match='masa molecular'):
        Flujo(buscar_especie('Ha'), tubo_ejemplo)


def test_procesar_sin_iniciar_falla(trayectoria, tubo_ejemplo):
    analisis = Pervaporacion(buscar_especie('Oa'), tubo_ejemplo)

    with pytest.raises(ErrorAnalisis, match='iniciarse'):
        analisis.procesar(trayectoria.cuadro(0))


def test_resultado_sin_cuadros_falla(trayectoria, tubo_ejemplo):
    analisis = Pervaporacion(buscar_especie('Oa'), tubo_ejemplo)
    analisis.iniciar(trayectoria)

    with pytest.raises(ErrorAnalisis, match='ningún cuadro'):
        analisis.resultado()


@pytest.mark.parametrize('construir', [
    lambda e, t: DensidadLongitudinal(e, 'centro', t),
    lambda e, t: DensidadLongitudinal(e, 'izq'),
    lambda e, t: DensidadLongitudinal(e, 'tubo', largo=100.0),
    lambda e, t: Pervaporacion(e, t, lado=0),
    lambda e, t: Pervaporacion(e)],
    ids=['region_invalida', 'sin_largo', 'sin_radio', 'lado_invalido',
         'sin_largo_pervaporacion'])
def test_parametros_invalidos_fallan(tubo_ejemplo, construir):
    with pytest.raises(ErrorAnalisis):
        construir(buscar_especie('Oa'), tubo_ejemplo)
