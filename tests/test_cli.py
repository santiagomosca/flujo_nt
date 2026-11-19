# -*- coding:utf-8 -*-

"""
Tests de la interfaz de línea de comandos (subcomando analizar).

Los tests llaman a :func:`flujo_nt.cli.main` con la lista de argumentos, sin
lanzar procesos, salvo el que comprueba la ejecución como módulo. Los datos
de ejemplo se enlazan en un directorio temporal con los nombres de la
convención de las simulaciones (``<base>.nc`` y ``<base>.final.data``), para
que la configuración se encuentre sola.

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: noviembre de 2026
"""

import os
import subprocess
import sys

import pytest
from netCDF4 import Dataset

from flujo_nt import (
    DensidadLongitudinal,
    DensidadRadial,
    Flujo,
    Pervaporacion,
    Trayectoria,
    Tubo,
    __version__,
    buscar_especie,
    escribir_resultado,
    recorrer)
from flujo_nt.cli import CALCULOS, main, _abrir
from flujo_nt.recorte import recortar_trayectoria


SALIDAS_PREDETERMINADAS = {
    'flujo.inst.Oa', 'flujo.inst.Om',
    'dens.num.radial.promedio.Oa', 'dens.num.radial.promedio.Ha',
    'dens.num.radial.promedio.Om', 'dens.num.radial.promedio.Cm',
    'dens.num.radial.promedio.Hm',
    'perfil.velocidad.promedio.Oa', 'perfil.velocidad.promedio.Om'}


#/ ---- PIEZAS COMUNES ---------------------------------------------------

@pytest.fixture
def sistema(tmp_path, rutas_ejemplo):
    """Trayectoria de ejemplo enlazada como sistema.1.nc, con su
    configuración sistema.1.final.data."""
    trayectoria, configuracion = rutas_ejemplo
    ruta = tmp_path / 'sistema.1.nc'
    os.symlink(trayectoria, ruta)
    os.symlink(configuracion, tmp_path / 'sistema.1.final.data')
    return ruta


def ejecutar(capsys, *argumentos):
    """Ejecución de flujo-nt; devuelve código, salida y errores."""
    codigo = main([str(a) for a in argumentos])
    capturado = capsys.readouterr()
    return codigo, capturado.out, capturado.err


def salidas(directorio, base='sistema.1'):
    """Resultados escritos en un directorio, sin la base ni la extensión."""
    return {nombre[len(base) + 1:-len('.dat')]
            for nombre in os.listdir(directorio) if nombre.endswith('.dat')}


def datos(ruta):
    """Líneas de datos de un archivo de resultados, sin el encabezado."""
    with open(ruta, encoding='utf-8') as archivo:
        return [linea for linea in archivo if not linea.startswith('#')]


#/ ---- FUNCIONAMIENTO --------------------------------------------------

def test_version(capsys):
    with pytest.raises(SystemExit) as salida:
        main(['--version'])
    assert salida.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_ejecucion_como_modulo():
    """python -m flujo_nt equivale al comando instalado."""
    proceso = subprocess.run(
        [sys.executable, '-m', 'flujo_nt', '--version'],
        capture_output=True, text=True)
    assert proceso.returncode == 0
    assert __version__ in proceso.stdout


def test_calculos_predeterminados(capsys, sistema):
    codigo, salida, errores = ejecutar(capsys, 'analizar', sistema)

    assert codigo == 0
    assert errores == ''
    assert salidas(sistema.parent) == SALIDAS_PREDETERMINADAS
    assert f'configuración: {sistema.parent}/sistema.1.final.data' in salida
    assert 'bins radiales: 0.8750 Å' in salida


def test_todos_los_calculos(capsys, sistema):
    codigo, salida, _ = ejecutar(
        capsys, 'analizar', sistema, '-c', ','.join(CALCULOS))

    esperadas = sum(len(calculo.especies) for calculo in CALCULOS.values())
    #/ La densidad longitudinal escribe un archivo por región
    esperadas += 2 * 2 * (len(DensidadLongitudinal.REGIONES) - 1)

    assert codigo == 0
    assert len(salidas(sistema.parent)) == esperadas
    assert 'bins longitudinales:' in salida


def test_resultados_identicos_a_la_biblioteca(capsys, sistema, tmp_path,
        configuracion):
    """El comando escribe lo mismo que la biblioteca usada directamente,
    incluidas las opciones que se pasan a los análisis."""
    ejecutar(capsys, 'analizar', sistema, '-c',
             'flujo,densidad-radial,densidad-longitudinal-serie,'
             'pervaporacion', '--fraccion-bin-r', '0.5')

    agua, metanol = buscar_especie('Oa'), buscar_especie('Om')
    hidrogeno = buscar_especie('Ha')
    with Trayectoria(str(sistema), configuracion=configuracion) as tr:
        tubo = Tubo.desde_trayectoria(tr)
        resultados = recorrer(tr, [
            Flujo(agua, tubo),
            DensidadRadial(hidrogeno, tubo, fraccion_bin_r=0.5),
            DensidadLongitudinal(metanol, 'tubo', tubo, por_cuadro=True),
            Pervaporacion(agua, tubo)])

    referencia = tmp_path / 'referencia'
    for resultado in resultados:
        ruta = escribir_resultado(resultado, str(referencia / 'sistema.1'))
        nombre = os.path.basename(ruta)
        assert datos(ruta) == datos(sistema.parent / nombre), nombre


class _Opciones:
    """Opciones mínimas para abrir una trayectoria con _abrir."""
    configuracion = None


def test_filtro_de_especies(capsys, sistema):
    codigo, _, _ = ejecutar(capsys, 'analizar', sistema, '-e', 'Oa')

    assert codigo == 0
    assert salidas(sistema.parent) == {
        'flujo.inst.Oa', 'dens.num.radial.promedio.Oa',
        'perfil.velocidad.promedio.Oa'}


def test_filtro_omite_calculos_sin_especies(capsys, sistema):
    codigo, _, errores = ejecutar(
        capsys, 'analizar', sistema, '-c', 'flujo,densidad-radial',
        '-e', 'Ha')

    assert codigo == 0
    assert 'se omite flujo' in errores
    assert salidas(sistema.parent) == {'dens.num.radial.promedio.Ha'}


def test_filtro_sin_ningun_calculo(capsys, sistema):
    codigo, _, errores = ejecutar(
        capsys, 'analizar', sistema, '-c', 'flujo', '-e', 'Ha')

    assert codigo == 2
    assert 'ningún cálculo' in errores
    assert salidas(sistema.parent) == set()


def test_lado_negativo(capsys, sistema):
    ejecutar(capsys, 'analizar', sistema, '-c', 'pervaporacion', '-e', 'Oa',
             '--lado', '-1')

    with open(sistema.parent / 'sistema.1.conteo.pervaporacion.Oa.dat',
              encoding='utf-8') as archivo:
        assert '# lado = -1\n' in archivo.read()


@pytest.mark.parametrize('argumentos', [
    ['-c', 'perfil'],
    ['-c', ''],
    ['-e', 'Cp'],
    ['--lado', '2'],
    ['--sigma-ref', '-1'],
    ['--tamanio-bloque', '0'],
    ['--paso-integracion', 'dos']])
def test_argumentos_invalidos(capsys, sistema, argumentos):
    with pytest.raises(SystemExit) as salida:
        main(['analizar', str(sistema), *argumentos])

    assert salida.value.code == 2
    assert 'flujo-nt analizar' in capsys.readouterr().err


#/ ---- VARIAS TRAYECTORIAS Y SALIDAS ------------------------------------

def test_ruta_inexistente_no_detiene_a_las_demas(capsys, sistema):
    faltante = sistema.parent / 'faltante.1.nc'
    codigo, _, errores = ejecutar(capsys, 'analizar', faltante, sistema)

    assert codigo == 1
    assert f'error: {faltante}' in errores
    assert '1 de 2 trayectorias fallaron' in errores
    assert salidas(sistema.parent) == SALIDAS_PREDETERMINADAS


def test_trayectoria_repetida_se_procesa_una_vez(capsys, sistema):
    codigo, salida, _ = ejecutar(capsys, 'analizar', sistema, sistema)

    assert codigo == 0
    assert '[1/1]' in salida


def test_avisa_al_sobrescribir(capsys, sistema):
    codigo, salida, errores = ejecutar(capsys, 'analizar', sistema, '-q')
    assert (codigo, salida, errores) == (0, '', '')

    codigo, salida, errores = ejecutar(capsys, 'analizar', sistema, '-q')
    avisos = errores.splitlines()

    assert codigo == 0
    assert salida == ''
    assert len(avisos) == len(SALIDAS_PREDETERMINADAS)
    assert all(aviso.startswith('aviso: se sobrescribe ')
               for aviso in avisos)


def test_directorio_de_salida(capsys, sistema, tmp_path):
    destino = tmp_path / 'resultados' / 'R1-5'
    codigo, _, _ = ejecutar(capsys, 'analizar', sistema, '-o', destino)

    assert codigo == 0
    assert salidas(destino) == SALIDAS_PREDETERMINADAS
    assert salidas(sistema.parent) == set()


#/ ---- CONFIGURACIÓN Y PASO DE INTEGRACIÓN -----------------------------

def test_celda_degenerada_sin_configuracion(capsys, tmp_path, rutas_ejemplo):
    ruta = tmp_path / 'sola.nc'
    os.symlink(rutas_ejemplo[0], ruta)

    codigo, _, errores = ejecutar(capsys, 'analizar', ruta)

    assert codigo == 1
    assert 'sola.final.data' in errores
    assert '--configuracion' in errores


def test_configuracion_explicita(capsys, tmp_path, rutas_ejemplo):
    ruta = tmp_path / 'sola.nc'
    os.symlink(rutas_ejemplo[0], ruta)

    codigo, _, errores = ejecutar(
        capsys, 'analizar', ruta, '-c', 'flujo', '-e', 'Oa',
        '--configuracion', rutas_ejemplo[1])

    assert (codigo, errores) == (0, '')
    assert salidas(tmp_path, base='sola') == {'flujo.inst.Oa'}


def test_configuracion_con_varias_trayectorias(capsys, sistema):
    otra = sistema.parent / 'otra.1.nc'
    os.symlink(sistema, otra)
    with pytest.raises(SystemExit) as salida:
        main(['analizar', str(sistema), str(otra), '--configuracion',
              str(sistema.parent / 'sistema.1.final.data')])
    assert salida.value.code == 2


def test_celda_valida_sin_configuracion(capsys, crear_netcdf):
    """Sin configuración y con celda válida se avisa y se sigue."""
    ruta = crear_netcdf()
    trayectoria, configuracion = _abrir(str(ruta), _Opciones())
    trayectoria.cerrar()

    assert configuracion is None
    assert 'no se encontró' in capsys.readouterr().err


def test_pervaporacion_requiere_paso_de_integracion(capsys, sistema):
    """Una trayectoria sin factor de escala en el tiempo (como las de OVITO)
    necesita --paso-integracion para la pervaporación."""
    ruta = sistema.parent / 'sin_escala.1.nc'
    recortar_trayectoria(str(sistema), str(ruta), 1)
    with Dataset(ruta, 'a') as ds:
        ds['time'].delncattr('scale_factor')
    os.symlink(sistema.parent / 'sistema.1.final.data',
               sistema.parent / 'sin_escala.1.final.data')

    codigo, _, errores = ejecutar(
        capsys, 'analizar', ruta, '-c', 'pervaporacion', '-e', 'Oa')
    assert codigo == 1
    assert '--paso-integracion' in errores

    codigo, _, _ = ejecutar(
        capsys, 'analizar', ruta, '-c', 'pervaporacion', '-e', 'Oa',
        '--paso-integracion', '2')
    assert codigo == 0
    assert datos(sistema.parent / 'sin_escala.1.conteo.pervaporacion.Oa.dat'
                 )[0].split()[0] == '0'
