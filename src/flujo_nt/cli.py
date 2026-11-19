#! /usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Interfaz de línea de comandos ``flujo-nt``.

Subcomandos
-----------
analizar
    Ejecuta los cálculos pedidos sobre una o más trayectorias, con una sola
    pasada por cada una, y escribe los resultados junto a ellas.
recortar
    Copia los primeros cuadros de una trayectoria a un archivo nuevo.

La selección de trayectorias se deja al intérprete de comandos: el
subcomando ``analizar`` recibe las rutas ya expandidas, de modo que no
depende de cómo estén organizados los directorios.

Examples
--------
Desde la terminal::

    flujo-nt analizar sistema.1.nc
    flujo-nt analizar $RAIZ/R{1-5,3-0}/**/*.398K.{1,2}.nc -c flujo
    flujo-nt recortar sistema.1.nc sistema.1.recortado.nc -n 2

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: noviembre de 2026
"""

import argparse
import collections
import os
import sys
import time

from . import __version__
from .analisis import (
    FRACCION_BIN_R_PREDETERMINADA,
    SIGMA_REF_PREDETERMINADO,
    DensidadLongitudinal,
    DensidadRadial,
    ErrorAnalisis,
    Flujo,
    PerfilVelocidad,
    Pervaporacion,
    recorrer)
from .escritura import (
    base_desde_trayectoria, escribir_resultado, nombre_archivo)
from .especies import ESPECIES_AGUA_METANOL, buscar_especie
from .geometria import ErrorGeometria, Tubo
from .recorte import (
    CUADROS_PREDETERMINADOS, FORMATOS, listar_contenido, recortar_trayectoria)
from .trayectoria import (
    TAMANIO_BLOQUE_PREDETERMINADO,
    Configuracion,
    ErrorTrayectoria,
    Trayectoria)


#/ ---- CÁLCULOS DISPONIBLES ---------------------------------------------

SUFIJO_CONFIGURACION = '.final.data'
"""str: Sufijo que, agregado a la base de una trayectoria, da la ruta de su
configuración."""

Calculo = collections.namedtuple('Calculo', ['especies', 'construir'])
Calculo.__doc__ = """
Cálculo seleccionable desde la línea de comandos.

Attributes
----------
especies : tuple of str
    Códigos de las especies para las que se calcula por defecto.
construir : callable
    ``construir(especie, tubo, opciones)`` devuelve la lista de análisis
    de esa especie.
"""


def _flujo(especie, tubo, opciones):
    return [Flujo(especie, tubo)]


def _densidad_radial(especie, tubo, opciones):
    return [DensidadRadial(especie, tubo,
                           fraccion_bin_r=opciones.fraccion_bin_r,
                           sigma_ref=opciones.sigma_ref)]


def _perfil_velocidad(por_cuadro):
    def construir(especie, tubo, opciones):
        return [PerfilVelocidad(especie, tubo,
                                fraccion_bin_r=opciones.fraccion_bin_r,
                                sigma_ref=opciones.sigma_ref,
                                por_cuadro=por_cuadro)]
    return construir


def _densidad_longitudinal(por_cuadro):
    def construir(especie, tubo, opciones):
        return [DensidadLongitudinal(especie, region, tubo,
                                     por_cuadro=por_cuadro)
                for region in DensidadLongitudinal.REGIONES]
    return construir


def _pervaporacion(especie, tubo, opciones):
    return [Pervaporacion(especie, tubo, lado=opciones.lado,
                          paso_integracion=opciones.paso_integracion)]


OXIGENOS = ('Oa', 'Om')
TODAS = tuple(especie.codigo for especie in ESPECIES_AGUA_METANOL)

CALCULOS = {
    'flujo': Calculo(OXIGENOS, _flujo),
    'densidad-radial': Calculo(TODAS, _densidad_radial),
    'perfil-velocidad': Calculo(OXIGENOS, _perfil_velocidad(False)),
    'perfil-velocidad-serie': Calculo(OXIGENOS, _perfil_velocidad(True)),
    'densidad-longitudinal': Calculo(OXIGENOS, _densidad_longitudinal(False)),
    'densidad-longitudinal-serie': Calculo(OXIGENOS,
                                           _densidad_longitudinal(True)),
    'pervaporacion': Calculo(OXIGENOS, _pervaporacion)}
"""dict: Cálculos seleccionables con ``-c``, con sus especies por defecto."""

CALCULOS_PREDETERMINADOS = ('flujo', 'densidad-radial', 'perfil-velocidad')
"""tuple of str: Cálculos que se ejecutan si no se indica ``-c``."""


#/ ---- MENSAJES ---------------------------------------------------------

def _aviso(texto):
    print(f'aviso: {texto}', file=sys.stderr)


def _error(texto):
    print(f'error: {texto}', file=sys.stderr)


#/ ---- ARGUMENTOS ------------------------------------------------------

def _lista(validos, que):
    """
    Conversor de argparse para listas separadas por comas.

    Devuelve los elementos sin repetir y en el orden dado, y rechaza los
    que no estén en ``validos``.
    """
    def convertir(texto):
        elementos = [e.strip() for e in texto.split(',') if e.strip()]
        invalidos = [e for e in elementos if e not in validos]
        if invalidos or not elementos:
            raise argparse.ArgumentTypeError(
                f'{que} inválido(s): {", ".join(invalidos) or texto!r}. '
                f'Válidos: {", ".join(validos)}')
        return list(dict.fromkeys(elementos))
    return convertir


def _positivo(tipo):
    """Conversor de argparse para números positivos."""
    def convertir(texto):
        try:
            valor = tipo(texto)
        except ValueError:
            raise argparse.ArgumentTypeError(f'número inválido: {texto!r}')
        if valor <= 0:
            raise argparse.ArgumentTypeError(
                f'debe ser positivo: {texto!r}')
        return valor
    return convertir


def _epilogo_analizar():
    lineas = ['cálculos disponibles (especies por defecto):']
    for nombre, calculo in CALCULOS.items():
        marca = ' *' if nombre in CALCULOS_PREDETERMINADOS else ''
        lineas.append(f'  {nombre:28s} {",".join(calculo.especies)}{marca}')
    lineas += [
        '  (*) se ejecutan si no se indica -c',
        '',
        'especies:',
        *(f'  {e.codigo}  {e.nombre}' for e in ESPECIES_AGUA_METANOL),
        '',
        'La configuración de cada trayectoria se busca como <base>'
        f'{SUFIJO_CONFIGURACION},',
        'donde <base> es la ruta de la trayectoria sin la extensión.',
        '',
        'ejemplos:',
        '  flujo-nt analizar sistema.1.nc',
        '  flujo-nt analizar sistema.1.nc -c pervaporacion,flujo -e Oa',
        '  flujo-nt analizar $RAIZ/R{1-5,3-0}/**/*.398K.{1,2}.nc '
        '-q 2> avisos.log',
        '  (el patrón ** requiere shopt -s globstar en bash)']
    return '\n'.join(lineas)


def construir_parser():
    """
    Parser de argumentos de ``flujo-nt``.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog='flujo-nt',
        description='Análisis de trayectorias de flujo a través de '
                    'nanotubos.')
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {__version__}')

    subparsers = parser.add_subparsers(dest='comando', metavar='COMANDO')
    subparsers.required = True

    #/ analizar
    analizar = subparsers.add_parser(
        'analizar',
        help='calcular flujo, densidades, perfiles y pervaporación',
        description='Ejecuta los cálculos pedidos sobre una o más '
                    'trayectorias, con una sola pasada por cada una. Los '
                    'resultados se escriben junto a cada trayectoria, '
                    'como <base>.<cálculo>.<especie>.dat.',
        epilog=_epilogo_analizar(),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    analizar.add_argument(
        'trayectorias', nargs='+', metavar='TRAYECTORIA',
        help='trayectorias netCDF a analizar')
    analizar.add_argument(
        '-c', '--calculos', metavar='LISTA',
        type=_lista(tuple(CALCULOS), 'cálculo'),
        default=list(CALCULOS_PREDETERMINADOS),
        help='cálculos separados por comas (ver la lista abajo)')
    analizar.add_argument(
        '-e', '--especies', metavar='LISTA', type=_lista(TODAS, 'especie'),
        default=None,
        help='restringe cada cálculo a estas especies, separadas por comas')
    analizar.add_argument(
        '--configuracion', metavar='ARCHIVO', default=None,
        help='archivo de configuración (sólo con una trayectoria); por '
             f'defecto <base>{SUFIJO_CONFIGURACION}')
    analizar.add_argument(
        '-o', '--directorio-salida', metavar='DIRECTORIO', default=None,
        help='directorio de los resultados; por defecto, el de cada '
             'trayectoria')
    analizar.add_argument(
        '--fraccion-bin-r', type=_positivo(float), metavar='FRACCION',
        default=FRACCION_BIN_R_PREDETERMINADA,
        help='ancho de los bins radiales en unidades de sigma_ref '
             '(por defecto: %(default)s)')
    analizar.add_argument(
        '--sigma-ref', type=_positivo(float), metavar='SIGMA',
        default=SIGMA_REF_PREDETERMINADO,
        help='sigma de referencia en Å (por defecto: %(default)s)')
    analizar.add_argument(
        '--lado', type=int, choices=(1, -1), default=1,
        help='reservorio de destino de la pervaporación, en el sentido '
             'del eje (por defecto: %(default)s)')
    analizar.add_argument(
        '--paso-integracion', type=_positivo(float), default=None,
        metavar='FS',
        help='paso de integración en fs, si la trayectoria no lo registra')
    analizar.add_argument(
        '--tamanio-bloque', type=_positivo(int),
        default=TAMANIO_BLOQUE_PREDETERMINADO, metavar='CUADROS',
        help='cuadros leídos por vez (por defecto: %(default)s)')
    analizar.add_argument(
        '-q', '--silencioso', action='store_true',
        help='no mostrar el resumen; los avisos y errores se siguen '
             'mostrando en la salida de errores')
    analizar.set_defaults(funcion=_analizar, parser=analizar)

    #/ recortar
    recortar = subparsers.add_parser(
        'recortar',
        help='copiar los primeros cuadros de una trayectoria',
        description='Copia los primeros N cuadros de una trayectoria '
                    'netCDF a un archivo nuevo, opcionalmente excluyendo '
                    'variables.',
        epilog='ejemplos:\n'
               '  flujo-nt recortar entrada.nc -l\n'
               '  flujo-nt recortar entrada.nc salida.nc -n 2\n'
               "  flujo-nt recortar entrada.nc salida.nc -n 2 "
               "-x 'Velocity Magnitude'",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    recortar.add_argument('entrada', help='trayectoria netCDF de origen')
    recortar.add_argument('salida', nargs='?', default=None,
                          help='trayectoria netCDF a crear')
    recortar.add_argument(
        '-n', '--cuadros', type=_positivo(int),
        default=CUADROS_PREDETERMINADOS,
        help='cantidad de cuadros a copiar (por defecto: %(default)s)')
    recortar.add_argument(
        '-x', '--excluir', nargs='+', default=[], metavar='VARIABLE',
        help='variables que no se copian')
    recortar.add_argument(
        '-d', '--dimension-cuadro', default='frame',
        help='nombre de la dimensión temporal (por defecto: %(default)s)')
    recortar.add_argument(
        '-f', '--formato', default=None, choices=FORMATOS,
        help='formato de salida (por defecto: el de la entrada)')
    recortar.add_argument(
        '-l', '--listar', action='store_true',
        help='sólo listar el contenido de la entrada')
    recortar.add_argument(
        '-q', '--silencioso', action='store_true',
        help='no mostrar el avance')
    recortar.set_defaults(funcion=_recortar, parser=recortar)

    return parser


#/ ---- SUBCOMANDO analizar ---------------------------------------------

def _planificar(calculos, especies):
    """
    Especies de cada cálculo pedido.

    Returns
    -------
    list of (str, list of Especie)
        Cálculos con al menos una especie, en el orden pedido. Los que
        quedan sin especies al aplicar el filtro se omiten con un aviso.
    """
    plan = []
    for nombre in calculos:
        codigos = [codigo for codigo in CALCULOS[nombre].especies
                   if especies is None or codigo in especies]
        if not codigos:
            _aviso(f'se omite {nombre}: no se calcula para '
                   f'{",".join(especies)} (especies posibles: '
                   f'{",".join(CALCULOS[nombre].especies)})')
            continue
        plan.append((nombre, [buscar_especie(c) for c in codigos]))
    return plan


def _abrir(ruta, opciones):
    """
    Apertura de una trayectoria con su configuración.

    Returns
    -------
    trayectoria : Trayectoria
    ruta_configuracion : str or None
        Configuración usada, o None si no se encontró.
    """
    if opciones.configuracion is not None:
        ruta_configuracion = opciones.configuracion
    else:
        ruta_configuracion = (base_desde_trayectoria(ruta)
                              + SUFIJO_CONFIGURACION)

        if not os.path.isfile(ruta_configuracion):
            try:
                trayectoria = Trayectoria(ruta)
            except ErrorTrayectoria as error:
                if not os.path.isfile(ruta):
                    raise
                raise ErrorTrayectoria(
                    f'{error} No se encontró {ruta_configuracion}; '
                    f'indicarla con --configuracion.') from None
            _aviso(f'{ruta}: no se encontró {ruta_configuracion}; se usan '
                   f'las longitudes de celda de la trayectoria')
            return trayectoria, None

    configuracion = Configuracion(ruta_configuracion)
    return Trayectoria(ruta, configuracion=configuracion), ruta_configuracion


def _analizar_trayectoria(ruta, plan, opciones, informar):
    """
    Análisis de una trayectoria y escritura de sus resultados.
    """
    inicio = time.perf_counter()
    trayectoria, ruta_configuracion = _abrir(ruta, opciones)

    with trayectoria as tr:
        calculos = [nombre for nombre, _ in plan]

        if ('pervaporacion' in calculos and opciones.paso_integracion is None
                and tr.paso_integracion is None):
            raise ErrorAnalisis(
                'la trayectoria no registra el paso de integración, '
                'necesario para la pervaporación; indicarlo con '
                '--paso-integracion.')

        tubo = Tubo.desde_trayectoria(tr)

        analisis = [uno
                    for nombre, especies in plan
                    for especie in especies
                    for uno in CALCULOS[nombre].construir(
                        especie, tubo, opciones)]

        informar(f'  configuración: {ruta_configuracion or "ninguna"}')
        informar(f'  {tr.n_cuadros} cuadros; tubo en {tubo.marco.eje}: '
                 f'largo {tubo.largo:.4f} Å, radio {tubo.radio:.4f} Å')
        for uno in analisis:
            if isinstance(uno, (DensidadRadial, PerfilVelocidad)):
                informar(f'  bins radiales: {uno.ancho_bin:.4f} Å '
                         f'(fraccion_bin_r {uno.fraccion_bin_r:g} × '
                         f'sigma_ref {uno.sigma_ref:g} Å)')
                break
        for uno in analisis:
            if isinstance(uno, DensidadLongitudinal):
                informar(f'  bins longitudinales: {uno.ancho_bin:.4f} Å '
                         f'(largo / {uno.numero_bins})')
                break

        resultados = recorrer(tr, analisis,
                              tamanio_bloque=opciones.tamanio_bloque)

    base = base_desde_trayectoria(ruta)
    if opciones.directorio_salida is not None:
        base = os.path.join(opciones.directorio_salida,
                            os.path.basename(base))

    for resultado in resultados:
        destino = nombre_archivo(base, resultado)
        if os.path.exists(destino):
            _aviso(f'se sobrescribe {destino}')
        escribir_resultado(resultado, base, fuente=ruta)
        informar(f'  escrito {destino}')

    informar(f'  terminado en {time.perf_counter() - inicio:.1f} s')


def _analizar(opciones, parser):
    """
    Subcomando ``analizar``. Devuelve el código de salida.
    """
    #/ Una misma ruta puede aparecer dos veces si los patrones se solapan
    trayectorias = list(dict.fromkeys(opciones.trayectorias))

    if opciones.configuracion is not None and len(trayectorias) > 1:
        parser.error('--configuracion sólo se admite con una trayectoria')

    plan = _planificar(opciones.calculos, opciones.especies)
    if not plan:
        _error('ningún cálculo corresponde a las especies pedidas')
        return 2

    def informar(texto):
        if not opciones.silencioso:
            print(texto, flush=True)

    fallidas = []
    for numero, ruta in enumerate(trayectorias, start=1):
        informar(f'[{numero}/{len(trayectorias)}] {ruta}')
        try:
            _analizar_trayectoria(ruta, plan, opciones, informar)
        except (ErrorTrayectoria, ErrorGeometria, ErrorAnalisis,
                OSError) as error:
            _error(f'{ruta}: {error}')
            fallidas.append(ruta)

    if fallidas:
        _error(f'{len(fallidas)} de {len(trayectorias)} trayectorias '
               f'fallaron:')
        for ruta in fallidas:
            print(f'  {ruta}', file=sys.stderr)
        return 1

    return 0


#/ ---- SUBCOMANDO recortar ---------------------------------------------

def _recortar(opciones, parser):
    """
    Subcomando ``recortar``. Devuelve el código de salida.
    """
    if not os.path.isfile(opciones.entrada):
        _error(f'no existe {opciones.entrada}')
        return 1

    if opciones.listar:
        listar_contenido(opciones.entrada)
        return 0

    if opciones.salida is None:
        parser.error('falta el archivo de salida (o -l para listar)')

    try:
        inexistentes = recortar_trayectoria(
            opciones.entrada, opciones.salida, opciones.cuadros,
            excluir=opciones.excluir,
            dimension_cuadro=opciones.dimension_cuadro,
            formato=opciones.formato,
            verboso=not opciones.silencioso)
    except (ValueError, OSError) as error:
        _error(str(error))
        return 1

    for nombre in inexistentes:
        _aviso(f'la variable a excluir {nombre!r} no está en la entrada')

    return 0


#/ ---- PUNTO DE ENTRADA -------------------------------------------------

def main(argumentos=None):
    """
    Punto de entrada de ``flujo-nt``.

    Parameters
    ----------
    argumentos : list of str, optional
        Argumentos de la línea de comandos, sin el nombre del programa. Por
        defecto, los de ``sys.argv``.

    Returns
    -------
    int
        Código de salida: 0 si todo terminó bien, 1 si falló alguna
        trayectoria y 2 si los argumentos son inválidos.
    """
    parser = construir_parser()
    opciones = parser.parse_args(argumentos)
    return opciones.funcion(opciones, opciones.parser)


#/ ---- FIN MÓDULO --------------------------------------------------------
