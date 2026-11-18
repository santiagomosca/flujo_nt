#! /usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Recorte de trayectorias netCDF (convención AMBER, de LAMMPS u OVITO).

Copia los primeros N cuadros de una trayectoria a un archivo nuevo, con la
opción de excluir variables. La copia se hace cuadro por cuadro, de modo que el
consumo de memoria es independiente del tamaño de la trayectoria de entrada.

Equivalente aproximado de:
    ncks -d frame,0,N-1 entrada.nc salida.nc

Ejemplos de uso:
    ./recortar_trayectoria.py entrada.nc salida.nc -n 6
    ./recortar_trayectoria.py entrada.nc salida.nc -n 6 -x 'Velocity Magnitude'
    ./recortar_trayectoria.py entrada.nc -l

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: julio de 2026 / revisión noviembre de 2026
"""

import argparse
import os
import sys

from netCDF4 import Dataset


#/ ---- FUNCIONES DEL PROGRAMA -------------------------------------------

def listar_contenido(archivo_entrada):
    """
    Impresión del contenido de la trayectoria: atributos globales,
    dimensiones y variables.
    """
    with Dataset(archivo_entrada, 'r') as ds:
        print(f'\nArchivo: {archivo_entrada}')
        print(f'Formato: {ds.data_model}\n')

        print('Atributos globales:')
        for atributo in ds.ncattrs():
            print(f'\t{atributo} = {getattr(ds, atributo)!r}')

        print('\nDimensiones:')
        for dimension in ds.dimensions.values():
            ilimitada = ' (ilimitada)' if dimension.isunlimited() else ''
            print(f'\t{dimension.name:16s} {dimension.size}{ilimitada}')

        print('\nVariables:')
        for variable in ds.variables.values():
            atributos = {a: getattr(variable, a) for a in variable.ncattrs()}
            texto_atributos = f' {atributos}' if atributos else ''
            print(f'\t{variable.name:20s} {str(variable.dtype):8s} '
                  f'{variable.dimensions}{texto_atributos}')
        print()

    return 0


def recortar_trayectoria(archivo_entrada, archivo_salida, numero_cuadros,
        excluir=(), dimension_cuadro='frame', formato=None, verboso=True):
    """
    Copia de los primeros 'numero_cuadros' cuadros de una trayectoria netCDF.

    Argumentos:
        archivo_entrada: trayectoria netCDF de origen.
        archivo_salida: trayectoria netCDF a crear.
        numero_cuadros: cantidad de cuadros a copiar desde el inicio.
        excluir: nombres de variables a omitir en la copia.
        dimension_cuadro: nombre de la dimensión temporal.
        formato: formato netCDF de salida. Si es None se usa el de entrada.
        verboso: impresión de progreso.
    """
    excluir = set(excluir)

    with Dataset(archivo_entrada, 'r') as origen:
        #/ Se evita el enmascarado automático: los datos se copian tal cual
        origen.set_auto_maskandscale(False)

        if dimension_cuadro not in origen.dimensions:
            raise ValueError(
                f'La dimensión {dimension_cuadro!r} no existe en '
                f'{archivo_entrada}. Dimensiones presentes: '
                f'{list(origen.dimensions)}')

        cuadros_disponibles = origen.dimensions[dimension_cuadro].size

        if numero_cuadros > cuadros_disponibles:
            raise ValueError(
                f'Se pidieron {numero_cuadros} cuadros pero la trayectoria '
                f'tiene {cuadros_disponibles}')

        #/ Verificación de que las variables a excluir existen
        inexistentes = excluir - set(origen.variables)
        if inexistentes:
            print(f'ADVERTENCIA: variables a excluir no presentes en la '
                  f'trayectoria: {sorted(inexistentes)}\n', file=sys.stderr)

        if formato is None:
            formato = origen.data_model

        if verboso:
            print(f'\nLeyendo {archivo_entrada}')
            print(f'\tFormato: {formato}')
            print(f'\tCuadros disponibles: {cuadros_disponibles}')
            print(f'\tCuadros a copiar: {numero_cuadros}\n')

        with Dataset(archivo_salida, 'w', format=formato) as destino:
            destino.set_auto_maskandscale(False)

            #/ Copia de atributos globales
            if verboso:
                print('Copiando atributos globales\n')
            destino.setncatts(
                {a: getattr(origen, a) for a in origen.ncattrs()})

            #/ Copia de dimensiones. La dimensión de cuadros se crea ilimitada.
            if verboso:
                print('Creando dimensiones\n')
            for dimension in origen.dimensions.values():
                if dimension.isunlimited():
                    tamanio = None
                else:
                    tamanio = dimension.size
                destino.createDimension(dimension.name, tamanio)

            #/ Creación de variables con sus atributos
            if verboso:
                print('Creando variables\n')
            for variable in origen.variables.values():
                if variable.name in excluir:
                    if verboso:
                        print(f'\tOmitiendo variable {variable.name!r}\n')
                    continue

                variable_nueva = destino.createVariable(
                    variable.name,
                    variable.dtype,
                    variable.dimensions)

                #/ El enmascarado y escalado automáticos deben desactivarse
                #/ sobre cada variable recién creada: Dataset.set_auto_* sólo
                #/ afecta a las variables existentes al momento de llamarlo.
                #/ Sin esto, una variable con 'scale_factor' (p. ej. 'time' en
                #/ trayectorias de LAMMPS) recibiría los valores crudos del
                #/ origen y netCDF los volvería a dividir por el factor,
                #/ corrompiendo la copia en silencio.
                variable_nueva.set_auto_maskandscale(False)

                variable_nueva.setncatts(
                    {a: getattr(variable, a) for a in variable.ncattrs()})

            #/ Copia de datos
            if verboso:
                print('Copiando datos\n')
            for nombre, variable in origen.variables.items():
                if nombre in excluir:
                    continue

                if dimension_cuadro in variable.dimensions:
                    #/ En la convención AMBER la dimensión de cuadros es la
                    #/ primera. Se copia de a un cuadro por vez para que el
                    #/ consumo de memoria no dependa del largo de la
                    #/ trayectoria.
                    if variable.dimensions[0] != dimension_cuadro:
                        raise ValueError(
                            f'La variable {nombre!r} tiene la dimensión '
                            f'{dimension_cuadro!r} en una posición no '
                            f'contemplada: {variable.dimensions}')

                    if verboso:
                        print(f'\tCopiando variable {nombre!r} '
                              f'({numero_cuadros} cuadros)\n')

                    for cuadro in range(numero_cuadros):
                        destino.variables[nombre][cuadro] = variable[cuadro]

                else:
                    #/ Variables estáticas: se copian completas
                    if verboso:
                        print(f'\tCopiando variable estática {nombre!r}\n')
                    destino.variables[nombre][:] = variable[:]

    if verboso:
        tamanio_entrada = os.path.getsize(archivo_entrada) / 1024**2
        tamanio_salida = os.path.getsize(archivo_salida) / 1024**2
        print(f'Escrito {archivo_salida}')
        print(f'\t{tamanio_entrada:.1f} MB -> {tamanio_salida:.1f} MB\n')

    return 0


def opciones():
    """
    Parseo de argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description='Recorte de trayectorias netCDF (convención AMBER). '
                    'Copia los primeros N cuadros, opcionalmente excluyendo '
                    'variables.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Ejemplos:\n'
               '  %(prog)s entrada.nc -l\n'
               '  %(prog)s entrada.nc salida.nc -n 6\n'
               "  %(prog)s entrada.nc salida.nc -n 6 "
               "-x 'Velocity Magnitude'\n")

    parser.add_argument('entrada',
        help='trayectoria netCDF de origen')

    parser.add_argument('salida', nargs='?', default=None,
        help='trayectoria netCDF a crear')

    parser.add_argument('-n', '--cuadros', type=int, default=6,
        help='cantidad de cuadros a copiar (predeterminado: 6)')

    parser.add_argument('-x', '--excluir', nargs='*', default=[],
        metavar='VAR',
        help='nombres de variables a excluir de la copia')

    parser.add_argument('-d', '--dimension-cuadro', default='frame',
        help="nombre de la dimensión temporal (predeterminado: 'frame')")

    parser.add_argument('-f', '--formato', default=None,
        choices=('NETCDF3_CLASSIC', 'NETCDF3_64BIT_OFFSET',
                 'NETCDF3_64BIT_DATA', 'NETCDF4_CLASSIC', 'NETCDF4'),
        help='formato de salida (predeterminado: el mismo de la entrada)')

    parser.add_argument('-l', '--listar', action='store_true',
        help='sólo listar el contenido de la trayectoria de entrada y salir')

    parser.add_argument('-q', '--silencioso', action='store_true',
        help='no imprimir progreso')

    return parser.parse_args()


def main():
    opts = opciones()

    if not os.path.isfile(opts.entrada):
        print(f'ERROR: no existe el archivo {opts.entrada}', file=sys.stderr)
        return 1

    if opts.listar:
        return listar_contenido(opts.entrada)

    if opts.salida is None:
        print('ERROR: falta el archivo de salida (o usar -l para listar)',
              file=sys.stderr)
        return 1

    if os.path.abspath(opts.entrada) == os.path.abspath(opts.salida):
        print('ERROR: el archivo de salida no puede ser el de entrada',
              file=sys.stderr)
        return 1

    try:
        recortar_trayectoria(
            opts.entrada,
            opts.salida,
            opts.cuadros,
            excluir=opts.excluir,
            dimension_cuadro=opts.dimension_cuadro,
            formato=opts.formato,
            verboso=not opts.silencioso)
    except (ValueError, OSError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1

    return 0


#/ ---- EJECUCIÓN DEL SCRIPT ----------------------------------------------
if __name__ == '__main__':
    sys.exit(main())


#/ ---- FIN SCRIPT --------------------------------------------------------
