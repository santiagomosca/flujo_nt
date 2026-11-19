#! /usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Recorte de trayectorias netCDF (convención AMBER, de LAMMPS u OVITO).

Copia los primeros N cuadros de una trayectoria a un archivo nuevo, con la
opción de excluir variables. La copia se hace cuadro por cuadro, de modo que
el consumo de memoria es independiente del tamaño de la trayectoria de
entrada. Es el equivalente aproximado de::

    ncks -d frame,0,N-1 entrada.nc salida.nc

Desde la línea de comandos se usa como ``flujo-nt recortar``.

Examples
--------
>>> recortar_trayectoria('entrada.nc', 'salida.nc', 2)
>>> recortar_trayectoria('entrada.nc', 'salida.nc', 2,
...                      excluir=['Velocity Magnitude'])

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: julio de 2026 / revisión noviembre de 2026
"""

import os
import sys

from netCDF4 import Dataset


#/ ---- CONSTANTES DEL MÓDULO --------------------------------------------

CUADROS_PREDETERMINADOS = 6
"""int: Cantidad de cuadros que se copian si no se indica otra."""

FORMATOS = ('NETCDF3_CLASSIC', 'NETCDF3_64BIT_OFFSET', 'NETCDF3_64BIT_DATA',
            'NETCDF4_CLASSIC', 'NETCDF4')
"""tuple of str: Formatos netCDF de salida admitidos."""


#/ ---- FUNCIONES --------------------------------------------------------

def listar_contenido(archivo, salida=None):
    """
    Impresión del contenido de una trayectoria netCDF.

    Parameters
    ----------
    archivo : str
        Trayectoria netCDF.
    salida : file-like, optional
        Destino del texto. Por defecto la salida estándar.

    Notes
    -----
    Se imprimen los atributos globales, las dimensiones y las variables con
    su tipo, sus dimensiones y sus atributos.
    """
    salida = sys.stdout if salida is None else salida

    def imprimir(texto=''):
        print(texto, file=salida)

    with Dataset(archivo, 'r') as ds:
        imprimir(f'Archivo: {archivo}')
        imprimir(f'Formato: {ds.data_model}')

        imprimir('\nAtributos globales:')
        for atributo in ds.ncattrs():
            imprimir(f'\t{atributo} = {getattr(ds, atributo)!r}')

        imprimir('\nDimensiones:')
        for dimension in ds.dimensions.values():
            ilimitada = ' (ilimitada)' if dimension.isunlimited() else ''
            imprimir(f'\t{dimension.name:16s} {dimension.size}{ilimitada}')

        imprimir('\nVariables:')
        for variable in ds.variables.values():
            atributos = {a: getattr(variable, a) for a in variable.ncattrs()}
            texto_atributos = f' {atributos}' if atributos else ''
            imprimir(f'\t{variable.name:20s} {str(variable.dtype):8s} '
                     f'{variable.dimensions}{texto_atributos}')


def recortar_trayectoria(archivo_entrada, archivo_salida,
        numero_cuadros=CUADROS_PREDETERMINADOS, excluir=(),
        dimension_cuadro='frame', formato=None, verboso=False):
    """
    Copia de los primeros cuadros de una trayectoria netCDF.

    Parameters
    ----------
    archivo_entrada : str
        Trayectoria netCDF de origen.
    archivo_salida : str
        Trayectoria netCDF a crear. Si existe, se reemplaza.
    numero_cuadros : int, optional
        Cantidad de cuadros a copiar desde el inicio. Por defecto 6.
    excluir : iterable of str, optional
        Nombres de las variables que no se copian.
    dimension_cuadro : str, optional
        Nombre de la dimensión temporal. Por defecto ``'frame'``.
    formato : str, optional
        Formato netCDF de salida, uno de :data:`FORMATOS`. Si es None se usa
        el de la entrada.
    verboso : bool, optional
        Si es verdadero, se informa el avance en la salida estándar.

    Returns
    -------
    list of str
        Variables a excluir que no estaban en la trayectoria.

    Raises
    ------
    ValueError
        Si la entrada y la salida son el mismo archivo, si no existe la
        dimensión temporal, si se piden más cuadros de los que hay o si
        alguna variable tiene la dimensión temporal fuera de la primera
        posición.

    Notes
    -----
    Los datos se copian crudos, sin enmascarado ni escalado. El escalado
    debe desactivarse sobre cada variable recién creada, porque
    ``Dataset.set_auto_maskandscale`` sólo afecta a las variables que
    existen al momento de llamarlo. Sin esto, una variable con
    ``scale_factor`` (por ejemplo ``time`` en las trayectorias de LAMMPS)
    recibiría los valores crudos del origen y netCDF4 los volvería a
    dividir por el factor.
    """
    if os.path.abspath(archivo_entrada) == os.path.abspath(archivo_salida):
        raise ValueError('El archivo de salida no puede ser el de entrada.')

    excluir = set(excluir)

    with Dataset(archivo_entrada, 'r') as origen:
        origen.set_auto_maskandscale(False)

        if dimension_cuadro not in origen.dimensions:
            raise ValueError(
                f'La dimensión {dimension_cuadro!r} no existe en '
                f'{archivo_entrada}. Dimensiones presentes: '
                f'{list(origen.dimensions)}')

        cuadros_disponibles = origen.dimensions[dimension_cuadro].size

        if not 0 < numero_cuadros <= cuadros_disponibles:
            raise ValueError(
                f'Se pidieron {numero_cuadros} cuadros pero la trayectoria '
                f'tiene {cuadros_disponibles}.')

        inexistentes = sorted(excluir - set(origen.variables))

        if formato is None:
            formato = origen.data_model

        if verboso:
            print(f'Leyendo {archivo_entrada} ({formato}): copiando '
                  f'{numero_cuadros} de {cuadros_disponibles} cuadros')

        with Dataset(archivo_salida, 'w', format=formato) as destino:
            destino.set_auto_maskandscale(False)
            destino.setncatts(
                {a: getattr(origen, a) for a in origen.ncattrs()})

            #/ La dimensión de cuadros se crea ilimitada, como en el origen
            for dimension in origen.dimensions.values():
                tamanio = None if dimension.isunlimited() else dimension.size
                destino.createDimension(dimension.name, tamanio)

            for variable in origen.variables.values():
                if variable.name in excluir:
                    continue

                if (dimension_cuadro in variable.dimensions
                        and variable.dimensions[0] != dimension_cuadro):
                    raise ValueError(
                        f'La variable {variable.name!r} tiene la dimensión '
                        f'{dimension_cuadro!r} en una posición no '
                        f'contemplada: {variable.dimensions}')

                nueva = destino.createVariable(
                    variable.name, variable.dtype, variable.dimensions)
                nueva.set_auto_maskandscale(False)
                nueva.setncatts(
                    {a: getattr(variable, a) for a in variable.ncattrs()})

            for nombre, variable in origen.variables.items():
                if nombre in excluir:
                    continue

                if dimension_cuadro in variable.dimensions:
                    for cuadro in range(numero_cuadros):
                        destino.variables[nombre][cuadro] = variable[cuadro]
                else:
                    destino.variables[nombre][:] = variable[:]

    if verboso:
        tamanio_entrada = os.path.getsize(archivo_entrada) / 1024**2
        tamanio_salida = os.path.getsize(archivo_salida) / 1024**2
        print(f'Escrito {archivo_salida} ({tamanio_entrada:.1f} MB -> '
              f'{tamanio_salida:.1f} MB)')

    return inexistentes


#/ ---- FIN MÓDULO --------------------------------------------------------
