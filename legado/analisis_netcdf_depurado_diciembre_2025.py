#! /usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: abril de 2021 / revisión diciembre 2025
"""

import argparse
import os
import numpy as np
from netCDF4 import Dataset
#/ TODO: import MDAnalysis as mda
try:
    from scipy.spatial.transform import Rotation as R
    rotacion_euler = True
except:
    ImportError
    rotacion_euler = False
try:
    from termcolor import colored, cprint
    terminal_colorida = True
except:
    ImportError
    terminal_colorida = False



#/ ---- FUNCIONES DEL PROGRAMA ------------------------------------------------
def main():
    informacion = (
        f'Archivo {os.path.basename(__file__)}.' +
        'Contiene funciones útiles para otros scripts de cálculo y análisis.')
    print(informacion)


#/ ---- FUNCIONES PARA LECTURA Y ESCRITURA DE ARCHIVOS ------------------------
def obtener_nombres(
    directorio_base,
    radio_nominal,
    largo_nominal,
    fraccion_molar,
    directorio_tipo,
    fuerza,
    direccion_campo_electrico,
    valor_campo_electrico,
    nombre,
    nombre_trayectoria,
    sufijo_trayectoria):
    """
    Creación de rutas para archivos de configuración, trayectoria e
    imágenes.
    """

    def reemplazo(valor, fraccion_molar=False, fuerza=False, campo_electrico=False):
        """
        Chequeo si la parte decimal del radio o largo es igual a 0.
        Hay que cambiar nomenclatura general para que sea consistente,
        ej: R3-0 -> R3-00.
        """
        if fraccion_molar:
            salida = f'{valor:.3f}'
        elif fuerza:
            salida = f'{valor:n}'
        elif campo_electrico:
            if valor == 0:
                salida = ''
            else:
                salida = f'-{valor:.0e}'
        else:
            if valor % 1 == 0:
                salida = f'{valor:.1f}'
            else:
                salida = f'{valor:.3g}'

        if campo_electrico:
            salida = salida.replace('e-0','eneg')
        else:
            salida = salida.replace('.','-')

        return salida

    #/ Reemplazo de valores por palabras en cada opción de simulación
    r = reemplazo(radio_nominal)
    l = reemplazo(largo_nominal)
    xm = reemplazo(fraccion_molar, fraccion_molar=True)

    if valor_campo_electrico != None:
        dirE = direccion_campo_electrico
        valE = reemplazo(valor_campo_electrico, campo_electrico=True)

    if fuerza != None:
        f = reemplazo(fuerza, fuerza=True)

    directorio_simulacion = os.path.join(directorio_base, f'R{r}', f'L{l}',
        f'X{xm}', directorio_tipo)

    #/ Construcción de directorios y nombres completos de simulación
    if valor_campo_electrico != None:
        directorio_simulacion = os.path.join(
            directorio_simulacion,
            f'E{dirE}{valE}',
            directorio_tipo)

    if fuerza != None:
        directorio_simulacion = os.path.join(directorio_simulacion, f'f.{f}')
        nombre_trayectoria = f'f.{f}.' + nombre_trayectoria

    #/ Construcción de rutas al archivo de trayectoria y archivos de salida
    if valor_campo_electrico != None:
        archivo_trayectoria = nombre.format(
            r,l,xm,f'E{dirE}' + f'{valE}' + '.' + nombre_trayectoria) \
            + f'.{sufijo_trayectoria}'
        archivo_salida  = nombre.format(
            r,l,xm,f'E{dirE}' + f'{valE}' + '.' + nombre_trayectoria)
    else:
        archivo_trayectoria = nombre.format(r,l,xm,nombre_trayectoria) \
            + f'.{sufijo_trayectoria}'
        archivo_salida  = nombre.format(r,l,xm,nombre_trayectoria)

    direccion_trayectoria = os.path.join(directorio_simulacion,
        nombre_trayectoria,archivo_trayectoria)
    direccion_salida = os.path.join(directorio_simulacion,
        nombre_trayectoria,archivo_salida)

    return direccion_trayectoria,direccion_salida


def leer_trayectoria(archivo_netcdf, copiar_atributos=False,
    copiar_dimensiones=False, copiar_variables=True, solo_analisis=True,
    sin_imagenes=False, sin_fuerzas=False):
    """
    Lectura de archivo de trayectoria. Actualmente debe ser una trayectoria
    netCDF.
    Devuelve de forma predeterminada un diccionario con todos los datos.

    Argumentos:
        archivo_netcdf: trayectoria netCDF generada con LAMMPS.
    """

    print('Intentando abrir archivo netCDF\n')
    dicc = {}

    try:
        with Dataset(archivo_netcdf, 'r') as ds:
            print('\tApertura exitosa\n')

            #/ Copia de los atributos del archivo de trayectoria
            if copiar_atributos:
                print('\tCopiando atributos de trayectoria a diccionario\n')
                dicc['atributos'] = ds.__dict__

            #/ Copia de las dimensiones del archivo de trayectoria
            if copiar_dimensiones:
                print('\tCopiando dimensiones de trayectoria a diccionario\n')
                dicc['dimensiones'] = {}
                for dim in ds.dimensions:
                    print(f'\t\tCopiando dimensión {dim}\n')
                    dicc['dimensiones'][dim] = {}
                    dicc['dimensiones'][dim]['name'] = ds.dimensions[dim].name
                    dicc['dimensiones'][dim]['size'] = ds.dimensions[dim].size

            #/ Copia de las variables del archivo de trayectoria
            if copiar_variables:
                print('\tCopiando variables de trayectoria a diccionario\n')
                dicc['variables'] = {}
                for var in ds.variables:
                    if solo_analisis: # evitar variables no usadas
                        if var in ('spatial','cell_spatial','cell_angular',
                        'cell_angles'):
                            continue
                    if sin_imagenes: # evitar imágenes periódicas
                        if var in ('ix','iy','iz'):
                            continue
                    if sin_fuerzas: # evitar fuerzas sobre partículas
                        if var == 'forces':
                            continue
                    print(f'\t\tCopiando variable {var}\n')
                    dicc['variables'][var] = {}
                    dicc['variables'][var]['name'] = ds.variables[var].name
                    dicc['variables'][var]['dtype'] = ds.variables[var].dtype
                    dicc['variables'][var]['dimensions'] = \
                        ds.variables[var].dimensions
                    dicc['variables'][var]['data'] = ds.variables[var][:]

                    ## #/ Alternativa para no copiar arrays repetidos
                    ## if ds.variables[var].data.dtype in ('float32','int32') \
                    ## and np.isclose(ds.variables[var][:].data,
                    ## ds.variables[var][0]).all():
                    ##     dicc[var]['data'] = ds.variables[var][0]
                    ## else:
                    ##     dicc[var]['data'] = ds.variables[var][:]

                    for attr in ('units', 'scale_factor'):
                        try:
                            dicc['variables'][var][attr] = \
                                getattr(ds.variables[var],attr)
                        except:
                            pass

    except OSError as os_error:
        print(os_error)
        exit(1)
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        exit(1)

    return dicc


def escribir_trayectoria(dicc_trayectoria, direccion_salida,
        sufijo_archivo, formato='NETCDF3_64BIT_OFFSET'):
    """
    Escritura de archivo de trayectoria en formato netCDF compatible
    con VMD.

    Argumentos:
        dicc_trayectoria: diccionario con datos de trayectoria.
    """
    #/ Asignación de los datos usados
    atributos = dicc_trayectoria['atributos']
    dimensiones = dicc_trayectoria['dimensiones']
    variables = dicc_trayectoria['variables']

    with Dataset(direccion_salida+'.'+sufijo_archivo+'.nc','w',
        format=formato) as ds:
        #/ Creación de atributos
        print('\tCopiando atributos de diccionario a trayectoria\n')
        for attr in atributos:
            print(f'\t\tCreando atributo {attr}\n')
            setattr(ds, attr, atributos[attr])

        #/ Creación de dimensiones
        print('\tCopiando dimensiones de diccionario a trayectoria\n')
        for dim in dimensiones:
            print(f'\t\tCreando dimension {dim}\n')
            if dim == 'frame':
                ds.createDimension(dimensiones[dim]['name'],None)
            else:
                ds.createDimension(
                    dimensiones[dim]['name'],
                    dimensiones[dim]['size'])

        #/ Creación de variables
        print('\tCreando variables de diccionario a trayectoria\n')
        for var in variables:
            print(f'\t\tCreando variable {var}\n')
            ds.createVariable(
                variables[var]['name'],
                variables[var]['dtype'],
                variables[var]['dimensions'])

            for varattr in ('units', 'scale_factor'):
                try:
                    setattr(ds.variables[var],varattr,variables[var][varattr])
                except:
                    continue

        print('\tCopiando variables de diccionario a trayectoria\n')
        for var in variables:
            print(f'\t\tCopiando datos de variable {var}\n')
            ds.variables[var][:] = variables[var]['data']

    return 0


def escribir_calculo(direccion_salida, sufijo_archivo, tipo_calculo, calculos,
    nombre_particular=None, columnas=None, unidades=None, especial=None, formato=None):
    """
    Escribir a archivo de texto plano los resultados de cálculos
    efectuados.
    """

    #/ Columnas para encabezado de archivo de salida
    encabezado_columnas = ''
    if columnas != None:
        for columna in columnas:
            encabezado_columnas += f' {columna}'

    #/ Unidades para encabezado de archivo de salida
    encabezado_unidades = ''
    if unidades != None:
        for unidad in unidades.keys():
            encabezado_unidades += f' {unidad} '
            encabezado_unidades += '[{}],'.format(unidades[unidad])
        encabezado_unidades = encabezado_unidades[:-1] # remover última coma

    #/ Texto especial para encabezado de archivo de salida
    encabezado_especial = ''
    if especial != None:
        encabezado_especial = especial

    encabezado_texto = (
        f'Cálculo de {tipo_calculo}\n' +
        f'Unidades:'+encabezado_unidades+'\n' +
        encabezado_especial+'\n' +
        encabezado_columnas
        )

    #/ Arreglo con datos de salida
    if hasattr(calculos,'shape'):
        salida = calculos
    else:
        salida = np.column_stack(calculos)

    ## formato = ['%d']
    ## for n in range(salida[:,1:].shape[1]):
    ##     formato.append('%.5e')
    if formato == None:
        formato = '%.5e'

    if nombre_particular != None:
        archivo_salida = f'.{tipo_calculo}'+f'.{nombre_particular}' \
            + f'.{sufijo_archivo}'
    else:
        archivo_salida = f'.{tipo_calculo}'+f'.{sufijo_archivo}'

    np.savetxt(direccion_salida+archivo_salida,salida,fmt=formato,
        header=encabezado_texto)

    return direccion_salida+archivo_salida


#/ ---- FUNCIONES PARA MANIPULACIÓN Y CREACIÓN DE TRAYECTORIAS ----------------
def obtener_longitudes_celda_desde_configuracion(archivo_configuracion,
    numero_lineas=30):
    """
    Obtención de longitudes de celda a partir de un archivo de
    configuración generado por LAMMPS.
    """
    print(f'Leyendo archivo de configuración {archivo_configuracion}\n')

    direcciones = ('x','y','z')

    longitudes = dict.fromkeys(direcciones)

    with open(archivo_configuracion, 'r') as f:
        for n in range(numero_lineas):
            linea = next(f)
            for direccion in direcciones:
                if linea.endswith(f'{direccion}lo {direccion}hi\n'):
                    print(f'\tBordes en {direccion} encontrados\n')
                    bordes = linea.split()
                    borde_inf = round(float(bordes[0]), 6)
                    borde_sup = round(float(bordes[1]), 6)
                    longitudes[direccion] = round(borde_sup - borde_inf, 6)

    return longitudes


def cambiar_longitud_celda(dicc_trayectoria, dicc_longitudes,
    cambiar_x=False, cambiar_y=False, cambiar_z=True):
    """
    Cambio de las dimensiones de la celda de simulación en la
    trayectoria. Requiere un diccionario con las longitudes deseadas.
    """
    rastreador_cambio = False

    longitudes_celda = dicc_trayectoria['variables']['cell_lengths']['data']

    print('\tLongitud original de celda en dirección x: {} Å\n'.format(
        longitudes_celda[0,0]))
    print('\tLongitud original de celda en dirección y: {} Å\n'.format(
        longitudes_celda[0,1]))
    print('\tLongitud original de celda en dirección z: {} Å\n'.format(
        longitudes_celda[0,2]))

    direcciones = {
        'x':{'indice':0, 'cambiar':cambiar_x},
        'y':{'indice':1, 'cambiar':cambiar_y},
        'z':{'indice':2, 'cambiar':cambiar_z}}

    for direccion in direcciones.keys():
        if direcciones[direccion]['cambiar']:
            print(f'\t\tCambiando longitud de celda en dirección {direccion}\n')
            dicc_trayectoria['variables']['cell_lengths']['data'] \
                [:,direcciones[direccion]['indice']] \
                = dicc_longitudes[direccion]
            rastreador_cambio = True

    return rastreador_cambio


def detectar_eje_tubo(longitudes_celda, longitud_tolerancia=2.84):
    """
    Identificación de qué dirección tiene mayor longitud en celda de
    simulación. Esta dirección coincide con el eje del tubo.

    Argumentos:
        longitudes_celda
        posiciones
    """
    print('Detectando lado mayor de celda de simulación\n')

    if longitudes_celda['data'].ndim == 1:
        idx = (0)
        idy = (1)
        idz = (2)
    elif longitudes_celda['data'].ndim == 2:
        idx = (0,0)
        idy = (0,1)
        idz = (0,2)
    else:
        msj = 'Error identificando lado mayor de celda de simulación'
        raise ValueError(msj)
    lxsim = longitudes_celda['data'][idx].round(6)
    lysim = longitudes_celda['data'][idy].round(6)
    lzsim = longitudes_celda['data'][idz].round(6)

    #/ Comparación de longitudes de lado
    diferencia_lz_lx = lzsim - lxsim
    diferencia_ly_lx = lysim - lxsim

    if abs(diferencia_lz_lx) > abs(diferencia_ly_lx):
        diferencia = diferencia_lz_lx
        eje_tubo = 'Z'
    else:
        diferencia = diferencia_ly_lx
        eje_tubo = 'Y'

    if abs(diferencia) > longitud_tolerancia:
        if diferencia == 0:
            eje_tubo = 'X'
        elif diferencia < 0:
            eje_tubo = 'X'
        else: # no hay cambios en eje tubo
            pass


    ## TODO: revisar algoritmo de comparación
    ## if diferencia_lz_lx > diferencia_ly_lx:
    ##     eje_tubo = 'Z'
    ##     if diferencia_lz_lx > longitud_tolerancia:
    ##         print('\tLado mayor paralelo a eje Z\n')
    ##         eje_tubo = 'Z'

    ## elif diferencia_ly_lx > diferencia_lz_lx:
    ##     eje_tubo = 'Y'
    ##     if diferencia_ly_lx > longitud_tolerancia:
    ##         print('\tLado mayor paralelo a eje Y\n')
    ##         eje_tubo = 'Y'

    ## else:
    ##     eje_tubo = 'X'

    print(f'\tLado mayor paralelo a eje {eje_tubo}\n')

    return eje_tubo


def rotar_a_eje_X(dicc_trayectoria, eje_tubo, modo='dot'):
    """
    Rotación a eje X de datos de longitud y origen de celda, posiciones y
    velocidades usando diferentes métodos. Si el eje ya coincide con la
    dirección X no se efectúa rotación.

    Dot:
    La matriz de rotación asume un giro de -90 sobre el eje Y o Z.

    Euler:Si la
    La rotación se hace +90 sobre eje Y, o -90 sobre eje Z. Necesita
    redimensionar los arrays que dimensiones superiores a dos
    (coordenadas y velocidades, que tienen (nframes, variable, vx vy
    vz)).

    Flip:
    Copia los efectos de 'Dot' pero intercambiando columnas de array, e
    invirtiendo los signos donde sea necesario.

    La toleranica en la diferencia de longitudes es el doble del enlace
    de carbono de grafeno [Å].

    Argumentos:
        dicc_trayectoria: diccionario con datos de trayectoria.
        eje_tubo
        modo
    """
    if eje_tubo == 'Z':
        rotar = True

        #/ Modo dot
        matriz_rotacion = np.array([
            [ 0, 0,-1],
            [ 0, 1, 0],
            [ 1, 0, 0]])

        matriz_intercambio_dot = np.array([
            [ 0, 0, 1],
            [ 0, 1, 0],
            [ 1, 0, 0]]) # para orígenes y longitudes de celda

        #/ Modo euler
        try:
            rotacion = R.from_euler('y', 90, degrees=True)
            indices_intercambio_euler = [0,2]
        except NameError:
            print('Rotación de Euler no disponible')

        #/ Modo flip
        indices_intercambio_flip = [0,2]
        matriz_intercambio_flip = np.array([1,-1])
        eje_original = 'z'

    elif eje_tubo == 'Y':
        rotar = True

        #/ Modo dot
        matriz_rotacion = np.array([
            [ 0,-1, 0],
            [ 1, 0, 0],
            [ 0, 0, 1]])

        matriz_intercambio_dot = np.array([
            [ 0, 1, 0],
            [ 1, 0, 0],
            [ 0, 0, 1]])

        #/ Modo euler
        rotacion = R.from_euler('z', -90, degrees=True)
        indices_intercambio_euler = [0,1]

        #/ Modo flip
        indices_intercambio_flip = [0,1]
        matriz_intercambio_flip = np.array([1,-1])
        eje_original = 'y'

    else:
        rotar = False

    if rotar:
        print('Rotando a eje X\n')

        #/ Rotación de orígenes y longitudes de celda, coordenadas y velocidades
        for var in ('cell_origin', 'cell_lengths'):
            print(f'\tRotando variable {var}\n')

            if modo == 'dot':
                dicc_trayectoria['variables'][var]['data'] \
                    = np.dot(dicc_trayectoria['variables'][var]['data'],
                        matriz_intercambio_dot)

            if modo == 'euler':
                dicc_trayectoria['variables'][var]['data'] \
                [:,indices_intercambio_euler] \
                    = dicc_trayectoria['variables'][var]['data'] \
                      [:,indices_intercambio_euler[::-1]]

            if modo == 'flip':
                dicc_trayectoria['variables'][var]['data'] \
                [...,indices_intercambio_flip] \
                    = dicc_trayectoria['variables'][var]['data'] \
                      [...,indices_intercambio_flip[::-1]]
                      # la elipsis '...' ejecuta tantos np.newaxis como
                      # dimensiones tenga la variable. El último
                      # indices_intercambio_flip invierte el orden de los
                      # índices escogidos en la última dimensión de la variable

        for var in ('coordinates', 'velocities'):
            print(f'\tRotando variable {var}\n')

            if modo == 'dot':
                dicc_trayectoria['variables'][var]['data'] \
                = np.dot(dicc_trayectoria['variables'][var]['data'],
                    matriz_rotacion)

            if modo == 'euler':
                dimensiones_originales \
                    = dicc_trayectoria['variables'][var]['data'].shape
                variable_redimensionada \
                    = dicc_trayectoria['variables'][var] \
                      ['data'].reshape(-1,dimensiones_originales[-1])
                      # -1 obtiene dimensiones de primeros dos ejes, el último
                      # debería ser 3 (velocidades y coordenadas con 3
                      # dimensiones)
                variable_redimensionada_rotada \
                    = rotacion.apply(variable_redimensionada)
                dicc_trayectoria['variables'][var]['data'] \
                    = variable_redimensionada_rotada.reshape(
                        dimensiones_originales)

            if modo == 'flip':
                dicc_trayectoria['variables']['ix']['data'], \
                dicc_trayectoria['variables'][f'i{eje_original}'] \
                ['data'] \
                    = dicc_trayectoria['variables'][f'i{eje_original}'] \
                      ['data'], -1*dicc_trayectoria['variables']['ix']['data']
                      # intercambio valores de diccionarios e invierto signo

        #/ Rotación de índices de imágenes periódicas
        if 'ix' in dicc_trayectoria['variables'].keys() and\
        'iy' in dicc_trayectoria['variables'].keys() and\
        'iz' in dicc_trayectoria['variables'].keys():
            try:
                print(f'\tRotando índices de imagen periódica\n')


                if modo == 'dot' or modo == 'euler':
                    #/ Combinación de índices de imágenes periódicas en arreglo
                    #/ para aplicar rotación
                    imagenes_periodicas = np.concatenate((
                        np.expand_dims(dicc_trayectoria['variables']['ix'] \
                            ['data'],axis=-1),
                        np.expand_dims(dicc_trayectoria['variables']['iy'] \
                            ['data'],axis=-1),
                        np.expand_dims(dicc_trayectoria['variables']['iz'] \
                            ['data'],axis=-1)),
                        axis=-1)

                    if modo == 'dot':
                        imagenes_rotadas = np.dot(imagenes_periodicas,
                                                matriz_rotacion)
                        dicc_trayectoria['variables']['ix']['data'] \
                            = imagenes_rotadas[:,:,0]
                        dicc_trayectoria['variables']['iy']['data'] \
                            = imagenes_rotadas[:,:,1]
                        dicc_trayectoria['variables']['iz']['data'] \
                            = imagenes_rotadas[:,:,2]

                    if modo == 'euler':
                        dimensiones_originales = imagenes_periodicas.shape
                        imagenes_periodicas_redimensionadas \
                            = imagenes_periodicas.reshape(
                                -1,dimensiones_originales[-1])
                        imagenes_periodicas_redimensionadas_rotadas \
                            = rotacion.apply(imagenes_periodicas_redimensionadas)
                        imagenes_periodicas_rotadas \
                            = imagenes_periodicas_redimensionadas_rotadas.reshape(
                                dimensiones_originales)
                        dicc_trayectoria['variables']['ix']['data'] \
                            = imagenes_periodicas_rotadas[:,:,0]
                        dicc_trayectoria['variables']['iy']['data'] \
                            = imagenes_periodicas_rotadas[:,:,1]
                        dicc_trayectoria['variables']['iz']['data'] \
                            = imagenes_periodicas_rotadas[:,:,2]

                if modo == 'flip':
                    dicc_trayectoria['variables']['ix']['data'], \
                    dicc_trayectoria['variables'][f'i{eje_original}'] \
                    ['data'] \
                        = dicc_trayectoria['variables'][f'i{eje_original}'] \
                          ['data'], -1*dicc_trayectoria['variables']['ix'] \
                          ['data']
                          # intercambio valores de diccionarios e invierto signo

            except:
                print(f'\t\tFalló rotación de índices de imagen periódica\n')
                pass

        else:
            print('\tÍndices de imágenes periódicas no están presentes en ' +
            'diccionario trayectoria\n')

    else:
        print('\tNo se aplica rotación\n')

    return 0


def calcular_imagenes_periodicas(dicc_trayectoria, vectorizado=True):
    """
    Cálculo de la imagen periódica en X, Y, Z de la posición de la partícula.

    Argumentos:
        dicc_trayectoria: diccionario con datos de trayectoria.
    """
    #/ Variables más utilizadas
    posiciones = dicc_trayectoria['variables']['coordinates']['data']
    longitudes_celda = dicc_trayectoria['variables']['cell_lengths']['data']

    #/ Datos para cálculo
    num_cuadros, num_particulas, num_dimensiones = posiciones.shape

    #/ Inicializo imágenes periódicas en 0
    #/ TODO: obtener imágenes periódicas de archivo de configuración
    imagenes_periodicas = \
        np.zeros((num_cuadros,num_particulas,num_dimensiones), dtype=int)

    #/ Modo vectorizado
    if vectorizado:
        #/ Diferencia de posiciones entre cuadros consecutivos
        deltas = np.diff(posiciones, axis=0)

        #/ Expansión de 'longitudes_celda' para poder operar con broadcasting
        longitudes_celda_expandido = longitudes_celda[:-1,np.newaxis,:]
            # Exclusión de último cuadro porque np.diff reduce

        #/ Arreglos booleanos con condiciones para incrementar o decrementar las
        #/ imágenes periódicas
        cruce_hacia_borde_mayor = deltas > ((+1/2) * longitudes_celda_expandido)
        cruce_hacia_borde_menor = deltas < ((-1/2) * longitudes_celda_expandido)

        #/ Arreglo para incrementos o decrementos de imágenes periódicas según
        #/ el cruce de borde
        incrementos = np.zeros_like(deltas, dtype='int32')
        incrementos[cruce_hacia_borde_mayor] = -1
        incrementos[cruce_hacia_borde_menor] = +1

        #/ Suma acumulativa a partir del segundo cuadro
        imagenes_periodicas[1:] += np.cumsum(incrementos, axis=0)

    #/ Modo no vectorizado
    else:
        for t in range(1, num_cuadros):
            for p in range(num_particulas):
                print(f'\tCalculando imagen perdiódica para cuadro {t}, partícula {p}', end='\r')
                for d in range(num_dimensiones):
                    delta = posiciones[t,p,d] - posiciones[t-1,p,d]
                    if delta > (1/2) * longitudes_celda[t,d]:
                        imagenes_periodicas[t,p,d] = \
                            imagenes_periodicas[t-1,p,d] - 1
                    elif delta < (-1/2) * longitudes_celda[t,d]:
                        imagenes_periodicas[t,p,d] = \
                            imagenes_periodicas[t-1,p,d] + 1
                    else:
                        imagenes_periodicas[t,p,d] = \
                            imagenes_periodicas[t-1,p,d]

    #/ Creación de variable imagen periódica en el diccionario de trayectoria
    for dimension in range(num_dimensiones):
        if dimension == 0:
            direccion = 'x'
        elif dimension == 1:
            direccion = 'y'
        elif dimension == 2:
            direccion = 'z'
        else:
            print('Error: número de dimensiones mayor a 3\n')

        dicc_trayectoria['variables'][f'i{direccion}'] = {}
        dicc_trayectoria['variables'][f'i{direccion}']['name'] = \
             f'i{direccion}'
        dicc_trayectoria['variables'][f'i{direccion}']['dtype'] = \
            'int32'
        dicc_trayectoria['variables'][f'i{direccion}']['dimensions'] = \
            ('frame', 'atom')
        dicc_trayectoria['variables'][f'i{direccion}']['data'] \
            = np.ma.masked_array(data=imagenes_periodicas[:,:,dimension],
            fill_value=999999, mask=False, dtype='int32')

    return 0


def desplegar_coordenadas(dicc_trayectoria):
    """
    Despliegue de coordenadas de posición de las partículas de la
    trayectoria. Únicamente posible si el diccionario contiene las
    variables 'ix', 'iy', e 'iz'.

    Argumentos:
        dicc_trayectoria: diccionario con datos de trayectoria.
    """
    print('Desplegando coordenadas de partículas\n')
    #/ Verifico que el diccionario contiene imágenes periódicas
    if 'ix' in dicc_trayectoria['variables']:
        print('\tImágenes periódicas en X encontradas\n')
        if 'iy' in dicc_trayectoria['variables']:
            print('\tImágenes periódicas en Y encontradas\n')
            if 'iz' in dicc_trayectoria['variables']:
                print('\tImágenes periódicas en Z encontradas\n')
            else:
                msj = 'Imágenes periódicas en Z no encontradas'
                raise ValueError(msj)
        else:
            msj = 'Imágenes periódicas en Y no encontradas'
            raise ValueError
    else:
        msj = 'Imágenes periódicas en X no encontradas'
        raise ValueError(msj)

    #/ Despliegue de coordenadas para cada dirección
    for direccion in (('x',0),('y',1),('z',2)):
        dicc_trayectoria['variables']['coordinates']['data'] \
        [:,:,direccion[1]] \
            += (dicc_trayectoria['variables'][f'i{direccion[0]}'] \
               ['data'][:,:] \
               * dicc_trayectoria['variables']['cell_lengths'] \
                 ['data'][0,direccion[1]])

    return 0


#/ ---- OBTENCIÓN DE DATOS Y CÁLCULOS A PARTIR DE TRAYECTORIAS ----------------
def obtener_dimensiones_tubo(
    tipos,
    posiciones,
    origenes_celda,
    longitudes_celda,
    radio_real,
    largo_real,
    radio_nominal,
    largo_nominal,
    eje_tubo='X',
    sigma_referencia=3.5,
    en_borde=True,
    fraccion_borde=0.975,
    fraccion_tubo=0.95,
    epsilon_tubo=1.1,
    id_C=6,
    l_enlace_C=1.42,
    epsilon_delta_l = 1.5,
    tubo_curvo=True):
    """
    Obtención del largo y radios reales del tubo, con opción de considerar su
    curvatura.
    La opción 'en_borde' encuentra los átomos de carbono que se ubicen próximos
    a los contornos de la caja de simulación, a una distancia menor al largo
    nominal.
    Si no se usa esta opción, entonces se toma la posición del átomo de
    carbono más alejado del origen de coordenadas y en torno al largo
    nominal.
    En ambos casos se asume un tubo simétrico con la distancia al
    origen del átomo de carbono como límite superior.

    Obtención a partir de diccionario con datos de trayectoria.
    """

    #/ Diccionarios con radios y largos reales de tubo según radio y largo
    #/ nominal
    dicc_radios = {
        3.0: {
            37.5: {
                'largo': 136.367114,
                'radio': 10.574940
            }
        },
        1.5: {
            37.5: {
                'largo': 135.825377,
                'radio': 5.553073
            }
        }
    }

    #/ Índices para direcciones 'largo', 'base' y 'altura' de celda.
    #/ Si el eje del tubo es paralelo a dirección X, 'base' es Y y altura es Z. 
    #/ Si es paralelo a dirección Z, 'base' es X y altura es Y. 
    #/ Si es paralelo a dirección Y, 'base' es Z y altura es X. 
    if eje_tubo == 'X':
        ind_largo = 0
        ind_base = 1
        ind_altura = 2
    elif eje_tubo == 'Y':
        ind_largo = 1
        ind_base = 2
        ind_altura = 0
    elif eje_tubo == 'Z':
        ind_largo = 2
        ind_base = 0
        ind_altura = 1
    else:
        raise ValueError('Coordenada de eje de tubo mal especificada')

    if largo_real == None or radio_real == None:
        try:
            print('Intentando obtener coordenadas de carbono de trayectoria\n')
            particulas_carbono = tipos['data'][0,:] == id_C
            if np.any(particulas_carbono):
                pass
            else:
                raise(ValueError)
            coords_carbono = posiciones['data'][0,particulas_carbono]
            base_borde_inf = origenes_celda['data'][0,ind_base]
            altura_borde_inf = origenes_celda['data'][0,ind_altura]
            base_borde_sup = base_borde_inf \
                + longitudes_celda['data'][0,ind_base]
            altura_borde_sup = altura_borde_inf \
                + longitudes_celda['data'][0,ind_altura]
        except:
            print('No se pudo obtener coordenadas de carbono de tubo de ' +
                'trayectoria\n')

    if largo_real == None:
        try:
            largo_borde = largo_nominal*sigma_referencia*epsilon_tubo
            carbono_borde = np.abs(coords_carbono[:,ind_largo]) < largo_borde
            if en_borde == True:
                carbono_borde = np.logical_and(
                    carbono_borde,
                    np.logical_or(
                        coords_carbono[:,ind_base]
                        > (base_borde_sup*fraccion_borde),
                        np.logical_or(
                            coords_carbono[:,ind_base]
                            < (base_borde_inf*fraccion_borde),
                            np.logical_or(
                                coords_carbono[:,ind_altura]
                                > (altura_borde_sup*fraccion_borde),
                                coords_carbono[:,ind_altura]
                                < (altura_borde_inf*fraccion_borde)
                            )
                        )
                    )
                )
                largo_real = 2 * np.abs(
                    coords_carbono[carbono_borde,ind_largo]).mean()
            else:
                largo_real = 2 * np.abs(coords_carbono[:,ind_largo]).max()

        except:
            print('No se pudo determinar longitud real de tubo de ' +
                'trayectoria\n')
            largo_real \
                = dicc_radios[radio_nominal][largo_nominal]['largo']

    if radio_real == None:
        largo_analisis = largo_real * fraccion_tubo
        try:
            #/ Tomo porción de tubo descontando paredes
            coords_sinparedes = np.abs(coords_carbono[:,ind_largo]) \
                <= largo_analisis/2

            #/ Considero curvatura del tubo
            if tubo_curvo:
                #/ Analizo la posición del centro y el radio del tubo en segmentos
                #/ de anillos
                delta_l = l_enlace_C * epsilon_delta_l
                bins_l_pos = np.arange(0,largo_analisis/2,delta_l)
                bins_l = np.concatenate((-bins_l_pos[:1:-1],bins_l_pos[:]))
                #/ Ubico en los bins las partículas de carbono según coordenada
                #/ longitudinal
                coords_tubo = coords_carbono[coords_sinparedes,:]
                inds_bins_l = np.digitize(coords_tubo[:,ind_largo],bins_l)
                #/ Arrays para guardar posiciones de los centros de anillos y
                #/ sus radios
                posicion_centro_anillos = np.zeros((bins_l.size-1,2))
                radio_anillos = np.zeros(bins_l.size-1)
                #/ Recorro los índices donde fueron ubicadas las partículas.
                #/ REVISAR EL MÉTODO SI HUBIERA BINS VACÍOS, NO COINCIDIRÍAN
                #/ 'I' CON 'IND' EN LA ENUMERACIÓN.
                for i,ind in enumerate(np.unique(inds_bins_l)):
                    mascara_anillo = np.where(inds_bins_l == ind)
                    coords_anillo = coords_tubo[mascara_anillo]
                    posicion_centro_anillos[i] = \
                        coords_anillo[:,[ind_base,ind_altura]].mean(axis=0)
                    radio_anillos[i] = np.linalg.norm(
                        (coords_anillo[:,[ind_base,ind_altura]]
                        - posicion_centro_anillos[i]), axis=1).mean()
                radio_real = radio_anillos.mean()

            #/ No considero curvatura del tubo
            else:
                radio_real = np.linalg.norm(
                    coords_carbono[coords_sinparedes][:,[ind_base,ind_altura]],
                    axis=1).mean()

        except:
            print('No se pudo determinar radio medio real de tubo de ' +
                'trayectoria\n')
            radio_real \
                = dicc_radios[radio_nominal][largo_nominal]['radio']

    print(f'\tLongitud real de tubo: {largo_real:.6f} Å\n')
    print(f'\tRadio medio de tubo: {radio_real:.6f} Å\n')

    salida = [radio_real, largo_real]

    if tubo_curvo:
        for i in (radio_anillos, posicion_centro_anillos, bins_l):
            salida.append(i)

    return salida


def calcular_limites_tubo(
    limites_tubo,
    radio_tubo,
    fraccion_region=0.9,
    fraccion_radio=1.0):
    """
    Cálculo de límites de región de tubo para hacer análisis.
    """
    #/ Arreglo para delimitar región con fracción requerida
    largo_tubo = np.diff(limites_tubo)
    centro_tubo = np.mean(limites_tubo)
    limites_tubo_fraccion = \
        centro_tubo + np.kron(largo_tubo * 0.5 * fraccion_region, [-1, 1])
    largo_tubo_fraccion = np.diff(limites_tubo_fraccion)
    radio_tubo_fraccion = radio_tubo * fraccion_radio

    return largo_tubo, centro_tubo, limites_tubo_fraccion, \
        largo_tubo_fraccion, radio_tubo_fraccion


def calcular_msd(
    tiempo,
    posiciones,
    tipo_particulas,
    id_tipo,
    limites_region,
    fraccion_region=(1.0,1.0,1.0),
    fraccion_tiempo=0.25,
    vectorizado=True):
    """
    Cálculo de desplazamiento cuadrático medio en fracción de región de
    interés.
    Calcula desplazamiento en direcciones x, y, z, y desplazamiento total.
    Opción para realizar cálculo vectorizado.
    """
    #/ Número de intervalos para cálculo
    num_intervalos = int(tiempo['data'].size * fraccion_tiempo) - 1
    print('\t\tPara intervalos con delta_t {} a {} {}\n'.format(tiempo['data'][1],tiempo['data'][num_intervalos],tiempo['units']))

    #/ Arreglo para delimitar región con fracción requerida
    largos_region = np.diff(limites_region, axis=1)
    centros_region = np.mean(limites_region, axis=1)
    limites_region_fraccion = centros_region[:,np.newaxis] + np.kron(largos_region*0.5*np.array(fraccion_region)[:,np.newaxis], [-1,1])

    #/ Arreglos para guardar cálculos
    msd_x = np.zeros(num_intervalos)
    msd_y = np.zeros(num_intervalos)
    msd_z = np.zeros(num_intervalos)
    msd_t = np.zeros(num_intervalos)
    contador = np.zeros(num_intervalos)

    #/ Índices de partículas del tipo deseado
    idx_particulas = np.argwhere(tipo_particulas['data'][0] == id_tipo).flatten()

    #/ Cálculo vectorizado
    if vectorizado:
        #/ Filtrar posiciones y tipos para las partículas de interés
        posiciones_particula = posiciones['data'][:,idx_particulas,:]

        #/ Verificar si la partícula está en la región en cualquier instante de
        #/ tiempo
        en_region = np.all(
            (posiciones_particula >= limites_region_fraccion[:,0]) &
            (posiciones_particula <= limites_region_fraccion[:,1]), axis=2)

        #/ Iteración para calcular desplazamientos con todos los orígenes
        #/ posibles para el mismo intervalo temporal
        for t in range(1, num_intervalos+1):
            #/ Calcular el desplazamiento para todos los orígenes temporales
            desplazamientos = \
                posiciones_particula[t:] - posiciones_particula[:-t]

            #/ Filtrar cuando las partículas están en la región
            mask_en_region = en_region[:-t] & en_region[t:]

            #/ Desplazamientos en cada dirección
            desplazamiento_x = desplazamientos[:,mask_en_region,0]
            desplazamiento_y = desplazamientos[:,mask_en_region,1]
            desplazamiento_z = desplazamientos[:,mask_en_region,2]

            #/ Calcular MSD acumulado de todas las partículas
            msd_x[t-1] = np.sum(desplazamiento_x**2)
            msd_y[t-1] = np.sum(desplazamiento_y**2)
            msd_z[t-1] = np.sum(desplazamiento_z**2)
            msd_t[t-1] = np.sum(
                desplazamiento_x**2+desplazamiento_y**2+desplazamiento_z**2)

            #/ Contar el número de desplazamientos válidos
            contador[t-1] = np.sum(mask_en_region)

    #/ Cálculo no vectorizado
    else:
        #/ Evaluación de msd para tipo de partículas
        contador_idx = 0
        for idx in idx_particulas:
            posiciones_particula = posiciones['data'][:,idx,:]
            en_region = np.all(
                (posiciones_particula >= limites_region_fraccion[:,0]) &
                (posiciones_particula <= limites_region_fraccion[:,1]), axis=1)

            #/ Si la partícula no está en la región, salto a la siguiente
            if not np.any(en_region):
                continue

            #/ Si la partícula está en región, calculo msd para fracción de
            #/ tiempo
            contador_idx += 1
            print(f'\t\t\tPara partícula {contador_idx}', end='\r')

            #/ Iteración para calcular desplazamientos con todos los orígenes
            #/ posibles para el mismo intervalo temporal
            for t in range(1, num_intervalos+1):
                #/ Calcular desplazamiento para todos los orígenes temporales
                desplazamientos = \
                    posiciones_particula[t:] - posiciones_particula[:-t]

                #/ Filtrar cuando las partículas están en región
                mask_en_region = en_region[:-t] & en_region[t:]

                #/ Desplazamientos en cada dirección
                desplazamiento_x = desplazamientos[:,mask_en_region,0]
                desplazamiento_y = desplazamientos[:,mask_en_region,1]
                desplazamiento_z = desplazamientos[:,mask_en_region,2]

                #/ Calcular MSD acumulado de la partícula
                msd_x[t-1] += np.sum(desplazamiento_x**2)
                msd_y[t-1] += np.sum(desplazamiento_y**2)
                msd_z[t-1] += np.sum(desplazamiento_z**2)
                msd_t[t-1] += np.sum(
                    desplazamiento_x**2+desplazamiento_y**2+desplazamiento_z**2)

                #/ Contar el número de desplazamientos válidos
                contador[t-1] += np.sum(mask_en_region)

    #/ Promedio el msd x, y, z, xyz con el total de partículas en todos los intervalos
    msd_x /= np.maximum(contador, 1)
    msd_y /= np.maximum(contador, 1)
    msd_z /= np.maximum(contador, 1)
    msd_t /= np.maximum(contador, 1)

    print('\n')

    #/ Intervalos temporales
    intervalos = tiempo['data'][1:num_intervalos+1]

    return intervalos, msd_x, msd_y, msd_z, msd_t, contador, limites_region_fraccion



def calcular_msd_con_posicion_inicial_vectorizado(
    tiempo,
    posiciones,
    tipo_particulas,
    id_tipo,
    limites_region,
    fraccion_region=(1.0,1.0,1.0),
    fraccion_tiempo=0.25):
    """
    Cálculo vectorizado de desplazamiento cuadrático medio en fracción de región de interés.
    Calcula desplazamiento en direcciones x, y, z, y desplazamiento total.
    Utiliza solamente las posiciones en t=0 como referencia.
    """
    #/ Número de intervalos para cálculo
    num_intervalos = int(tiempo['data'].size * fraccion_tiempo) - 1
    print('\t\tPara intervalos con delta_t {} a {} {}\n'.format(tiempo['data'][1],tiempo['data'][num_intervalos],tiempo['units']))

    #/ Arreglo para delimitar región con fracción requerida
    largos_region = np.diff(limites_region, axis=1)
    centros_region = np.mean(limites_region, axis=1)
    limites_region_fraccion = centros_region[:,np.newaxis] + np.kron(largos_region*0.5*np.array(fraccion_region)[:,np.newaxis], [-1,1])

    #/ Arreglos para guardar cálculos
    msd_x = np.zeros(num_intervalos)
    msd_y = np.zeros(num_intervalos)
    msd_z = np.zeros(num_intervalos)
    msd_t = np.zeros(num_intervalos)
    contador = np.ones(num_intervalos)

    #/ Índices de partículas del tipo deseado
    idx_particulas = np.argwhere(tipo_particulas['data'][0] == id_tipo).flatten()

    #/ Filtrar posiciones y tipos para las partículas de interés
    posiciones_interes = posiciones['data'][:,idx_particulas,:]

    #/ Obtención de posiciones iniciales
    posiciones_iniciales = posiciones_interes[0,:,:]

    # Verificar si la partícula está en la región en instante inicial
    en_region = np.all(
        (posiciones_iniciales >= limites_region_fraccion[:,0]) &
        (posiciones_iniciales <= limites_region_fraccion[:,1]), axis=1)

    #/ Calcular desplazamientos respecto a la posición inicial
    desplazamientos = posiciones_interes[1:] - posiciones_iniciales

    #/ Filtrar desplazamientos para partículas de interés en región e instante inicial
    desplazamiento_x = desplazamientos[:,en_region,0]
    desplazamiento_y = desplazamientos[:,en_region,1]
    desplazamiento_z = desplazamientos[:,en_region,2]

    # Calcular MSD
    msd_x = np.mean(desplazamiento_x**2, axis=1)
    msd_y = np.mean(desplazamiento_y**2, axis=1)
    msd_z = np.mean(desplazamiento_z**2, axis=1)
    msd_t = np.mean(desplazamiento_x**2+desplazamiento_y**2+desplazamiento_z**2, axis=1)

    print('\n')

    #/ Intervalos temporales
    intervalos = tiempo['data'][1:num_intervalos+1]

    return intervalos, msd_x, msd_y, msd_z, msd_t, contador, limites_region_fraccion


def calcular_flujo(
    posiciones,
    velocidades,
    tipo_particulas,
    id_tipo,
    masa_tipo,
    limites_tubo,
    limite_radio,
    centros_segmentos,
    bins_l,
    fraccion_region=0.9,
    fraccion_radio=1.0,
    eje_tubo='Z'):
    """
    Cálculo de flujo másico y numérico en región limitada de tubo.
    La región de interés se asume como un tubo de eje X, con límites xmin, xmax
    y r.
    Se computa el flujo sobre una fracción de la longitud X del tubo.
    """
    #/ Límites de fracción de tubo requerida
    largo_tubo, centro_tubo, limites_tubo_fraccion, largo_tubo_fraccion, \
    radio_tubo_fraccion = \
    calcular_limites_tubo(
        limites_tubo,
        limite_radio,
        fraccion_region=fraccion_region,
        fraccion_radio=fraccion_radio)

    #/ Determino índice de coordenada del eje del tubo
    if eje_tubo == 'X':
        ind_largo = 0
        ind_base = 1
        ind_altura = 2
    elif eje_tubo == 'Y':
        ind_largo = 1
        ind_base = 2
        ind_altura = 0
    elif eje_tubo == 'Z':
        ind_largo = 2
        ind_base = 0
        ind_altura = 1
    else:
        raise ValueError('Coordenada de eje de tubo mal especificada')

    #/ Elijo tipo de partículas, sus coordenadas (xyz) y velocidades (componente longitudinal)
    idx_particulas = tipo_particulas['data'][0] == id_tipo
    posiciones_particulas = posiciones['data'][:,idx_particulas,:]
    velocidades_particulas = velocidades['data'][:,idx_particulas,ind_largo]

    #/ CORRECCIÓN DE POSICIONES POR CURVATURA DE TUBO
    #/ ----------------------------------------------

    #/ Obtener índices de bins longitudinales (base-1)
    indices_bins = np.digitize(posiciones_particulas[:,:,ind_largo], bins_l)

    #/ Pliego a índice 1 y len(bins_l)-1 aquellos que sobrepasan (0 o len(bins_l))
    indices_bins = np.where(indices_bins==0, 1, indices_bins)
    indices_bins = np.where(indices_bins==len(bins_l), len(bins_l)-1, indices_bins)
    
    #/ Obtener centros de referencia para interpolación
    centros_actuales = centros_segmentos[indices_bins - 1]  # Convertir a base-0
    
    #/ Calcular centro longitudinal de cada bin actual
    centro_actual_largo = (bins_l[indices_bins - 1] + bins_l[indices_bins]) / 2.0
    
    #/ Determinar dirección de interpolación
    interpolacion_anterior = posiciones_particulas[:,:,ind_largo] < centro_actual_largo
    
    #/ Preparar pesos y centros adyacentes
    peso = np.zeros(posiciones_particulas[:,:,ind_largo].shape)
    centros_adyacentes = np.zeros_like(centros_actuales)
    
    #/ Caso 1: Interpolar con el segmento ANTERIOR (si existe)
    mascara_anterior = interpolacion_anterior & (indices_bins > 1)
    if np.any(mascara_anterior):
        idx_anterior = indices_bins[mascara_anterior] - 2  # -2 por base-1 de digitize
        centros_adyacentes[mascara_anterior] = centros_segmentos[idx_anterior]
        pos_bin_inferior = bins_l[idx_anterior]
        pos_bin_superior = bins_l[idx_anterior + 1]
        peso[mascara_anterior] = \
            (posiciones_particulas[mascara_anterior,ind_largo] - pos_bin_inferior) \
            / (pos_bin_superior - pos_bin_inferior)
    
    #/ Caso 2: Interpolar con el segmento SIGUIENTE (si existe)
    mascara_siguiente = np.logical_not(interpolacion_anterior) & (indices_bins < len(bins_l) - 1)
    if np.any(mascara_siguiente):
        idx_siguiente = indices_bins[mascara_siguiente]  # Ya en base-0 (centros_segmentos)
        centros_adyacentes[mascara_siguiente] = centros_segmentos[idx_siguiente]
        pos_bin_inferior = bins_l[idx_siguiente - 1]
        pos_bin_superior = bins_l[idx_siguiente]
        peso[mascara_siguiente] = \
            (posiciones_particulas[mascara_siguiente,ind_largo] - pos_bin_inferior) \
            / (pos_bin_superior - pos_bin_inferior)
    
    #/ Interpolación lineal de centros
    centros_interpolados = \
        (1 - peso[:,:,np.newaxis]) * centros_adyacentes \
        + peso [:,:,np.newaxis]* centros_actuales
    
    #/ Corregir posiciones
    posiciones_corregidas = posiciones_particulas.copy()
    posiciones_corregidas[:,:,[ind_base,ind_altura]] -= centros_interpolados[:,:,[ind_base,ind_altura]]

    #/ CÁLCULO DE FLUJO
    #/ ----------------

    #/ Obtengo partículas y sus velocidades (componente x) dentro del tubo
    #/ Enmascaro con NaN las velocidades de las partículas que no están en la
    #/ región, y uso una función que no las tiene en cuenta en la suma.
    #/ También podría enmascararlas con 0, pues no suman, pero NaN es más claro.
    en_largo_tubo = \
        (posiciones_corregidas[:,:,ind_largo] >= limites_tubo_fraccion[0]) & \
        (posiciones_corregidas[:,:,ind_largo] <= limites_tubo_fraccion[1])
    en_radio_tubo = \
        np.linalg.norm(posiciones_corregidas[:,:,[ind_base,ind_altura]], axis=2) <= radio_tubo_fraccion
    en_tubo = en_radio_tubo & en_largo_tubo
    velocidades_en_tubo = np.where(en_tubo, velocidades_particulas, np.nan)

    #/ Cálculo del flujo instantáneo de partículas
    conteo = np.count_nonzero(en_tubo, axis=1)
    volumen_tubo_fraccion = radio_tubo_fraccion**2 * np.pi * largo_tubo_fraccion
    flujo_numerico_instantaneo = \
        np.nansum(velocidades_en_tubo, axis=1) / volumen_tubo_fraccion
    flujo_masico_instantaneo = flujo_numerico_instantaneo * masa_tipo

    return conteo, flujo_numerico_instantaneo, flujo_masico_instantaneo, radio_tubo_fraccion, limites_tubo_fraccion


def contar_particulas_pervaporadas(
    posiciones,
    tipo_particulas,
    id_tipo,
    largo_tubo,
    omitir_primer_cuadro=True,
    eje_tubo='Z'):
    """
    Conteo de partículas pervaporadas en reservorio vacío de sistema.
    Toma como región de interés aquellas con coordenada X superior a extremo de
    tubo.
    """
    #/ Cantidad de cuadros en la trayectoria
    if omitir_primer_cuadro:
        ini_cuadro = 1
    else:
        ini_cuadro = 0
    num_cuadros = posiciones['data'][ini_cuadro:,:,:].shape[0]

    #/ Determino índice de coordenada del eje del tubo
    if eje_tubo == 'X':
        ind_largo = 0
        ind_base = 1
        ind_altura = 2
    elif eje_tubo == 'Y':
        ind_largo = 1
        ind_base = 2
        ind_altura = 0
    elif eje_tubo == 'Z':
        ind_largo = 2
        ind_base = 0
        ind_altura = 1
    else:
        raise ValueError('Coordenada de eje de tubo mal especificada')

    #/ Elijo tipo de partículas y su coordenada longitudinal
    idx_particulas = tipo_particulas['data'][ini_cuadro] == id_tipo
    posiciones_particulas = posiciones['data'][ini_cuadro:,idx_particulas,ind_largo]

    #/ Conteo por cuadro de partículas pervaporadas
    particulas_pervaporadas = posiciones_particulas > largo_tubo/2
    conteo = np.count_nonzero(particulas_pervaporadas, axis=1)

    return conteo


def calcular_densidad_numerica_radial_tubo_curvo_interpolado_por_cuadro(
    posiciones,
    tipo_particulas,
    id_tipo,
    limites_tubo,
    limite_radio,
    centros_segmentos,
    bins_l,
    fraccion_region=0.9,
    fraccion_radio=1.0,
    fraccion_bin_r=0.25,
    sigma_ref=3.5,
    volumenes_iguales=False,
    normalizar=False,
    eje_tubo='Z'):
    """
    Cálculo de densidad numérica en coordenadas radiales en un tubo curvo.
    - Interpola entre el segmento actual y el adyacente más cercano (anterior o
      siguiente).
    - Índices 0 o len(bins_l) indican partículas fuera del tubo (se descartan).
    """
    #/ Límites de fracción de tubo requerida
    largo_tubo, centro_tubo, limites_tubo_fraccion, largo_tubo_fraccion, \
    radio_tubo_fraccion = \
    calcular_limites_tubo(
        limites_tubo,
        limite_radio,
        fraccion_region=fraccion_region,
        fraccion_radio=fraccion_radio)

    #/ Seleccionar partículas del tipo deseado
    idx_particulas = tipo_particulas['data'][0] == id_tipo
    posiciones_particulas = posiciones['data'][:,idx_particulas,:]

    #/ Determino índice de coordenada del eje del tubo
    if eje_tubo == 'X':
        ind_largo = 0
        ind_base = 1
        ind_altura = 2
    elif eje_tubo == 'Y':
        ind_largo = 1
        ind_base = 2
        ind_altura = 0
    elif eje_tubo == 'Z':
        ind_largo = 2
        ind_base = 0
        ind_altura = 1
    else:
        raise ValueError('Coordenada de eje de tubo mal especificada')

    #/ CORRECCIÓN DE POSICIONES POR CURVATURA DE TUBO
    #/ ----------------------------------------------

    #/ Obtener índices de bins longitudinales (base-1)
    indices_bins = np.digitize(posiciones_particulas[:,:,ind_largo], bins_l)

    #/ Pliego a índice 1 y len(bins_l)-1 aquellos que sobrepasan (0 o len(bins_l))
    indices_bins = np.where(indices_bins==0, 1, indices_bins)
    indices_bins = np.where(indices_bins==len(bins_l), len(bins_l)-1, indices_bins)
    
    #/ Obtener centros de referencia para interpolación
    centros_actuales = centros_segmentos[indices_bins - 1]  # Convertir a base-0
    
    #/ Calcular centro longitudinal de cada bin actual
    centro_actual_largo = (bins_l[indices_bins - 1] + bins_l[indices_bins]) / 2.0
    
    #/ Determinar dirección de interpolación
    interpolacion_anterior = posiciones_particulas[:,:,ind_largo] < centro_actual_largo
    
    #/ Preparar pesos y centros adyacentes
    peso = np.zeros(posiciones_particulas[:,:,ind_largo].shape)
    centros_adyacentes = np.zeros_like(centros_actuales)
    
    #/ Caso 1: Interpolar con el segmento ANTERIOR (si existe)
    mascara_anterior = interpolacion_anterior & (indices_bins > 1)
    if np.any(mascara_anterior):
        idx_anterior = indices_bins[mascara_anterior] - 2  # -2 por base-1 de digitize
        centros_adyacentes[mascara_anterior] = centros_segmentos[idx_anterior]
        pos_bin_inferior = bins_l[idx_anterior]
        pos_bin_superior = bins_l[idx_anterior + 1]
        peso[mascara_anterior] = \
            (posiciones_particulas[mascara_anterior,ind_largo] - pos_bin_inferior) \
            / (pos_bin_superior - pos_bin_inferior)
    
    #/ Caso 2: Interpolar con el segmento SIGUIENTE (si existe)
    mascara_siguiente = np.logical_not(interpolacion_anterior) & (indices_bins < len(bins_l) - 1)
    if np.any(mascara_siguiente):
        idx_siguiente = indices_bins[mascara_siguiente]  # Ya en base-0 (centros_segmentos)
        centros_adyacentes[mascara_siguiente] = centros_segmentos[idx_siguiente]
        pos_bin_inferior = bins_l[idx_siguiente - 1]
        pos_bin_superior = bins_l[idx_siguiente]
        peso[mascara_siguiente] = \
            (posiciones_particulas[mascara_siguiente,ind_largo] - pos_bin_inferior) \
            / (pos_bin_superior - pos_bin_inferior)
    
    #/ Interpolación lineal de centros
    centros_interpolados = \
        (1 - peso[:,:,np.newaxis]) * centros_adyacentes \
        + peso [:,:,np.newaxis]* centros_actuales
    
    #/ Corregir posiciones
    posiciones_corregidas = posiciones_particulas.copy()
    posiciones_corregidas[:,:,[ind_base,ind_altura]] \
        -= centros_interpolados[:,:,[ind_base,ind_altura]]

    #/ CÁLCULO DE DENSIDAD RADIAL
    #/ --------------------------
    
    #/ Calcular la posición radial corregida
    posiciones_radiales = np.linalg.norm(posiciones_corregidas[:,:,[ind_base,ind_altura]], axis=2)

    #/ Filtrar partículas dentro del tubo en longitud
    en_largo_tubo = \
        (posiciones_corregidas[:,:,ind_largo] >= limites_tubo_fraccion[0]) & \
        (posiciones_corregidas[:,:,ind_largo] <= limites_tubo_fraccion[1])
    
    #/ Definir bins radiales
    ancho_bin_r = fraccion_bin_r * sigma_ref # delta_r (mínimo)
    if volumenes_iguales: # delta_r variable, volúmenes fijos
        #/ Despejo el número de bins de la progresión aritmética en r
        #/ r_i^2 = i/N * R^2 --> r_i = R * sqrt(i/N)
        #/ r_(N-1) = R * sqrt((N-1) / N)
        #/ Si r_N = R --> delta-r_N = R * (1 - sqrt((N-1) / N))
        numero_bins_r = int(
            np.round(1 / (1 - (1 - ancho_bin_r / radio_tubo_fraccion)**2), 0))
        bordes_bins_r = np.sqrt(
            np.linspace(0, radio_tubo_fraccion**2, numero_bins_r+1))
    else: # 17/07/25: delta_r fijo (no ajusta a radio de tubo), volúmenes variables
        numero_bins_r = np.ceil(radio_tubo_fraccion / ancho_bin_r).astype('int')
        bordes_bins_r = np.arange(0, ancho_bin_r*(numero_bins_r+1), ancho_bin_r)
    ## else: # delta_r fijo (ajustable a radio de tubo), volúmenes variables
    ##     numero_bins_r = \
    ##         int(round((radio_tubo_fraccion - 0.0) / ancho_bin_r, 0))
    ##     bordes_bins_r = np.linspace(0, radio_tubo_fraccion, numero_bins_r+1)
    print(f'Ancho bin r: {ancho_bin_r}')
    print(f'Numero bins r: {numero_bins_r}')
    print(f'Bordes bins r: {bordes_bins_r}')

    #/ Número de cuadros de trayectoria
    n_cuadros = posiciones['data'].shape[0]
    
    #/ Construcción de histograma para cada cuadro.
    conteos = np.zeros((n_cuadros, numero_bins_r+1))

    idx_posiciones_radiales = np.digitize(posiciones_radiales, bordes_bins_r)
    idx_cuadros = np.arange(n_cuadros).reshape((n_cuadros,1)) * np.ones_like(idx_posiciones_radiales)
    idx_posiciones_radiales_en_tubo = np.where(
        en_largo_tubo, idx_posiciones_radiales, numero_bins_r+1) # asigno bin
            # exterior a partículas fuera del tubo. Los conteos en este bin son
            # luego borrados

    np.add.at(conteos, (idx_cuadros, idx_posiciones_radiales_en_tubo-1), 1)
    conteos = np.delete(conteos, -1, axis=1)

    histograma_r = np.sum(conteos, axis=0)
    histograma_r_mean = np.mean(conteos, axis=0)
    histograma_r_std = np.std(conteos, axis=0)

    #/ Calcular densidad numérica considerando volúmenes de láminas cilíndricas
    ## radio_central_bins = (bordes_bins_r[:-1] + bordes_bins_r[1:]) / 2
    volumenes_bins = largo_tubo_fraccion * \
        (bordes_bins_r[1:]**2 - bordes_bins_r[:-1]**2) * np.pi
    densidad_numerica_r_mean = histograma_r_mean / volumenes_bins
    densidad_numerica_r_std = histograma_r_std / volumenes_bins
    
    return histograma_r, densidad_numerica_r_mean, densidad_numerica_r_std, bordes_bins_r, limites_tubo_fraccion


def calcular_densidad_numerica_longitudinal(
    posiciones,
    tipo_particulas,
    id_tipo,
    limites_region,
    numero_bins=32,
    fraccion_ancho_bin=0.05,
    sigma_ref=3.5,
    ancho_bin_desde_fraccion=False,
    normalizar=False):
    """
    Cálculo de densidad numérica en coordenada longitudinal (paralela al tubo)
    en región requerida.
    Ancho de bins como fracción del sigma de referencia [Å], o de valor por
    defecto.
    """
    #/ Cantidad de cuadros en la trayectoria
    num_cuadros = posiciones['data'].shape[0]

    #/ Arreglo para delimitar región
    largos_region = np.diff(limites_region, axis=1)
    centros_region = np.mean(limites_region, axis=1)

    #/ Elijo tipo de partículas y su coordenadas longitudinal
    idx_particulas = tipo_particulas['data'][0] == id_tipo
    posiciones_particulas = posiciones['data'][:,idx_particulas,:]

    #/ Obtengo partículas dentro de la región ¿NECESARIO?
    ## en_region = \
    ##     (posiciones_particulas[:,0] >= limites_region[0,0]) \
    ##     & (posiciones_particulas[:,:,0] <= limites_region[0,1])

    #/ Obtener posición longitudinal de las partículas
    ## posiciones_longitudinales = posiciones_particulas[en_region,0]
    posiciones_longitudinales = posiciones_particulas[:,:,0]

    #/ Cantidad de bins en función de longitud de región y fracción de sigma
    #/ referencia, o ancho de bins desde número de bins prescrito.
    limites_bins = (limites_region[0,0], limites_region[0,1]) # sólo coordenada longitudinal
    if ancho_bin_desde_fraccion:
        ancho_bins = fraccion_ancho_bin * sigma_ref
        numero_bins = int(round((limites_bins[1] - limites_bins[0]) / ancho_bins,0))
    else:
         ancho_bins = (limites_bins[1] - limites_bins[0]) / numero_bins
    print(f'\t\tNúmero de bins longitudinales: {numero_bins}\n')

    #/ Generación del histograma 1D longitudinal
    histograma_longitudinal, bordes_bins_l = np.histogram(posiciones_longitudinales, bins=numero_bins, range=limites_bins, density=normalizar)

    #/ Calcular densidad numérica por volumen
    coordenada_central_bins = (bordes_bins_l[:-1] + bordes_bins_l[1:]) / 2
    volumenes_bins = (bordes_bins_l[1:] - bordes_bins_l[:-1]) * largos_region[1] * largos_region[2]
    densidad_numerica_longitudinal = histograma_longitudinal / volumenes_bins
    densidad_numerica_longitudinal_promedio = densidad_numerica_longitudinal / num_cuadros

    return densidad_numerica_longitudinal_promedio, bordes_bins_l


def calcular_densidad_numerica_longitudinal_por_cuadro(posiciones,
    tipo_particulas, id_tipo, limites_region, numero_bins=32,
    fraccion_ancho_bin=0.05, sigma_ref=3.5, ancho_bin_desde_fraccion=False,
    normalizar=False):
    """
    Cálculo de densidad numérica en coordenada longitudinal (paralela al tubo)
    en región requerida.
    Ancho de bins como fracción del sigma de referencia [Å], o de valor por
    defecto.
    """
    #/ Cantidad de cuadros en la trayectoria
    num_cuadros = posiciones['data'].shape[0]

    #/ Arreglo para delimitar región
    largos_region = np.diff(limites_region, axis=1)
    centros_region = np.mean(limites_region, axis=1)

    #/ Elijo tipo de partículas y su coordenadas longitudinal
    idx_particulas = tipo_particulas['data'][0] == id_tipo
    posiciones_particulas = posiciones['data'][:,idx_particulas,:]

    #/ Obtengo partículas dentro de la región ¿NECESARIO?
    ## en_region = (posiciones_particulas[:,0] >= limites_region[0,0]) & (posiciones_particulas[:,:,0] <= limites_region[0,1])

    #/ Obtener posición longitudinal de las partículas
    ## posiciones_longitudinales = posiciones_particulas[en_region,0]
    posiciones_longitudinales = posiciones_particulas[:,:,0]

    #/ Cantidad de bins en función de longitud de región y fracción de sigma referencia,
    #/ o ancho de bins desde número de bins prescrito
    limites_bins = (limites_region[0,0], limites_region[0,1]) # sólo coordenada longitudinal
    if ancho_bin_desde_fraccion:
        ancho_bins = fraccion_ancho_bin * sigma_ref
        numero_bins = int(round((limites_bins[1] - limites_bins[0]) / ancho_bins,0))
    else:
         ancho_bins = (limites_bins[1] - limites_bins[0]) / numero_bins
    print(f'\t\tNúmero de bins longitudinales: {numero_bins}\n')

    #/ Construcción de histograma para cada cuadro
    bordes_bins_l = np.linspace(limites_bins[0], limites_bins[1], numero_bins+1)
    conteos = np.zeros((num_cuadros, bordes_bins_l.size))
    indices_posiciones = np.digitize(posiciones_longitudinales, bordes_bins_l)
    print(f'Forma indices posiciones {indices_posiciones.shape}')
    indices_cuadros = np.arange(num_cuadros)[:,np.newaxis] * np.ones_like(indices_posiciones)
    print(f'Forma indices cuadros {indices_cuadros.shape}')
    np.add.at(conteos, (indices_cuadros,indices_posiciones-1), 1)
    histograma_longitudinal = np.delete(conteos,-1,axis=1) # Elimino el último bin donde caen las partículas que no se ubican en alguno de los otros bins

    #/ Calcular densidad numérica por volumen
    coordenada_central_bins = (bordes_bins_l[:-1] + bordes_bins_l[1:]) / 2
    volumenes_bins = (bordes_bins_l[1:] - bordes_bins_l[:-1]) * largos_region[1] * largos_region[2]
    densidad_numerica_longitudinal = histograma_longitudinal / volumenes_bins

    ## #/ Suma del histograma para obtener conteo por cuadro
    ## histograma_longitudinal = np.sum(histograma_longitudinal,axis=1)

    return histograma_longitudinal, densidad_numerica_longitudinal, bordes_bins_l


def calcular_densidad_numerica_longitudinal_promedio(
    posiciones,
    tipo_particulas,
    id_tipo,
    limites_region,
    numero_bins=32,
    fraccion_ancho_bin=0.05,
    sigma_ref=3.5,
    ancho_bin_desde_fraccion=False,
    normalizar=False):
    """
    Cálculo de densidad numérica en coordenada longitudinal (paralela al tubo)
    en región requerida.
    Ancho de bins como fracción del sigma de referencia [Å], o de valor por
    defecto.
    """
    #/ Cantidad de cuadros en la trayectoria
    num_cuadros = posiciones['data'].shape[0]

    #/ Arreglo para delimitar región
    largos_region = np.diff(limites_region, axis=1)
    centros_region = np.mean(limites_region, axis=1)

    #/ Elijo tipo de partículas y su coordenadas longitudinal
    idx_particulas = tipo_particulas['data'][0] == id_tipo
    posiciones_particulas = posiciones['data'][:,idx_particulas,:]

    #/ Obtengo partículas dentro de la región ¿NECESARIO?
    ## en_region = \
    ##     (posiciones_particulas[:,0] >= limites_region[0,0])
    ##     & (posiciones_particulas[:,:,0] <= limites_region[0,1])

    #/ Obtener posición longitudinal de las partículas
    ## posiciones_longitudinales = posiciones_particulas[en_region,0]
    posiciones_longitudinales = posiciones_particulas[:,:,0]

    #/ Cantidad de bins en función de longitud de región y fracción de sigma
    #/ referencia, o ancho de bins desde número de bins prescrito.
    limites_bins = (limites_region[0,0], limites_region[0,1]) # sólo coordenada longitudinal
    if ancho_bin_desde_fraccion:
        ancho_bins = fraccion_ancho_bin * sigma_ref
        numero_bins = int(round((limites_bins[1] - limites_bins[0]) / ancho_bins,0))
    else:
         ancho_bins = (limites_bins[1] - limites_bins[0]) / numero_bins
    print(f'\t\tNúmero de bins longitudinales: {numero_bins}\n')

    #/ Construcción de histograma para cada cuadro
    bordes_bins_l = np.linspace(limites_bins[0], limites_bins[1], numero_bins+1)
    conteos = np.zeros((num_cuadros, bordes_bins_l.size))
    indices_posiciones = np.digitize(posiciones_longitudinales, bordes_bins_l)
    indices_cuadros = np.arange(num_cuadros)[:,np.newaxis] * np.ones_like(indices_posiciones)
    np.add.at(conteos, (indices_cuadros,indices_posiciones-1), 1)
    conteos = np.delete(conteos,-1,axis=1) # Elimino el último bin donde caen las partículas que no se ubican en alguno de los otros bins
    histograma_longitudinal_promedio = np.mean(conteos,axis=0)
    histograma_longitudinal_desvio = np.std(conteos,axis=0)

    #/ Calcular densidad numérica por volumen
    coordenada_central_bins = (bordes_bins_l[:-1] + bordes_bins_l[1:]) / 2
    volumenes_bins = (bordes_bins_l[1:] - bordes_bins_l[:-1]) * largos_region[1] * largos_region[2]
    densidad_numerica_longitudinal_promedio = histograma_longitudinal_promedio / volumenes_bins
    densidad_numerica_longitudinal_desvio = histograma_longitudinal_desvio / volumenes_bins

    return densidad_numerica_longitudinal_promedio, densidad_numerica_longitudinal_desvio, bordes_bins_l


def calcular_perfil_radial_velocidad_tubo_curvo_interpolado_por_cuadro(
    posiciones,
    tipo_particulas,
    velocidades,
    id_tipo,
    limites_tubo,
    limite_radio,
    centros_segmentos,
    bins_l,
    fraccion_region=0.9,
    fraccion_radio=1.0,
    fraccion_bin_r=0.05,
    sigma_ref=3.5,
    volumenes_iguales=False,
    normalizar=False,
    eje_tubo='Z'):
    """
    Cálculo de perfil de velocidad según coordenada radial.
    - Interpola entre el segmento actual y el adyacente más cercano (anterior o
      siguiente).
    - Índices 0 o len(bins_l) indican partículas fuera del tubo (se descartan).
    """
    #/ Límites de fracción de tubo requerida
    largo_tubo, centro_tubo, limites_tubo_fraccion, largo_tubo_fraccion, \
    radio_tubo_fraccion = \
    calcular_limites_tubo(
        limites_tubo,
        limite_radio,
        fraccion_region=fraccion_region,
        fraccion_radio=fraccion_radio)

    #/ Determino índice de coordenada del eje del tubo
    if eje_tubo == 'X':
        ind_largo = 0
        ind_base = 1
        ind_altura = 2
    elif eje_tubo == 'Y':
        ind_largo = 1
        ind_base = 2
        ind_altura = 0
    elif eje_tubo == 'Z':
        ind_largo = 2
        ind_base = 0
        ind_altura = 1
    else:
        raise ValueError('Coordenada de eje de tubo mal especificada')

    #/ Seleccionar partículas del tipo deseado
    idx_particulas = tipo_particulas['data'][0] == id_tipo
    posiciones_particulas = posiciones['data'][:,idx_particulas,:]
    velocidades_particulas = velocidades['data'][:,idx_particulas,ind_largo] # componente longitudinal

    #/ CORRECCIÓN DE POSICIONES POR CURVATURA DE TUBO
    #/ ----------------------------------------------

    #/ Obtener índices de bins longitudinales (base-1)
    indices_bins = np.digitize(posiciones_particulas[:,:,ind_largo], bins_l)

    #/ Pliego a índice 1 y len(bins_l)-1 aquellos que sobrepasan (0 o len(bins_l))
    indices_bins = np.where(indices_bins==0, 1, indices_bins)
    indices_bins = np.where(indices_bins==len(bins_l), len(bins_l)-1, indices_bins)
    
    #/ Obtener centros de referencia para interpolación
    centros_actuales = centros_segmentos[indices_bins - 1]  # Convertir a base-0
    
    #/ Calcular centro longitudinal de cada bin actual
    centro_actual_largo = (bins_l[indices_bins - 1] + bins_l[indices_bins]) / 2.0
    
    #/ Determinar dirección de interpolación
    interpolacion_anterior = posiciones_particulas[:,:,ind_largo] < centro_actual_largo
    
    #/ Preparar pesos y centros adyacentes
    peso = np.zeros(posiciones_particulas[:,:,ind_largo].shape)
    centros_adyacentes = np.zeros_like(centros_actuales)
    
    #/ Caso 1: Interpolar con el segmento ANTERIOR (si existe)
    mascara_anterior = interpolacion_anterior & (indices_bins > 1)
    if np.any(mascara_anterior):
        idx_anterior = indices_bins[mascara_anterior] - 2  # -2 por base-1 de digitize
        centros_adyacentes[mascara_anterior] = centros_segmentos[idx_anterior]
        pos_bin_inferior = bins_l[idx_anterior]
        pos_bin_superior = bins_l[idx_anterior + 1]
        peso[mascara_anterior] = \
            (posiciones_particulas[mascara_anterior,ind_largo] - pos_bin_inferior) \
            / (pos_bin_superior - pos_bin_inferior)
    
    #/ Caso 2: Interpolar con el segmento SIGUIENTE (si existe)
    mascara_siguiente = np.logical_not(interpolacion_anterior) & (indices_bins < len(bins_l) - 1)
    if np.any(mascara_siguiente):
        idx_siguiente = indices_bins[mascara_siguiente]  # Ya en base-0 (centros_segmentos)
        centros_adyacentes[mascara_siguiente] = centros_segmentos[idx_siguiente]
        pos_bin_inferior = bins_l[idx_siguiente - 1]
        pos_bin_superior = bins_l[idx_siguiente]
        peso[mascara_siguiente] = \
            (posiciones_particulas[mascara_siguiente,ind_largo] - pos_bin_inferior) \
            / (pos_bin_superior - pos_bin_inferior)
    
    #/ Interpolación lineal de centros
    centros_interpolados = \
        (1 - peso[:,:,np.newaxis]) * centros_adyacentes \
        + peso [:,:,np.newaxis]* centros_actuales
    
    #/ Corregir posiciones
    posiciones_corregidas = posiciones_particulas.copy()
    posiciones_corregidas[:,:,[ind_base,ind_altura]] -= centros_interpolados[:,:,[ind_base,ind_altura]]

    #/ CÁLCULO DE PERFIL DE VELOCIDAD
    #/ ------------------------------
    
    #/ Calcular la posición radial corregida
    posiciones_radiales = np.linalg.norm(posiciones_corregidas[:,:,[ind_base,ind_altura]], axis=2)

    #/ Filtrar partículas dentro del tubo en longitud
    en_largo_tubo = \
        (posiciones_corregidas[:,:,ind_largo] >= limites_tubo_fraccion[0]) & \
        (posiciones_corregidas[:,:,ind_largo] <= limites_tubo_fraccion[1])
    
    #/ Definir bins radiales
    ancho_bin_r = fraccion_bin_r * sigma_ref # delta_r (mínimo)
    if volumenes_iguales: # delta_r variable, volúmenes fijos
        #/ Despejo el número de bins de la progresión aritmética en r
        #/ r_i^2 = i/N * R^2 --> r_i = R * sqrt(i/N)
        #/ r_(N-1) = R * sqrt((N-1) / N)
        #/ Si r_N = R --> delta-r_N = R * (1 - sqrt((N-1) / N))
        numero_bins_r = int(
            np.round(1 / (1 - (1 - ancho_bin_r / radio_tubo_fraccion)**2), 0))
        bordes_bins_r = np.sqrt(
            np.linspace(0, radio_tubo_fraccion**2, numero_bins_r+1))
    else: # 17/07/25: delta_r fijo (no ajusta a radio de tubo), volúmenes variables
        numero_bins_r = np.ceil(radio_tubo_fraccion / ancho_bin_r).astype('int')
        bordes_bins_r = np.arange(0, ancho_bin_r*(numero_bins_r+1), ancho_bin_r)
    ## else: # delta_r fijo (ajustable a radio de tubo), volúmenes variables
    ##     numero_bins_r = \
    ##         int(round((radio_tubo_fraccion - 0.0) / ancho_bin_r, 0))
    ##     bordes_bins_r = np.linspace(0, radio_tubo_fraccion, numero_bins_r+1)
    
    #/ Obtención de velocidades promedio por partícula y tiempo (asumo que los
    #/ frames son equidistantes temporalmente)
    conteos = np.zeros(numero_bins_r+1)
    suma_vels = np.zeros_like(conteos) # para media
    suma_vels2 = np.zeros_like(conteos) # para desvío std

    idx_posiciones_radiales = np.digitize(posiciones_radiales, bordes_bins_r)
    print(f'Forma idx posiciones radiales {idx_posiciones_radiales.shape}')
    idx_posiciones_radiales_en_tubo = np.where(
        en_largo_tubo, idx_posiciones_radiales, numero_bins_r+1) # asigno bin
            # exterior a partículas fuera del tubo. Los conteos en este bin son
            # luego borrados (también de suma_vels y suma_vels2)

    print(f'Forma idx posiciones radiales en tubo {idx_posiciones_radiales_en_tubo.shape}')
    print(f'Forma velocidades particulas {velocidades_particulas.shape}')

    np.add.at(conteos, idx_posiciones_radiales_en_tubo-1, 1)
    np.add.at(suma_vels, idx_posiciones_radiales_en_tubo-1, velocidades_particulas)
    np.add.at(suma_vels2, idx_posiciones_radiales_en_tubo-1, velocidades_particulas**2)
    ## np.add.at(conteos, idx_posiciones_radiales_en_tubo.ravel()-1, 1)
    ## np.add.at(suma_vels, idx_posiciones_radiales_en_tubo.ravel()-1, velocidades_particulas.ravel())
    ## np.add.at(suma_vels2, idx_posiciones_radiales_en_tubo.ravel()-1, velocidades_particulas.ravel()**2)

    conteos = np.delete(conteos, -1)
    suma_vels = np.delete(suma_vels, -1)
    suma_vels2 = np.delete(suma_vels2, -1)

    #/ Calcular estadística. Donde los conteos son 0,los valores para la
    #/ velocidad son NaN
    out_mean = np.full_like(suma_vels,np.nan)
    promedio_vels = np.divide(suma_vels, conteos, out=out_mean, where=(conteos>0))
    out_std = np.full_like(suma_vels,np.nan)
    desvio_vels = np.sqrt(
        np.divide(suma_vels2, conteos, out=out_std, where=(conteos>0)) - promedio_vels**2)
        

    return conteos, promedio_vels, desvio_vels, bordes_bins_r, limites_tubo_fraccion


def calcular_perfil_radial_velocidad_tubo_curvo_interpolado_por_cuadro_v2(
    posiciones,
    tipo_particulas,
    velocidades,
    id_tipo,
    limites_tubo,
    limite_radio,
    centros_segmentos,
    bins_l,
    fraccion_region=0.9,
    fraccion_radio=1.0,
    fraccion_bin_r=0.05,
    sigma_ref=3.5,
    volumenes_iguales=False,
    normalizar=False,
    eje_tubo='Z'):
    """
    Cálculo de perfil de velocidad según coordenada radial.
    - Interpola entre el segmento actual y el adyacente más cercano (anterior o
      siguiente).
    - Índices 0 o len(bins_l) indican partículas fuera del tubo (se descartan).
    """
    #/ Límites de fracción de tubo requerida
    largo_tubo, centro_tubo, limites_tubo_fraccion, largo_tubo_fraccion, \
    radio_tubo_fraccion = \
    calcular_limites_tubo(
        limites_tubo,
        limite_radio,
        fraccion_region=fraccion_region,
        fraccion_radio=fraccion_radio)

    #/ Determino índice de coordenada del eje del tubo
    if eje_tubo == 'X':
        ind_largo = 0
        ind_base = 1
        ind_altura = 2
    elif eje_tubo == 'Y':
        ind_largo = 1
        ind_base = 2
        ind_altura = 0
    elif eje_tubo == 'Z':
        ind_largo = 2
        ind_base = 0
        ind_altura = 1
    else:
        raise ValueError('Coordenada de eje de tubo mal especificada')

    #/ Seleccionar partículas del tipo deseado
    idx_particulas = tipo_particulas['data'][0] == id_tipo
    posiciones_particulas = posiciones['data'][:,idx_particulas,:]
    velocidades_particulas = velocidades['data'][:,idx_particulas,ind_largo] # componente longitudinal

    #/ CORRECCIÓN DE POSICIONES POR CURVATURA DE TUBO
    #/ ----------------------------------------------

    #/ Obtener índices de bins longitudinales (base-1)
    indices_bins = np.digitize(posiciones_particulas[:,:,ind_largo], bins_l)

    #/ Pliego a índice 1 y len(bins_l)-1 aquellos que sobrepasan (0 o len(bins_l))
    indices_bins = np.where(indices_bins==0, 1, indices_bins)
    indices_bins = np.where(indices_bins==len(bins_l), len(bins_l)-1, indices_bins)
    
    #/ Obtener centros de referencia para interpolación
    centros_actuales = centros_segmentos[indices_bins - 1]  # Convertir a base-0
    
    #/ Calcular centro longitudinal de cada bin actual
    centro_actual_largo = (bins_l[indices_bins - 1] + bins_l[indices_bins]) / 2.0
    
    #/ Determinar dirección de interpolación
    interpolacion_anterior = posiciones_particulas[:,:,ind_largo] < centro_actual_largo
    
    #/ Preparar pesos y centros adyacentes
    peso = np.zeros(posiciones_particulas[:,:,ind_largo].shape)
    centros_adyacentes = np.zeros_like(centros_actuales)
    
    #/ Caso 1: Interpolar con el segmento ANTERIOR (si existe)
    mascara_anterior = interpolacion_anterior & (indices_bins > 1)
    if np.any(mascara_anterior):
        idx_anterior = indices_bins[mascara_anterior] - 2  # -2 por base-1 de digitize
        centros_adyacentes[mascara_anterior] = centros_segmentos[idx_anterior]
        pos_bin_inferior = bins_l[idx_anterior]
        pos_bin_superior = bins_l[idx_anterior + 1]
        peso[mascara_anterior] = \
            (posiciones_particulas[mascara_anterior,ind_largo] - pos_bin_inferior) \
            / (pos_bin_superior - pos_bin_inferior)
    
    #/ Caso 2: Interpolar con el segmento SIGUIENTE (si existe)
    mascara_siguiente = np.logical_not(interpolacion_anterior) & (indices_bins < len(bins_l) - 1)
    if np.any(mascara_siguiente):
        idx_siguiente = indices_bins[mascara_siguiente]  # Ya en base-0 (centros_segmentos)
        centros_adyacentes[mascara_siguiente] = centros_segmentos[idx_siguiente]
        pos_bin_inferior = bins_l[idx_siguiente - 1]
        pos_bin_superior = bins_l[idx_siguiente]
        peso[mascara_siguiente] = \
            (posiciones_particulas[mascara_siguiente,ind_largo] - pos_bin_inferior) \
            / (pos_bin_superior - pos_bin_inferior)
    
    #/ Interpolación lineal de centros
    centros_interpolados = \
        (1 - peso[:,:,np.newaxis]) * centros_adyacentes \
        + peso [:,:,np.newaxis]* centros_actuales
    
    #/ Corregir posiciones
    posiciones_corregidas = posiciones_particulas.copy()
    posiciones_corregidas[:,:,[ind_base,ind_altura]] -= centros_interpolados[:,:,[ind_base,ind_altura]]

    #/ CÁLCULO DE PERFIL DE VELOCIDAD
    #/ ------------------------------
    
    #/ Calcular la posición radial corregida
    posiciones_radiales = np.linalg.norm(posiciones_corregidas[:,:,[ind_base,ind_altura]], axis=2)

    #/ Filtrar partículas dentro del tubo en longitud
    en_largo_tubo = \
        (posiciones_corregidas[:,:,ind_largo] >= limites_tubo_fraccion[0]) & \
        (posiciones_corregidas[:,:,ind_largo] <= limites_tubo_fraccion[1])
    
    #/ Definir bins radiales
    ancho_bin_r = fraccion_bin_r * sigma_ref # delta_r (mínimo)
    if volumenes_iguales: # delta_r variable, volúmenes fijos
        #/ Despejo el número de bins de la progresión aritmética en r
        #/ r_i^2 = i/N * R^2 --> r_i = R * sqrt(i/N)
        #/ r_(N-1) = R * sqrt((N-1) / N)
        #/ Si r_N = R --> delta-r_N = R * (1 - sqrt((N-1) / N))
        numero_bins_r = int(
            np.round(1 / (1 - (1 - ancho_bin_r / radio_tubo_fraccion)**2), 0))
        bordes_bins_r = np.sqrt(
            np.linspace(0, radio_tubo_fraccion**2, numero_bins_r+1))
    else: # 17/07/25: delta_r fijo (no ajusta a radio de tubo), volúmenes variables
        numero_bins_r = np.ceil(radio_tubo_fraccion / ancho_bin_r).astype('int')
        bordes_bins_r = np.arange(0, ancho_bin_r*(numero_bins_r+1), ancho_bin_r)
    ## else: # delta_r fijo (ajustable a radio de tubo), volúmenes variables
    ##     numero_bins_r = \
    ##         int(round((radio_tubo_fraccion - 0.0) / ancho_bin_r, 0))
    ##     bordes_bins_r = np.linspace(0, radio_tubo_fraccion, numero_bins_r+1)
    
    #/ Obtención de velocidades promedio por partícula y tiempo (asumo que los
    #/ frames son equidistantes temporalmente)
    idx_posiciones_radiales = np.digitize(posiciones_radiales, bordes_bins_r)
    idx_posiciones_radiales_en_tubo = np.where(
        en_largo_tubo, idx_posiciones_radiales, numero_bins_r+1) # asigno bin
            # exterior a partículas fuera del tubo. Los conteos en este bin son
            # luego borrados (también de suma_vels y suma_vels2)

    conteos = np.bincount(
        np.ravel(idx_posiciones_radiales_en_tubo-1),
        minlength=numero_bins_r+1)
    suma_vels = np.bincount(
        np.ravel(idx_posiciones_radiales_en_tubo-1),
        weights=velocidades_particulas.ravel(),
        minlength=numero_bins_r+1)
    suma_vels2 = np.bincount(
        np.ravel(idx_posiciones_radiales_en_tubo-1),
        weights=velocidades_particulas.ravel()**2,
        minlength=numero_bins_r+1)

    conteos = np.delete(conteos, -1)
    suma_vels = np.delete(suma_vels, -1)
    suma_vels2 = np.delete(suma_vels2, -1)

    #/ Calcular estadística. Donde los conteos son 0,los valores para la
    #/ velocidad son NaN
    out_mean = np.full_like(suma_vels,np.nan)
    promedio_vels = np.divide(suma_vels, conteos, out=out_mean, where=(conteos>0))
    out_std = np.full_like(suma_vels,np.nan)
    desvio_vels = np.sqrt(
        np.divide(suma_vels2, conteos, out=out_std, where=(conteos>0)) - promedio_vels**2)
        

    return conteos, promedio_vels, desvio_vels, bordes_bins_r, limites_tubo_fraccion


#/ ---- FUNCIONES PARA IMPRIMIR INFORMACIÓN -------------------------------


#/ ---- PARSER GENERAL PARA OTROS SCRIPTS ---------------------------------
def opciones_analisis(descripcion=None):
    """
    Parser con opciones comunes a análisis y manipulaciones de archivos de
    trayectoria netCDF generados con LAMMPS.

    Devuelve
    --------
    opciones: Namespace
        Colección con nombres de parámetros y opciones escogidas.
    """
    parser = argparse.ArgumentParser(
        description=descripcion,
        formatter_class=argparse.MetavarTypeHelpFormatter
    )

    parser.add_argument(
        '-sr', '--sigma_referencia',
        help=("Distancia sigma de referencia (sigma_ref)"),
        required=False,
        nargs=None,
        action="store",
        type=float,
        default=3.5
    )

    parser.add_argument(
        '-rn', '--radio_nominal',
        help=("Radio nominal del tubo (medido en sigma_ref)"),
        required=False,
        nargs=None,
        action="store",
        type=float,
        default=3.0
    )

    parser.add_argument(
        '-ln', '--largo_nominal',
        help=("Largo nominal del tubo (medido en sigma_ref)"),
        required=False,
        nargs=None,
        action="store",
        type=float,
        default=37.5
    )

    parser.add_argument(
        '-rr', '--radio_real',
        help=("Radio medio real del tubo (medido en angstroms)"),
        required=False,
        nargs=None,
        action="store",
        type=float,
    )

    parser.add_argument(
        '-lr', '--largo_real',
        help=("Largo real del tubo (medido en angstroms)"),
        required=False,
        nargs=None,
        action="store",
        type=float,
    )

    parser.add_argument(
        '-X', '--fraccion_molar',
        help="Fracción molar del fluido de la simulación",
        required=True,
        nargs=None,
        action="store",
        type=float
    )

    parser.add_argument(
        '-dE', '--direccion_campo_electrico',
        help="Dirección de campo eléctrico de la simulación",
        required=False,
        nargs=None,
        action="store",
        type=str,
        default=None
    )

    parser.add_argument(
        '-vE', '--valor_campo_electrico',
        help="Valor de campo eléctrico de la simulación",
        required=False,
        nargs=None,
        action="store",
        type=float,
        default=None
    )

    parser.add_argument(
        '-sd', '--subdirectorio',
        help="Tipo principal de simulación",
        required=False,
        nargs=None,
        action="store",
        type=str,
        default="vacio"
    )

    parser.add_argument(
        '-at', '--trayectoria',
        help="Nombre de simulación con archivo de trayectoria",
        required=True,
        nargs=None,
        action="store",
        type=str
    )

    parser.add_argument(
        '-st', '--sufijo_trayectoria',
        help="Sufijo del archivo de trayectoria",
        required=False,
        nargs=None,
        action="store",
        type=str,
        default="nc"
    )

    parser.add_argument(
        '-db', '--dir_base',
        help="Directorio base donde se ubican las simulaciones de los sistemas",
        required=False,
        nargs=None,
        action="store",
        type=str,
        default="/yggdrasil/users/smosca/tip4p-2005/campo_electrico/run/reservorios"
    )

    parser.add_argument(
        '-df', '--fuerza',
        help="Directorio de fuerzas donde se ubican las simulaciones de los sistemas",
        required=False,
        nargs=None,
        action="store",
        type=int,
        default=None
    )

    parser.add_argument(
        '-n', '--nombre',
        help="Nombre común de los archivos de configuración, trayectoria e imágenes",
        required=False,
        nargs=None,
        action="store",
        type=str,
        default="reservorios.R{}.L{}.X{}.{}"
    )

    opciones = parser.parse_args()

    return opciones


#/ ---- EJECUCIÓN DEL SCRIPT ----------------------------------------------
if __name__ == '__main__':
    main()


#/ ---- FIN SCRIPT --------------------------------------------------------
