#! /usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Figura de rendimiento de la memoria: pico de memoria y tiempo real según el
tamaño de bloque, a partir de la tabla de medir_rendimiento.py.

Usa el mismo estilo (figuras.mplstyle) y el mismo tamaño que las figuras de
dos paneles del artículo, para que ambas se vean iguales en la memoria.
Requiere LaTeX con siunitx, porque el estilo usa text.usetex.

Uso:
    python rendimiento.py [rendimiento.dat] [rendimiento.pdf]

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: noviembre de 2026
"""

import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colormaps as cmap  # noqa: E402
import numpy as np  # noqa: E402

DIRECTORIO = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(DIRECTORIO, 'figuras.mplstyle'))
plt.rcParams['markers.fillstyle'] = 'none'  # marcadores vacíos


#/ ---- CONFIGURACIÓN DEL GRÁFICO ---------------------------------------

COLOR = cmap['Paired'].colors[1]  # azul, como el metanol en el artículo
GRIS = '0.45'

#/ Mismo tamaño que las figuras de dos paneles del artículo
TAMANIO = (2 * plt.rcParams['figure.figsize'][0],
           plt.rcParams['figure.figsize'][1])

dax_tick_params = {
    'axis': 'both',
    'which': 'both',
    'direction': 'in',
    'bottom': True,
    'top': True,
    'left': True,
    'right': True}

dax_subplot_adjust = {
    'left': 0.08,
    'right': 0.99,
    'top': 0.99,
    'bottom': 0.17,
    'wspace': 0.22}


#/ ---- DATOS -------------------------------------------------------------

def leer(ruta):
    """Tabla de mediciones, cantidad de cuadros y memoria base en MB."""
    tabla = np.loadtxt(ruta, ndmin=2)
    with open(ruta, encoding='utf-8') as archivo:
        encabezado = archivo.read()
    cuadros = int(re.search(r'cuadros = (\d+)', encabezado).group(1))
    base = float(re.search(r'base: .*memoria_max = ([\d.]+) MB',
                           encabezado).group(1))
    return tabla, cuadros, base


def promediar(tabla, cuadros):
    """Media de cada columna por tamaño de bloque; -1 pasa a ser cuadros."""
    tamanios = tabla[:, 0].copy()
    tamanios[tamanios < 0] = cuadros
    unicos = np.unique(tamanios)
    media = np.array([tabla[tamanios == t, 2:].mean(axis=0) for t in unicos])
    return unicos, media


#/ ---- GRÁFICO -----------------------------------------------------------

def graficar(ruta_datos, ruta_figura):
    tabla, cuadros, base = leer(ruta_datos)
    tamanios, media = promediar(tabla, cuadros)
    tiempo, memoria = media[:, 0], media[:, 2]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=TAMANIO)
    fig.subplots_adjust(**dax_subplot_adjust)

    estilo = {'color': COLOR, 'marker': 'o', 'linewidth': 1.5}

    #/ (a) Pico de memoria, en escala logarítmica
    ax[0].plot(tamanios, memoria, **estilo)
    ax[0].axhline(base, color=GRIS, linestyle='--', linewidth=1.)
    ax[0].text(tamanios[-1], base * 1.15, r'$\mathrm{base}$', color=GRIS,
               ha='right', va='bottom',
               fontsize=plt.rcParams['legend.fontsize'])
    ax[0].set_yscale('log')
    ax[0].set_ylabel(r'$\mathrm{memoria} \ \left[ \mathrm{MB} \right]$')

    #/ (b) Tiempo real
    ax[1].plot(tamanios, tiempo, **estilo)
    ymax = tiempo.max() * 1.2
    ax[1].set_ylim((0, ymax))
    ax[1].set_ylabel(r'$t \ \left[ \mathrm{s} \right]$')
    ax[1].text(55, 0.97 * ymax, r'$\mathrm{por \ defecto}$', color=GRIS,
               ha='left', va='top', fontsize=plt.rcParams['legend.fontsize'])

    for etiqueta, eje in zip(('(a)', '(b)'), ax):
        eje.set_xscale('log')
        eje.set_xticks(tamanios)
        eje.set_xticklabels([rf'${int(t)}$' for t in tamanios])
        eje.minorticks_off()
        eje.set_xlim((0.7, cuadros * 1.4))
        eje.axvline(50, color=GRIS, linestyle=':', linewidth=1., zorder=0)
        eje.tick_params(**dax_tick_params)
        eje.grid(visible=False)
        eje.text(0.04, 0.95, rf'$\mathrm{{{etiqueta}}}$',
                 transform=eje.transAxes, ha='left', va='top',
                 fontsize=plt.rcParams['axes.labelsize'])

    #/ Etiqueta común para el eje X, como en las figuras del artículo
    x_centro = (dax_subplot_adjust['left'] + dax_subplot_adjust['right']) / 2
    fig.text(x_centro, 0.01,
             r'$\mathrm{tama\tilde{n}o \ de \ bloque} \ '
             r'\left[ \mathrm{cuadros} \right]$',
             fontsize=plt.rcParams['axes.labelsize'], ha='center',
             va='bottom')

    fig.savefig(ruta_figura)
    plt.close(fig)


if __name__ == '__main__':
    datos = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        DIRECTORIO, 'rendimiento.dat')
    salida = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        DIRECTORIO, 'rendimiento.pdf')
    graficar(datos, salida)


#/ ---- FIN SCRIPT --------------------------------------------------------
