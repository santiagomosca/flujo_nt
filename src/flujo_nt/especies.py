#! /usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Especies de partículas del sistema simulado.

Una especie asocia un tipo de partícula de LAMMPS con un nombre legible, el
código corto con que se nombran los archivos de resultados, y la masa de la
molécula a la que pertenece. Reúne en un solo lugar la información que antes
se repetía en un diccionario dentro de cada script de análisis.

La tabla :data:`ESPECIES_AGUA_METANOL` describe las mezclas de agua TIP4P/2005
y metanol de las simulaciones de referencia. Para otro sistema basta con
construir otra tabla de objetos :class:`Especie`.

Examples
--------
>>> oxigeno_agua = buscar_especie('Oa')
>>> oxigeno_agua.tipo, oxigeno_agua.masa_molecular
(1, 18.0154)

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: noviembre de 2026
"""


#/ ---- ESPECIE ----------------------------------------------------------

class Especie:
    """
    Especie de partícula del sistema.

    Parameters
    ----------
    tipo : int
        Tipo de partícula en la simulación de LAMMPS.
    nombre : str
        Nombre legible, para mensajes y encabezados de archivos.
    codigo : str
        Código corto, para nombres de archivos de resultados.
    masa_molecular : float, optional
        Masa de la molécula a la que pertenece la partícula, en g/mol. Se
        define sólo para la partícula que representa a la molécula en el
        cálculo del flujo másico, de modo que cada molécula se cuente una
        vez. Por defecto None.

    Attributes
    ----------
    tipo : int
        Tipo de partícula en la simulación de LAMMPS.
    nombre : str
        Nombre legible.
    codigo : str
        Código corto.
    masa_molecular : float or None
        Masa de la molécula, en g/mol, o None si la partícula no representa
        a su molécula.
    """

    def __init__(self, tipo, nombre, codigo, masa_molecular=None):
        self.tipo = int(tipo)
        self.nombre = nombre
        self.codigo = codigo
        self.masa_molecular = masa_molecular

    def __repr__(self):
        return (f'Especie({self.tipo}, {self.nombre!r}, {self.codigo!r}, '
                f'masa_molecular={self.masa_molecular})')


#/ ---- TABLA DEL SISTEMA AGUA-METANOL ------------------------------------

ESPECIES_AGUA_METANOL = (
    Especie(1, 'oxígeno H2O', 'Oa', masa_molecular=18.0154),
    Especie(2, 'hidrógeno H2O', 'Ha'),
    Especie(3, 'oxígeno MeOH', 'Om', masa_molecular=32.0416),
    Especie(4, 'metilo MeOH', 'Cm'),
    Especie(5, 'hidrógeno MeOH', 'Hm'))
"""tuple of Especie: Especies del fluido en las mezclas de agua TIP4P/2005 y
metanol. El oxígeno representa a cada molécula en el cálculo del flujo; el
metilo es un átomo unido (CH3). Los tipos 6 y 7 son carbonos de la estructura
y no forman parte del fluido."""


#/ ---- BÚSQUEDA ----------------------------------------------------------

def buscar_especie(clave, especies=ESPECIES_AGUA_METANOL):
    """
    Búsqueda de una especie por su código o por su tipo de partícula.

    Parameters
    ----------
    clave : str or int
        Código de la especie (p. ej. ``'Oa'``) o tipo de partícula.
    especies : iterable of Especie, optional
        Tabla en la que buscar. Por defecto :data:`ESPECIES_AGUA_METANOL`.

    Returns
    -------
    Especie
        Especie encontrada.

    Raises
    ------
    KeyError
        Si ninguna especie de la tabla tiene ese código o tipo.
    """
    for especie in especies:
        if clave == especie.codigo or clave == especie.tipo:
            return especie

    disponibles = [f'{e.codigo} ({e.tipo})' for e in especies]
    raise KeyError(f'Especie desconocida: {clave!r}. Disponibles: '
                   f'{", ".join(disponibles)}')
