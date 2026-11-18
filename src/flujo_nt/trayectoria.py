#! /usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Lectura de trayectorias de dinámica molecular en formato netCDF.

Este módulo interpreta trayectorias con convención AMBER generadas por LAMMPS
o reescritas por OVITO, y archivos de configuración de LAMMPS (formato data).

El objeto :class:`Trayectoria` es una vista perezosa sobre el archivo abierto,
no un contenedor de arreglos. Los metadatos baratos (tiempos, tipos, celda) se
leen al construir el objeto; los datos pesados (posiciones, velocidades,
fuerzas, imágenes periódicas) se leen por bloques de cuadros y sólo cuando se
piden. Una trayectoria de mil cuadros y cien mil partículas ocupa así unos
cientos de megabytes en lugar de varios gigabytes.

Clases
------
Trayectoria
    Vista perezosa sobre una trayectoria netCDF.
Bloque
    Ventana de cuadros consecutivos de una trayectoria.
Configuracion
    Archivo de configuración de LAMMPS (formato data).
Celda
    Celda de simulación ortogonal.
Dialecto
    Traducción entre nombres canónicos y nombres de variables del archivo.

Examples
--------
>>> configuracion = Configuracion('sistema.final.data')
>>> with Trayectoria('sistema.nc', configuracion=configuracion) as tr:
...     mascara = tr.mascara_tipo(1)
...     for bloque in tr.bloques(tamanio=50):
...         posiciones = bloque.posiciones[:, mascara]

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: julio de 2026
"""

import os

import numpy as np
from netCDF4 import Dataset


#/ ---- CONSTANTES DEL MÓDULO --------------------------------------------

TAMANIO_BLOQUE_PREDETERMINADO = 50
"""int: Cantidad de cuadros por bloque si no se especifica otra cosa."""

DIMENSION_CUADRO = 'frame'
"""str: Nombre de la dimensión temporal en la convención AMBER."""

TOLERANCIA_CELDA = 1e-6
"""float: Tolerancia relativa para considerar constante una longitud
de celda."""


#/ ---- EXCEPCIONES -------------------------------------------------------

class ErrorTrayectoria(Exception):
    """
    Error en la lectura o validación de una trayectoria o configuración.

    Se define una excepción propia para que el código que use este módulo
    pueda distinguir sus fallas de las de netCDF4 o numpy, y para no
    interrumpir el intérprete desde una biblioteca.
    """


#/ ---- DIALECTOS ---------------------------------------------------------

class Dialecto:
    """
    Traducción entre nombres canónicos y nombres de variables del archivo.

    LAMMPS y OVITO escriben trayectorias con la misma convención AMBER pero
    con distinta nomenclatura para algunas variables: donde LAMMPS escribe
    ``type`` e ``id``, OVITO escribe ``atom_types`` e ``identifier``.
    Encapsular la diferencia acá permite que el resto de la biblioteca ignore
    su existencia y pida siempre nombres canónicos.

    Parameters
    ----------
    nombre : str
        Nombre del dialecto. Debe ser una clave de :attr:`TABLAS`.

    Attributes
    ----------
    nombre : str
        Nombre del dialecto.
    variables : dict
        Traducción de nombre canónico a nombre en el archivo. El valor es una
        cadena, salvo para las variables que el archivo almacena en varias
        componentes escalares, donde es una tupla de cadenas.

    Raises
    ------
    ErrorTrayectoria
        Si el nombre del dialecto no se reconoce.

    Examples
    --------
    >>> dialecto = Dialecto('OVITO')
    >>> dialecto.nombre_en_archivo('tipos')
    'atom_types'
    """

    VARIABLES_LAMMPS = {
        'posiciones': 'coordinates',
        'velocidades': 'velocities',
        'fuerzas': 'forces',
        'imagenes': ('ix', 'iy', 'iz'),
        'tipos': 'type',
        'identificadores': 'id',
        'moleculas': 'mol',
        'tiempos': 'time',
        'longitudes_celda': 'cell_lengths',
        'origen_celda': 'cell_origin',
        'angulos_celda': 'cell_angles'}
    """dict: Traducción para trayectorias escritas por LAMMPS."""

    VARIABLES_OVITO = dict(VARIABLES_LAMMPS,
        tipos='atom_types',
        identificadores='identifier')
    """dict: Traducción para trayectorias reescritas por OVITO."""

    TABLAS = {
        'LAMMPS': VARIABLES_LAMMPS,
        'OVITO': VARIABLES_OVITO}
    """dict: Traducciones disponibles, por nombre de dialecto."""

    def __init__(self, nombre):
        if nombre not in self.TABLAS:
            raise ErrorTrayectoria(
                f'Dialecto desconocido: {nombre!r}. '
                f'Conocidos: {sorted(self.TABLAS)}')

        self.nombre = nombre
        self.variables = self.TABLAS[nombre]

    @classmethod
    def detectar(cls, ds):
        """
        Identificación del dialecto de un Dataset abierto.

        Se usa el atributo global ``program`` de la convención AMBER. Si no
        está presente o no se reconoce, se recurre a los nombres de variables.
        Si tampoco así se resuelve, se levanta un error en lugar de adivinar:
        un dialecto mal identificado produce KeyError en el mejor caso, y
        lecturas de la variable equivocada en el peor.

        Parameters
        ----------
        ds : netCDF4.Dataset
            Trayectoria abierta en modo lectura.

        Returns
        -------
        Dialecto
            Dialecto identificado.

        Raises
        ------
        ErrorTrayectoria
            Si no se puede identificar el dialecto.
        """
        programa = getattr(ds, 'program', None)

        if programa in cls.TABLAS:
            return cls(programa)

        #/ Identificación por nombres de variables
        if 'type' in ds.variables:
            return cls('LAMMPS')

        if 'atom_types' in ds.variables:
            return cls('OVITO')

        raise ErrorTrayectoria(
            f'No se pudo identificar el dialecto de la trayectoria. '
            f'Atributo program = {programa!r}. '
            f'Variables presentes: {sorted(ds.variables)}. '
            f'Especificar el dialecto explícitamente.')

    def nombre_en_archivo(self, canonico):
        """
        Nombre que tiene en el archivo una variable canónica.

        Parameters
        ----------
        canonico : str
            Nombre canónico de la variable, p. ej. ``'posiciones'``.

        Returns
        -------
        str or tuple of str
            Nombre de la variable en el archivo. Es una tupla para las
            variables almacenadas en componentes escalares separadas, como las
            imágenes periódicas.

        Raises
        ------
        ErrorTrayectoria
            Si el nombre canónico no pertenece al dialecto.
        """
        if canonico not in self.variables:
            raise ErrorTrayectoria(
                f'Variable canónica desconocida: {canonico!r}. '
                f'Conocidas: {sorted(self.variables)}')

        return self.variables[canonico]

    def __repr__(self):
        return f'Dialecto({self.nombre!r})'


#/ ---- CELDA -------------------------------------------------------------

class Celda:
    """
    Celda de simulación ortogonal, constante a lo largo de la trayectoria.

    Se asume constancia porque las simulaciones de referencia son NVT. La
    constancia se verifica al leer la trayectoria; si no se cumple se levanta
    un error, en lugar de promediar o tomar el primer cuadro en silencio.

    Parameters
    ----------
    longitudes : array_like
        Longitudes de la celda por dirección, con forma ``(3,)``, en Å.
    origen : array_like
        Esquina inferior de la celda, con forma ``(3,)``, en Å.
    angulos : array_like, optional
        Ángulos de la celda, con forma ``(3,)``, en grados. Por defecto son
        rectos.

    Attributes
    ----------
    longitudes : numpy.ndarray
        Longitudes de la celda, con forma ``(3,)`` y tipo float64, en Å.
    origen : numpy.ndarray
        Esquina inferior de la celda, con forma ``(3,)`` y tipo float64, en Å.
    angulos : numpy.ndarray
        Ángulos de la celda, con forma ``(3,)`` y tipo float64, en grados.

    Raises
    ------
    ErrorTrayectoria
        Si las longitudes o el origen no tienen forma ``(3,)``.
    """

    def __init__(self, longitudes, origen, angulos=None):
        self.longitudes = np.asarray(longitudes, dtype='f8')
        self.origen = np.asarray(origen, dtype='f8')

        if angulos is None:
            angulos = np.array([90.0, 90.0, 90.0])
        self.angulos = np.asarray(angulos, dtype='f8')

        if self.longitudes.shape != (3,):
            raise ErrorTrayectoria(
                f'Longitudes de celda con forma inesperada: '
                f'{self.longitudes.shape}')

        if self.origen.shape != (3,):
            raise ErrorTrayectoria(
                f'Origen de celda con forma inesperada: {self.origen.shape}')

    @property
    def limites(self):
        """
        numpy.ndarray: Límites inferior y superior de la celda por dirección.

        Tiene forma ``(3, 2)`` y tipo float64, en Å, ordenados como
        ``[[xlo, xhi], [ylo, yhi], [zlo, zhi]]``.

        Se calcula en cada acceso a partir de :attr:`origen` y
        :attr:`longitudes`, de modo que no puede quedar desactualizado
        respecto de ellos.
        """
        return np.stack([self.origen, self.origen + self.longitudes], axis=1)

    @property
    def degenerada(self):
        """
        bool: Verdadero si alguna longitud de la celda es nula.

        LAMMPS escribe longitud nula en la dirección no periódica cuando se
        usa ``kspace_modify slab``, aunque la longitud verdadera exista y esté
        registrada en el archivo de configuración. Una celda degenerada no
        alcanza para analizar la trayectoria.
        """
        return bool(np.any(self.longitudes == 0.0))

    def __repr__(self):
        lx, ly, lz = self.longitudes
        return (f'Celda(longitudes=[{lx:.4f}, {ly:.4f}, {lz:.4f}], '
                f'origen=[{self.origen[0]:.4f}, {self.origen[1]:.4f}, '
                f'{self.origen[2]:.4f}])')


#/ ---- CONFIGURACIÓN -----------------------------------------------------

class Configuracion:
    """
    Archivo de configuración de LAMMPS (formato data).

    Describe el estado de un sistema de partículas en un instante, típicamente
    el último de una simulación. Es una clase hermana de :class:`Trayectoria`,
    no subordinada: existe por derecho propio, y es además el único lugar donde
    consta la longitud verdadera de la celda en las simulaciones con slab.

    Por ahora se interpreta únicamente el encabezado, que es lo que se necesita
    para corregir la celda de una trayectoria. La lectura de las secciones de
    partículas queda para cuando se implemente la rotación y reescritura de
    configuraciones.

    Parameters
    ----------
    archivo : str
        Ruta al archivo de configuración.

    Attributes
    ----------
    archivo : str
        Ruta al archivo de configuración.
    celda : Celda
        Celda de simulación declarada en el encabezado.
    n_atomos : int or None
        Cantidad de partículas declarada en el encabezado.
    n_tipos : int or None
        Cantidad de tipos de partícula declarada en el encabezado.
    masas : dict
        Masa por tipo de partícula, en g/mol. Las claves son los
        identificadores de tipo.

    Raises
    ------
    ErrorTrayectoria
        Si el archivo no existe o si falta algún límite de celda.

    Examples
    --------
    >>> configuracion = Configuracion('sistema.final.data')
    >>> configuracion.celda.longitudes[2]
    522.66
    """

    def __init__(self, archivo):
        if not os.path.isfile(archivo):
            raise ErrorTrayectoria(
                f'No existe el archivo de configuración: {archivo}')

        self.archivo = archivo
        self.n_atomos = None
        self.n_tipos = None
        self.masas = {}

        self._leer_encabezado()

    def _leer_encabezado(self):
        """
        Interpretación del encabezado del archivo data.

        Se leen los límites de la celda, la cantidad de átomos y de tipos, y
        las masas por tipo. La lectura se detiene al llegar a la sección
        ``Atoms``, de modo que el costo es independiente del tamaño del
        sistema.

        Raises
        ------
        ErrorTrayectoria
            Si falta algún límite de celda.
        """
        limites = {}
        seccion = None

        with open(self.archivo, 'r') as f:
            for linea in f:
                linea = linea.split('#')[0].strip()

                if not linea:
                    continue

                #/ Fin del encabezado
                if linea.startswith('Atoms'):
                    break

                if linea.startswith('Masses'):
                    seccion = 'Masses'
                    continue

                if linea[0].isalpha():
                    seccion = linea
                    continue

                campos = linea.split()

                if seccion == 'Masses' and len(campos) == 2:
                    self.masas[int(campos[0])] = float(campos[1])
                    continue

                #/ Límites de celda: '-31.97 31.97 xlo xhi'
                if len(campos) == 4 and campos[2].endswith('lo'):
                    direccion = campos[2][0]
                    limites[direccion] = (float(campos[0]), float(campos[1]))
                    continue

                #/ Conteos: '101112 atoms' / '7 atom types'
                if len(campos) == 2 and campos[1] == 'atoms':
                    self.n_atomos = int(campos[0])
                    continue

                if len(campos) == 3 and campos[1:] == ['atom', 'types']:
                    self.n_tipos = int(campos[0])
                    continue

        faltantes = {'x', 'y', 'z'} - set(limites)
        if faltantes:
            raise ErrorTrayectoria(
                f'Faltan límites de celda en {self.archivo}: '
                f'{sorted(faltantes)}')

        origen = [limites[d][0] for d in 'xyz']
        longitudes = [limites[d][1] - limites[d][0] for d in 'xyz']

        self.celda = Celda(longitudes, origen)

    def __repr__(self):
        return (f'Configuracion({os.path.basename(self.archivo)!r}, '
                f'n_atomos={self.n_atomos}, n_tipos={self.n_tipos})')


#/ ---- BLOQUE ------------------------------------------------------------

class Bloque:
    """
    Ventana de cuadros consecutivos de una trayectoria.

    Expone los datos pesados con forma ``(n_cuadros, n_atomos, 3)``, que es la
    forma sobre la que opera el código vectorizado de los análisis. Cada
    variable se lee del disco la primera vez que se la pide y queda cacheada
    mientras el bloque viva: un análisis que sólo usa posiciones nunca paga la
    lectura de fuerzas.

    No se construye directamente sino a través de :meth:`Trayectoria.bloques`
    o :meth:`Trayectoria.cuadro`.

    Parameters
    ----------
    trayectoria : Trayectoria
        Trayectoria de la que proviene el bloque.
    inicio : int
        Índice del primer cuadro del bloque.
    fin : int
        Índice siguiente al del último cuadro del bloque.

    Attributes
    ----------
    trayectoria : Trayectoria
        Trayectoria de la que proviene el bloque.
    inicio : int
        Índice del primer cuadro del bloque.
    fin : int
        Índice siguiente al del último cuadro del bloque.

    Notes
    -----
    En este módulo, ``n_cuadros`` en la forma de los arreglos de un bloque se
    refiere a la cantidad de cuadros del bloque, ``len(bloque)``, no a la de
    la trayectoria completa.
    """

    def __init__(self, trayectoria, inicio, fin):
        self.trayectoria = trayectoria
        self.inicio = inicio
        self.fin = fin
        self._cache = {}

    def __len__(self):
        return self.fin - self.inicio

    @property
    def indices(self):
        """
        numpy.ndarray: Índices de los cuadros del bloque en la trayectoria.

        Tiene forma ``(n_cuadros,)`` y tipo entero.
        """
        return np.arange(self.inicio, self.fin)

    @property
    def tiempos(self):
        """
        numpy.ndarray: Tiempos de los cuadros del bloque.

        Tiene forma ``(n_cuadros,)``, en femtosegundos.
        """
        return self.trayectoria.tiempos[self.inicio:self.fin]

    def _leer(self, canonico):
        """
        Lectura perezosa y cacheada de una variable del bloque.

        El caché es lo que vuelve utilizable a la pereza: sin él, un acceso
        repetido a la misma variable dentro de un bucle releería el disco en
        cada vuelta, sin que nada en el código lo delatara.

        Parameters
        ----------
        canonico : str
            Nombre canónico de la variable.

        Returns
        -------
        numpy.ndarray
            Datos de la variable para los cuadros del bloque.

        Raises
        ------
        ErrorTrayectoria
            Si la variable no está presente en la trayectoria.
        """
        if canonico in self._cache:
            return self._cache[canonico]

        datos = self.trayectoria._leer_bloque(canonico, self.inicio, self.fin)
        self._cache[canonico] = datos

        return datos

    @property
    def posiciones(self):
        """
        numpy.ndarray: Posiciones de las partículas.

        Tiene forma ``(n_cuadros, n_atomos, 3)``, en Å. El tipo depende del
        dialecto: float32 en trayectorias de LAMMPS, float64 en las
        reescritas por OVITO.

        Se lee del disco en el primer acceso.
        """
        return self._leer('posiciones')

    @property
    def velocidades(self):
        """
        numpy.ndarray: Velocidades de las partículas.

        Tiene forma ``(n_cuadros, n_atomos, 3)``, en Å/fs.

        Se lee del disco en el primer acceso.
        """
        return self._leer('velocidades')

    @property
    def fuerzas(self):
        """
        numpy.ndarray: Fuerzas sobre las partículas.

        Tiene forma ``(n_cuadros, n_atomos, 3)``, en (kcal/mol)/Å.

        Se lee del disco en el primer acceso. No todas las trayectorias
        registran fuerzas; consultar :attr:`Trayectoria.variables_disponibles`.
        """
        return self._leer('fuerzas')

    @property
    def imagenes(self):
        """
        numpy.ndarray: Índices de imagen periódica de las partículas.

        Tiene forma ``(n_cuadros, n_atomos, 3)`` y tipo entero. Se ensambla a
        partir de las variables escalares ``ix``, ``iy`` e ``iz`` del archivo,
        que esta biblioteca expone como una sola variable vectorial.

        Se lee del disco en el primer acceso.
        """
        return self._leer('imagenes')

    @property
    def moleculas(self):
        """
        numpy.ndarray: Identificador de molécula de cada partícula.

        Tiene forma ``(n_cuadros, n_atomos)`` y tipo entero.

        Se lee del disco en el primer acceso.
        """
        return self._leer('moleculas')

    @property
    def identificadores(self):
        """
        numpy.ndarray: Identificador de cada partícula.

        Tiene forma ``(n_cuadros, n_atomos)`` y tipo entero.

        Se lee del disco en el primer acceso.
        """
        return self._leer('identificadores')

    def liberar(self):
        """
        Descarte de los datos cacheados del bloque.

        Útil para soltar memoria a mitad de un bloque cuando ya no se
        necesitan las variables leídas. No hace falta al terminar de usar el
        bloque: el generador de :meth:`Trayectoria.bloques` lo descarta al
        avanzar.

        Returns
        -------
        None
        """
        self._cache.clear()

    def __repr__(self):
        return (f'Bloque(cuadros {self.inicio}:{self.fin}, '
                f'en cache: {sorted(self._cache)})')


#/ ---- TRAYECTORIA -------------------------------------------------------

class Trayectoria:
    """
    Vista perezosa sobre una trayectoria netCDF con convención AMBER.

    Al construirse se leen sólo los metadatos: tiempos, tipos de partícula
    (una vez, del primer cuadro), celda, atributos y dimensiones. Los datos
    pesados se obtienen iterando bloques con :meth:`bloques`.

    Debe usarse como administrador de contexto para garantizar el cierre del
    archivo aunque un análisis falle a mitad de camino.

    Parameters
    ----------
    archivo : str
        Ruta a la trayectoria netCDF.
    configuracion : Configuracion, optional
        Configuración de la que tomar las longitudes de celda. Sus longitudes
        prevalecen sobre las de la trayectoria. Es obligatoria si la celda de
        la trayectoria es degenerada.
    dialecto : str, optional
        Nombre del dialecto, ``'LAMMPS'`` u ``'OVITO'``. Si es None se detecta
        a partir del archivo.
    verificar : bool, optional
        Si es verdadero, se comprueban al abrir los invariantes que el resto
        de la biblioteca da por sentados. Por defecto True.

    Attributes
    ----------
    archivo : str
        Ruta a la trayectoria netCDF.
    configuracion : Configuracion or None
        Configuración usada para corregir la celda.
    dialecto : Dialecto
        Dialecto de la trayectoria.
    n_cuadros : int
        Cantidad de cuadros de la trayectoria.
    n_atomos : int
        Cantidad de partículas del sistema.
    tiempos : numpy.ndarray
        Tiempos de todos los cuadros, con forma ``(n_cuadros,)``, en
        femtosegundos. El factor de escala del archivo ya está aplicado.
    tipos : numpy.ndarray
        Tipo de cada partícula, con forma ``(n_atomos,)`` y tipo entero.
        Leído del primer cuadro; su validez para toda la trayectoria se
        verifica al abrir.
    celda : Celda
        Celda de simulación, ya corregida si se pasó una configuración.
    atributos : dict
        Atributos globales del archivo netCDF.
    dimensiones : dict
        Dimensiones del archivo netCDF. El valor es None para la dimensión
        ilimitada.

    Raises
    ------
    ErrorTrayectoria
        Si el archivo no existe, si no se puede identificar el dialecto, si
        falta alguna variable o dimensión esperada, si la celda es degenerada
        y no se pasó una configuración, o si no se cumple algún invariante.

    See Also
    --------
    Bloque : Ventana de cuadros consecutivos.
    Configuracion : Archivo de configuración de LAMMPS.

    Notes
    -----
    Al abrir se desactiva el enmascarado automático de netCDF4 pero no el
    escalado. Las trayectorias de LAMMPS traen un factor de escala en la
    variable de tiempo, igual al paso de integración en femtosegundos;
    desactivar el escalado devolvería números de paso en lugar de tiempos
    físicos, sin ningún error visible.

    Examples
    --------
    >>> configuracion = Configuracion('sistema.final.data')
    >>> with Trayectoria('sistema.nc', configuracion=configuracion) as tr:
    ...     tr.n_cuadros
    ...     mascara = tr.mascara_tipo(1)
    ...     for bloque in tr.bloques(tamanio=50):
    ...         posiciones = bloque.posiciones[:, mascara]
    """

    def __init__(self, archivo, configuracion=None, dialecto=None,
            verificar=True):
        if not os.path.isfile(archivo):
            raise ErrorTrayectoria(f'No existe la trayectoria: {archivo}')

        self.archivo = archivo
        self.configuracion = configuracion

        self._ds = Dataset(archivo, 'r')

        #/ Se desactiva el enmascarado automático pero NO el escalado, por lo
        #/ explicado en las notas de la clase. Es correcto llamarlo a nivel
        #/ Dataset porque todas las variables ya existen al abrir el archivo
        #/ en modo lectura.
        self._ds.set_auto_mask(False)

        try:
            if dialecto is None:
                self.dialecto = Dialecto.detectar(self._ds)
            else:
                self.dialecto = Dialecto(dialecto)

            self._verificar_estructura()

            self.n_cuadros = self._ds.dimensions[DIMENSION_CUADRO].size
            self.n_atomos = self._ds.dimensions['atom'].size

            #/ Metadatos baratos, leídos al abrir. Que sean atributos comunes
            #/ y no propiedades es deliberado: en esta clase, atributo común
            #/ significa lectura ansiosa y propiedad significa lectura
            #/ perezosa o valor derivado.
            self.tiempos = self._leer_variable('tiempos')
            self.tipos = self._leer_cuadro('tipos', 0)
            self.celda = self._construir_celda()

            self.atributos = {a: getattr(self._ds, a)
                              for a in self._ds.ncattrs()}

            self.dimensiones = {d.name: (None if d.isunlimited() else d.size)
                                for d in self._ds.dimensions.values()}

            self._mascaras = {}

            if verificar:
                self._verificar_invariantes()

        except Exception:
            self._ds.close()
            raise

    #/ ---- administración de contexto ------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cerrar()
        return False

    def cerrar(self):
        """
        Cierre del archivo netCDF.

        Se llama automáticamente al salir del bloque ``with``. Es idempotente.

        Returns
        -------
        None
        """
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    #/ ---- verificaciones -------------------------------------------------

    def _verificar_estructura(self):
        """
        Comprobación de que el archivo tiene la estructura mínima esperada.

        Returns
        -------
        None

        Raises
        ------
        ErrorTrayectoria
            Si falta alguna dimensión o variable indispensable.
        """
        for dimension in (DIMENSION_CUADRO, 'atom'):
            if dimension not in self._ds.dimensions:
                raise ErrorTrayectoria(
                    f'Falta la dimensión {dimension!r} en {self.archivo}. '
                    f'Presentes: {sorted(self._ds.dimensions)}')

        for canonico in ('posiciones', 'tipos', 'tiempos', 'longitudes_celda'):
            if not self.disponible(canonico):
                nombre = self.dialecto.nombre_en_archivo(canonico)
                raise ErrorTrayectoria(
                    f'Falta la variable {canonico!r} ({nombre!r}) en '
                    f'{self.archivo}')

    def _verificar_invariantes(self):
        """
        Comprobación de las propiedades que el resto de la biblioteca asume.

        Se verifican contra el primer y el último cuadro, que es suficiente
        para detectar los casos patológicos y cuesta la lectura de dos
        cuadros.

        El orden de partículas constante y creciente no está garantizado por
        el formato. El estilo de volcado ``netcdf`` de LAMMPS escribe las
        partículas ordenadas por identificador, pero el estilo
        ``netcdf/mpiio`` las escribe en orden de rango MPI. Ese orden no sólo
        no es creciente: cambia entre cuadros, porque las partículas migran
        entre procesadores a medida que se difunden.

        Sin esta propiedad, la máscara de tipos calculada sobre el primer
        cuadro no sería aplicable al resto, y el error no se manifestaría más
        que como resultados sin sentido.

        Returns
        -------
        None

        Raises
        ------
        ErrorTrayectoria
            Si la trayectoria no tiene cuadros, si los tipos cambian entre
            cuadros, o si el orden de las partículas no es creciente o no es
            constante.
        """
        if self.n_cuadros < 1:
            raise ErrorTrayectoria(f'Trayectoria sin cuadros: {self.archivo}')

        ultimo = self.n_cuadros - 1

        #/ Constancia de los tipos
        if ultimo > 0:
            tipos_ultimo = self._leer_cuadro('tipos', ultimo)
            if not np.array_equal(self.tipos, tipos_ultimo):
                raise ErrorTrayectoria(
                    'Los tipos de partícula cambian entre el primer y el '
                    'último cuadro. La máscara de tipos calculada sobre el '
                    'primer cuadro no sería válida.')

        #/ Orden de partículas
        if self.disponible('identificadores'):
            ids_primero = self._leer_cuadro('identificadores', 0)

            if not np.all(np.diff(ids_primero) > 0):
                raise ErrorTrayectoria(
                    'Los identificadores de partícula no están ordenados de '
                    'forma creciente en el primer cuadro. El estilo de '
                    'volcado netcdf de LAMMPS ordena por identificador; el '
                    'estilo netcdf/mpiio escribe en orden de rango MPI. '
                    'Regenerar la trayectoria con el estilo netcdf.')

            if ultimo > 0:
                ids_ultimo = self._leer_cuadro('identificadores', ultimo)
                if not np.array_equal(ids_primero, ids_ultimo):
                    raise ErrorTrayectoria(
                        'El orden de las partículas cambia entre el primer y '
                        'el último cuadro. Esto ocurre con el estilo de '
                        'volcado netcdf/mpiio, que escribe en orden de rango '
                        'MPI: las partículas migran entre procesadores '
                        'durante la simulación. Regenerar la trayectoria con '
                        'el estilo netcdf, que ordena por identificador.')

    #/ ---- celda ----------------------------------------------------------

    def _construir_celda(self):
        """
        Construcción de la celda de simulación.

        Se verifica la constancia de la celda y se aplica la corrección desde
        el archivo de configuración si corresponde.

        La corrección se hace acá, en el constructor, para que no exista
        ningún instante en que un objeto Trayectoria sea observable con la
        celda corrupta. El orden 'corregir antes de detectar el eje del tubo'
        deja así de depender de la disciplina de quien use la biblioteca y
        pasa a ser una garantía del tipo.

        Returns
        -------
        Celda
            Celda de simulación, corregida si se pasó una configuración.

        Raises
        ------
        ErrorTrayectoria
            Si la celda varía entre cuadros, o si es degenerada y no se pasó
            una configuración.
        """
        longitudes = self._leer_variable('longitudes_celda')
        origenes = self._leer_variable('origen_celda')

        if self.disponible('angulos_celda'):
            angulos = self._leer_variable('angulos_celda')
        else:
            angulos = None

        #/ Constancia de la celda: se asume NVT
        for nombre, datos in (('longitudes', longitudes),
                              ('origen', origenes)):
            if not np.allclose(datos, datos[0], rtol=TOLERANCIA_CELDA):
                raise ErrorTrayectoria(
                    f'La celda de simulación no es constante ({nombre} varía '
                    f'entre cuadros). Esta biblioteca asume simulaciones NVT.')

        celda = Celda(
            longitudes[0],
            origenes[0],
            None if angulos is None else angulos[0])

        if self.configuracion is None:
            if celda.degenerada:
                raise ErrorTrayectoria(
                    f'La celda de {self.archivo} tiene longitudes nulas '
                    f'({celda.longitudes}). Esto ocurre en trayectorias '
                    f'escritas con kspace_modify slab. Se requiere un archivo '
                    f'de configuración para obtener las longitudes '
                    f'verdaderas.')

            return celda

        #/ Las longitudes de la configuración prevalecen. Se conserva el
        #/ origen de la trayectoria, que es el comportamiento de los scripts
        #/ originales; el origen de la configuración queda accesible en
        #/ trayectoria.configuracion.celda.origen
        return Celda(
            self.configuracion.celda.longitudes,
            celda.origen,
            celda.angulos)

    #/ ---- lectura --------------------------------------------------------

    def disponible(self, canonico):
        """
        Presencia de una variable canónica en el archivo.

        No todas las trayectorias traen fuerzas o imágenes periódicas.

        Parameters
        ----------
        canonico : str
            Nombre canónico de la variable.

        Returns
        -------
        bool
            Verdadero si la variable está presente.

        Raises
        ------
        ErrorTrayectoria
            Si el nombre canónico no pertenece al dialecto.
        """
        nombre = self.dialecto.nombre_en_archivo(canonico)

        if isinstance(nombre, tuple):
            return all(n in self._ds.variables for n in nombre)

        return nombre in self._ds.variables

    def _variable(self, canonico):
        """
        Obtención de la variable netCDF de un nombre canónico.

        Parameters
        ----------
        canonico : str
            Nombre canónico de la variable. No puede ser una variable
            almacenada en componentes escalares separadas.

        Returns
        -------
        netCDF4.Variable
            Variable del archivo.

        Raises
        ------
        ErrorTrayectoria
            Si la variable no está presente en el archivo.
        """
        if not self.disponible(canonico):
            nombre = self.dialecto.nombre_en_archivo(canonico)
            raise ErrorTrayectoria(
                f'La variable {canonico!r} ({nombre!r}) no está presente en '
                f'{self.archivo}. Variables disponibles: '
                f'{sorted(self.variables_disponibles)}')

        return self._ds.variables[self.dialecto.nombre_en_archivo(canonico)]

    @property
    def variables_disponibles(self):
        """
        list of str: Nombres canónicos de las variables presentes en
        el archivo.

        Se calcula en cada acceso consultando el archivo.
        """
        return [c for c in self.dialecto.variables if self.disponible(c)]

    def _leer_variable(self, canonico):
        """
        Lectura completa de una variable liviana.

        Pensada para tiempos y celda, no para variables por partícula.

        Parameters
        ----------
        canonico : str
            Nombre canónico de la variable.

        Returns
        -------
        numpy.ndarray
            Datos de la variable para todos los cuadros.
        """
        return self._variable(canonico)[:]

    def _leer_cuadro(self, canonico, cuadro):
        """
        Lectura de un único cuadro de una variable.

        Parameters
        ----------
        canonico : str
            Nombre canónico de la variable.
        cuadro : int
            Índice del cuadro.

        Returns
        -------
        numpy.ndarray
            Datos de la variable en ese cuadro.
        """
        return self._variable(canonico)[cuadro]

    def _leer_bloque(self, canonico, inicio, fin):
        """
        Lectura de un rango de cuadros de una variable.

        Las variables que el archivo almacena en componentes escalares
        separadas, como las imágenes periódicas, se ensamblan acá en un único
        arreglo vectorial.

        Parameters
        ----------
        canonico : str
            Nombre canónico de la variable.
        inicio : int
            Índice del primer cuadro a leer.
        fin : int
            Índice siguiente al del último cuadro a leer.

        Returns
        -------
        numpy.ndarray
            Datos de la variable para los cuadros pedidos, con la dimensión de
            cuadros primera.

        Raises
        ------
        ErrorTrayectoria
            Si la variable no está presente en el archivo.
        """
        nombre = self.dialecto.nombre_en_archivo(canonico)

        if isinstance(nombre, tuple):
            if not self.disponible(canonico):
                raise ErrorTrayectoria(
                    f'La variable {canonico!r} ({nombre!r}) no está presente '
                    f'en {self.archivo}')

            componentes = [self._ds.variables[n][inicio:fin] for n in nombre]

            return np.stack(componentes, axis=-1)

        return self._variable(canonico)[inicio:fin]

    #/ ---- selección de partículas ----------------------------------------

    def mascara_tipo(self, id_tipo):
        """
        Máscara booleana sobre las partículas de un tipo.

        Parameters
        ----------
        id_tipo : int
            Identificador del tipo de partícula.

        Returns
        -------
        numpy.ndarray
            Máscara booleana con forma ``(n_atomos,)``.

        Raises
        ------
        ErrorTrayectoria
            Si el tipo no está presente en la trayectoria.

        See Also
        --------
        mascara_tipos : Máscara sobre un conjunto de tipos.
        """
        return self.mascara_tipos((id_tipo,))

    def mascara_tipos(self, ids_tipo):
        """
        Máscara booleana sobre las partículas de un conjunto de tipos.

        Se calcula una sola vez por conjunto y queda cacheada. Los tipos se
        leen del primer cuadro, cuya validez para toda la trayectoria se
        verifica al construir el objeto.

        Sirve para referirse a la estructura del sistema sin cablear qué tipo
        es el carbono: en los sistemas con slab la estructura son los tipos 6
        y 7, y en los periódicos sólo el 6.

        Parameters
        ----------
        ids_tipo : iterable of int
            Identificadores de los tipos de partícula.

        Returns
        -------
        numpy.ndarray
            Máscara booleana con forma ``(n_atomos,)``, verdadera en las
            partículas de alguno de los tipos pedidos.

        Raises
        ------
        ErrorTrayectoria
            Si alguno de los tipos no está presente en la trayectoria.

        Examples
        --------
        >>> estructura = tr.mascara_tipos((6, 7))
        >>> posiciones_estructura = bloque.posiciones[:, estructura]
        """
        clave = tuple(sorted(ids_tipo))

        if clave in self._mascaras:
            return self._mascaras[clave]

        presentes = set(np.unique(self.tipos).tolist())
        ausentes = set(clave) - presentes

        if ausentes:
            raise ErrorTrayectoria(
                f'Tipos no presentes en la trayectoria: {sorted(ausentes)}. '
                f'Presentes: {sorted(presentes)}')

        mascara = np.isin(self.tipos, clave)
        self._mascaras[clave] = mascara

        return mascara

    @property
    def tipos_presentes(self):
        """
        dict: Cantidad de partículas por tipo presente en la trayectoria.

        Las claves son los identificadores de tipo y los valores la cantidad
        de partículas de cada uno. Se calcula en cada acceso a partir de
        :attr:`tipos`.
        """
        valores, conteos = np.unique(self.tipos, return_counts=True)

        return dict(zip(valores.tolist(), conteos.tolist()))

    #/ ---- iteración ------------------------------------------------------

    def bloques(self, tamanio=TAMANIO_BLOQUE_PREDETERMINADO, inicio=0,
            fin=None):
        """
        Generación de bloques de cuadros consecutivos.

        Es la forma prevista de recorrer los datos pesados de la trayectoria.

        Parameters
        ----------
        tamanio : int, optional
            Cantidad de cuadros por bloque. Por defecto
            :data:`TAMANIO_BLOQUE_PREDETERMINADO`.
        inicio : int, optional
            Índice del primer cuadro a recorrer. Por defecto 0.
        fin : int, optional
            Índice siguiente al del último cuadro a recorrer. Si es None se
            recorre hasta el final de la trayectoria.

        Yields
        ------
        Bloque
            Ventana de a lo sumo ``tamanio`` cuadros consecutivos. El último
            bloque puede ser más corto.

        Raises
        ------
        ErrorTrayectoria
            Si el tamaño de bloque es menor que uno, o si el rango de cuadros
            es inválido.

        Notes
        -----
        El tamaño de bloque regula el compromiso entre memoria y velocidad:
        bloques grandes aprovechan mejor las operaciones vectorizadas de numpy
        pero elevan el pico de memoria, que es aproximadamente
        ``tamanio * n_atomos * 3 * 4`` bytes por cada variable pesada que se
        pida al bloque. Para dimensionarlo con datos y no a ojo, ver
        :meth:`memoria_por_cuadro`.

        Examples
        --------
        >>> for bloque in tr.bloques(tamanio=50):
        ...     posiciones = bloque.posiciones
        """
        if tamanio < 1:
            raise ErrorTrayectoria(
                f'Tamaño de bloque inválido: {tamanio}')

        if fin is None:
            fin = self.n_cuadros

        if not 0 <= inicio < fin <= self.n_cuadros:
            raise ErrorTrayectoria(
                f'Rango de cuadros inválido: [{inicio}, {fin}) sobre una '
                f'trayectoria de {self.n_cuadros} cuadros')

        for principio in range(inicio, fin, tamanio):
            yield Bloque(self, principio, min(principio + tamanio, fin))

    def cuadro(self, indice):
        """
        Obtención de un bloque de un único cuadro.

        Parameters
        ----------
        indice : int
            Índice del cuadro.

        Returns
        -------
        Bloque
            Bloque de longitud uno. Sus arreglos conservan la dimensión de
            cuadros, de modo que las posiciones tienen forma
            ``(1, n_atomos, 3)``.

        Raises
        ------
        ErrorTrayectoria
            Si el índice está fuera de rango.
        """
        if not 0 <= indice < self.n_cuadros:
            raise ErrorTrayectoria(
                f'Cuadro fuera de rango: {indice} '
                f'(la trayectoria tiene {self.n_cuadros})')

        return Bloque(self, indice, indice + 1)

    #/ ---- dimensionamiento ------------------------------------------------

    def memoria_por_cuadro(self):
        """
        Costo en memoria de cada variable pesada, por cuadro.

        Sirve para elegir el tamaño de bloque con datos en lugar de a ojo: el
        pico de memoria de un bloque es aproximadamente el tamaño del bloque
        por la suma de las variables que se le pidan.

        Returns
        -------
        dict
            Bytes por cuadro de cada variable disponible que dependa del
            tiempo. Las claves son nombres canónicos.

        Examples
        --------
        >>> costo = tr.memoria_por_cuadro()
        >>> costo['posiciones'] / 1024**2
        1.157
        """
        tamanios = {}

        for canonico in self.variables_disponibles:
            nombre = self.dialecto.nombre_en_archivo(canonico)
            nombres = nombre if isinstance(nombre, tuple) else (nombre,)

            total = 0
            for n in nombres:
                variable = self._ds.variables[n]

                if DIMENSION_CUADRO not in variable.dimensions:
                    continue

                elementos = int(np.prod([
                    self._ds.dimensions[d].size
                    for d in variable.dimensions[1:]])) if \
                    len(variable.dimensions) > 1 else 1

                total += elementos * variable.dtype.itemsize

            if total:
                tamanios[canonico] = total

        return tamanios

    def __repr__(self):
        return (f'Trayectoria({os.path.basename(self.archivo)!r}, '
                f'dialecto={self.dialecto.nombre!r}, '
                f'n_cuadros={self.n_cuadros}, n_atomos={self.n_atomos})')


#/ ---- FIN MÓDULO --------------------------------------------------------
