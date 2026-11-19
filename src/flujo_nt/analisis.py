#! /usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Análisis de trayectorias de flujo a través de nanotubos.

Cada análisis es un acumulador que recorre la trayectoria por bloques de
cuadros: se prepara una vez, suma lo que corresponde a cada bloque, y al
final entrega su resultado. Como en ningún análisis el resultado de un cuadro
depende de otro, y lo que se acumula entre bloques son conteos y sumas, el
resultado es idéntico al de procesar la trayectoria completa en memoria.

La función :func:`recorrer` alimenta varios análisis con una sola pasada
sobre la trayectoria. Cada bloque se lee una vez y lo comparten todos; como el
bloque guarda lo que leyó, diez análisis sobre las mismas posiciones no leen
el disco diez veces.

Clases
------
Resultado
    Columnas de un resultado, con sus unidades y metadatos.
Analisis
    Clase base de los análisis.
Flujo
    Serie temporal de conteo y flujo numérico y másico en el tubo.
DensidadRadial
    Densidad numérica en función de la distancia al eje del tubo.
PerfilVelocidad
    Velocidad longitudinal en función de la distancia al eje del tubo.
DensidadLongitudinal
    Densidad numérica a lo largo del eje, como la calcula LAMMPS.
Pervaporacion
    Conteo de partículas que atravesaron el tubo.

Examples
--------
>>> agua = buscar_especie('Oa')
>>> with Trayectoria('sistema.nc', configuracion=configuracion) as tr:
...     tubo = Tubo.desde_trayectoria(tr)
...     flujo, densidad = recorrer(tr, [Flujo(agua, tubo),
...                                     DensidadRadial(agua, tubo)])
>>> flujo['Jn']

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: noviembre de 2026
"""

import abc

import numpy as np

from .geometria import MarcoTubo
from .trayectoria import TAMANIO_BLOQUE_PREDETERMINADO


#/ ---- CONSTANTES DEL MÓDULO --------------------------------------------

FRACCION_REGION_PREDETERMINADA = 0.9
"""float: Fracción del largo del tubo sobre la que se analiza, para excluir
la entrada y la salida."""

FRACCION_RADIO_PREDETERMINADA = 1.0
"""float: Fracción del radio del tubo sobre la que se analiza."""

FRACCION_BIN_R_PREDETERMINADA = 0.25
"""float: Ancho de los bins radiales, en unidades de sigma de referencia."""

SIGMA_REF_PREDETERMINADO = 3.5
"""float: Sigma de referencia del campo de fuerzas, en Å."""

FACTOR_RADIO_CILINDRO = 1.5
"""float: Radio del cilindro de la densidad longitudinal en el tubo, en
unidades del radio del tubo. Es el valor del input de LAMMPS."""

NUMERO_BINS_LONGITUDINAL = 32
"""int: Cantidad de bins sobre el largo del tubo en la densidad
longitudinal. El ancho de bin resultante se usa también en los
reservorios."""

TOLERANCIA_BINS = 1e-9
"""float: Tolerancia al contar cuántos bins caben en una región, para que un
cociente entero afectado por el redondeo no agregue un bin."""


#/ ---- EXCEPCIONES -------------------------------------------------------

class ErrorAnalisis(Exception):
    """
    Error en la configuración o el uso de un análisis.
    """


#/ ---- RESULTADO --------------------------------------------------------

class Resultado:
    """
    Resultado de un análisis: columnas con nombre, unidades y metadatos.

    No sabe escribirse a disco: la escritura es un paso aparte, de modo que
    el mismo resultado sirve para guardar un archivo o para graficar
    directamente.

    Parameters
    ----------
    calculo : str
        Identificador del cálculo, usado para nombrar archivos de salida.
    especie : Especie
        Especie sobre la que se calculó.
    columnas : dict
        Columnas del resultado, en orden: nombre a arreglo. Las columnas son
        unidimensionales, salvo en los resultados por cuadro, donde las
        magnitudes por bin tienen forma ``(n_cuadros, n_bins)``.
    unidades : dict
        Unidades de cada columna, por nombre.
    metadatos : dict, optional
        Parámetros y valores derivados del cálculo, para el encabezado de
        los archivos y para la trazabilidad.
    formatos : dict, optional
        Formato de escritura de las columnas que no usan el predeterminado,
        por nombre; por ejemplo ``{'paso': '%d'}``.

    Attributes
    ----------
    calculo : str
        Identificador del cálculo.
    especie : Especie
        Especie sobre la que se calculó.
    columnas : dict
        Columnas del resultado.
    unidades : dict
        Unidades de cada columna.
    metadatos : dict
        Parámetros y valores derivados del cálculo.
    formatos : dict
        Formatos de escritura particulares de algunas columnas.

    Examples
    --------
    >>> resultado['Jn']
    """

    def __init__(self, calculo, especie, columnas, unidades, metadatos=None,
            formatos=None):
        self.calculo = calculo
        self.especie = especie
        self.columnas = dict(columnas)
        self.unidades = dict(unidades)
        self.metadatos = dict(metadatos or {})
        self.formatos = dict(formatos or {})

    def __getitem__(self, nombre):
        return self.columnas[nombre]

    @property
    def nombres(self):
        """list of str: Nombres de las columnas, en orden."""
        return list(self.columnas)

    def __repr__(self):
        return (f'Resultado({self.calculo!r}, {self.especie.codigo!r}, '
                f'columnas={self.nombres})')


#/ ---- FUNCIONES AUXILIARES ----------------------------------------------

def _conteo_por_cuadro(indices, n_bins, pesos=None):
    """
    Histograma por cuadro a partir de índices de bin.

    Se hace con un único ``numpy.bincount`` desplazando los índices de cada
    cuadro, en lugar de recorrer los cuadros.

    Parameters
    ----------
    indices : numpy.ndarray
        Índice de bin de cada partícula, con forma ``(n_cuadros, n)``. El
        valor ``n_bins`` marca a las partículas fuera de todo bin.
    n_bins : int
        Cantidad de bins.
    pesos : numpy.ndarray, optional
        Peso de cada partícula, con la forma de ``indices``. Si es None se
        cuentan partículas.

    Returns
    -------
    numpy.ndarray
        Suma de pesos (o conteo) por cuadro y bin, con forma
        ``(n_cuadros, n_bins)``.
    """
    n_cuadros = indices.shape[0]
    desplazados = indices + (n_bins + 1) * np.arange(n_cuadros)[:, None]

    if pesos is not None:
        pesos = np.asarray(pesos, dtype='f8').ravel()

    suma = np.bincount(desplazados.ravel(), weights=pesos,
                       minlength=n_cuadros * (n_bins + 1))

    return suma.reshape(n_cuadros, n_bins + 1)[:, :n_bins]


def _media_y_desvio(suma, suma_cuadrados, cantidad):
    """
    Media y desvío a partir de sumas acumuladas.

    El desvío es el poblacional, como el de ``numpy.std``. Donde la
    cantidad es nula, ambos son NaN.

    Parameters
    ----------
    suma, suma_cuadrados : numpy.ndarray
        Suma de los valores y de sus cuadrados.
    cantidad : numpy.ndarray or int
        Cantidad de valores sumados.

    Returns
    -------
    media, desvio : numpy.ndarray
        Media y desvío poblacional.
    """
    cantidad = np.broadcast_to(np.asarray(cantidad, dtype='f8'),
                               np.shape(suma))
    hay = cantidad > 0

    media = np.full(np.shape(suma), np.nan)
    np.divide(suma, cantidad, out=media, where=hay)

    momento = np.full(np.shape(suma), np.nan)
    np.divide(suma_cuadrados, cantidad, out=momento, where=hay)

    #/ El redondeo puede dar una varianza levemente negativa cuando todos los
    #/ valores son iguales
    varianza = np.clip(momento - media**2, 0.0, None)

    return media, np.sqrt(varianza)


def _cantidad_de_bins(largo, ancho):
    """
    Cantidad de bins de un ancho dado necesaria para cubrir un largo.

    El último bin puede extenderse más allá del largo, como en LAMMPS.
    """
    return int(np.ceil(largo / ancho - TOLERANCIA_BINS))


#/ ---- CLASE BASE -------------------------------------------------------

class Analisis(abc.ABC):
    """
    Clase base de los análisis.

    Define el ciclo de vida común: :meth:`iniciar` prepara el análisis para
    una trayectoria, :meth:`procesar` acumula un bloque de cuadros, y
    :meth:`resultado` entrega lo acumulado. Las subclases sólo implementan
    qué preparar, qué acumular de cada bloque y cómo armar el resultado.

    Un análisis puede reutilizarse: :meth:`iniciar` descarta lo acumulado.

    Parameters
    ----------
    especie : Especie
        Especie de partículas sobre la que se calcula.

    Attributes
    ----------
    especie : Especie
        Especie de partículas sobre la que se calcula.
    calculo : str
        Identificador del cálculo, usado para nombrar archivos de salida.
    """

    calculo = None

    def __init__(self, especie):
        self.especie = especie
        self._mascara = None

    def iniciar(self, trayectoria):
        """
        Preparación del análisis para recorrer una trayectoria.

        Parameters
        ----------
        trayectoria : Trayectoria
            Trayectoria abierta que se va a recorrer.

        Returns
        -------
        None

        Raises
        ------
        ErrorTrayectoria
            Si el tipo de la especie no está presente en la trayectoria.
        """
        self._mascara = trayectoria.mascara_tipo(self.especie.tipo)
        self._n_cuadros = 0
        self._tiempos = []
        self._preparar(trayectoria)

    def procesar(self, bloque):
        """
        Acumulación de un bloque de cuadros.

        Parameters
        ----------
        bloque : Bloque
            Bloque de la trayectoria con que se inició el análisis.

        Returns
        -------
        None

        Raises
        ------
        ErrorAnalisis
            Si el análisis no fue iniciado.
        """
        if self._mascara is None:
            raise ErrorAnalisis(
                f'{type(self).__name__} debe iniciarse con una trayectoria '
                f'antes de procesar bloques.')

        self._n_cuadros += len(bloque)
        self._tiempos.append(bloque.tiempos)
        self._acumular(bloque)

    @abc.abstractmethod
    def resultado(self):
        """
        Resultado de lo acumulado.

        Returns
        -------
        Resultado
            Resultado del análisis.

        Raises
        ------
        ErrorAnalisis
            Si no se procesó ningún cuadro.
        """

    def _preparar(self, trayectoria):
        """
        Preparación propia de cada análisis. Por defecto no hace nada.
        """

    @abc.abstractmethod
    def _acumular(self, bloque):
        """
        Acumulación propia de cada análisis para un bloque.
        """

    def _seleccion(self, arreglo):
        """
        Partículas de la especie en un arreglo de un bloque.
        """
        return arreglo[:, self._mascara]

    def _verificar_datos(self):
        """
        Comprobación de que se procesó al menos un cuadro.
        """
        if self._mascara is None or self._n_cuadros == 0:
            raise ErrorAnalisis(
                f'{type(self).__name__} no procesó ningún cuadro.')

    @property
    def tiempos(self):
        """
        numpy.ndarray: Tiempos de los cuadros procesados, en fs.
        """
        return np.concatenate(self._tiempos)

    def __repr__(self):
        return f'{type(self).__name__}({self.especie.codigo!r})'


#/ ---- FLUJO ------------------------------------------------------------

class Flujo(Analisis):
    """
    Conteo y flujo numérico y másico de una especie en el tubo.

    Para cada cuadro se cuentan las partículas dentro de una región
    cilíndrica coaxial con el eje curvo del tubo, y se suman sus velocidades
    longitudinales. El flujo numérico es esa suma dividida por el volumen de
    la región; el másico, el numérico por la masa de la molécula.

    Parameters
    ----------
    especie : Especie
        Especie de partículas. Debe tener masa molecular definida.
    tubo : Tubo
        Tubo medido.
    fraccion_region : float, optional
        Fracción del largo del tubo que abarca la región. Por defecto 0.9.
    fraccion_radio : float, optional
        Fracción del radio del tubo que abarca la región. Por defecto 1.

    Raises
    ------
    ErrorAnalisis
        Si la especie no tiene masa molecular.
    """

    calculo = 'flujo.inst'

    def __init__(self, especie, tubo,
            fraccion_region=FRACCION_REGION_PREDETERMINADA,
            fraccion_radio=FRACCION_RADIO_PREDETERMINADA):
        super().__init__(especie)

        if especie.masa_molecular is None:
            raise ErrorAnalisis(
                f'La especie {especie.codigo} no tiene masa molecular: no '
                f'representa a su molécula en el cálculo del flujo.')

        self.tubo = tubo
        self.fraccion_region = fraccion_region
        self.fraccion_radio = fraccion_radio

    def _preparar(self, trayectoria):
        self._radio_region = self.tubo.radio * self.fraccion_radio
        self._volumen = (np.pi * self._radio_region**2
                         * self.tubo.largo * self.fraccion_region)
        self._conteo = []
        self._suma_velocidades = []

    def _acumular(self, bloque):
        marco = self.tubo.marco
        posiciones = self.tubo.corregir(self._seleccion(bloque.posiciones))
        velocidades = marco.longitudinal(self._seleccion(bloque.velocidades))

        en_region = self.tubo.en_region(
            posiciones, self.fraccion_region, self.fraccion_radio)

        self._conteo.append(en_region.sum(axis=1))
        self._suma_velocidades.append(
            np.where(en_region, velocidades, 0.0).sum(axis=1, dtype='f8'))

    def resultado(self):
        """
        Serie temporal del conteo y de los flujos.

        Returns
        -------
        Resultado
            Columnas ``tiempo`` (fs), ``n`` (partículas en la región),
            ``Jn`` (flujo numérico, 1/(fs Å²)) y ``Jm`` (flujo másico,
            g/(mol fs Å²)), una fila por cuadro.

        Raises
        ------
        ErrorAnalisis
            Si no se procesó ningún cuadro.
        """
        self._verificar_datos()

        numerico = np.concatenate(self._suma_velocidades) / self._volumen

        return Resultado(
            self.calculo, self.especie,
            columnas={
                'tiempo': self.tiempos,
                'n': np.concatenate(self._conteo),
                'Jn': numerico,
                'Jm': numerico * self.especie.masa_molecular},
            unidades={
                'tiempo': 'fs', 'n': '-', 'Jn': '1 / fs Å^2',
                'Jm': 'g / mol fs Å^2'},
            metadatos={
                'limites_region': self.tubo.limites(self.fraccion_region),
                'radio_region': self._radio_region,
                'radio_tubo': self.tubo.radio,
                'fraccion_region': self.fraccion_region,
                'fraccion_radio': self.fraccion_radio,
                'volumen_region': self._volumen,
                'masa_molecular': self.especie.masa_molecular,
                'n_cuadros': self._n_cuadros})


#/ ---- ANÁLISIS RADIALES ------------------------------------------------

class _AnalisisRadial(Analisis):
    """
    Base común de los análisis en bins radiales dentro del tubo.

    Define los bins radiales y la asignación de cada partícula a un bin. Los
    bins tienen ancho fijo ``fraccion_bin_r * sigma_ref`` y cubren el radio
    de la región; el último puede pasarlo. Sólo se asignan a un bin las
    partículas dentro del largo de la región.

    Parameters
    ----------
    especie : Especie
        Especie de partículas.
    tubo : Tubo
        Tubo medido.
    fraccion_region : float
        Fracción del largo del tubo que abarca la región.
    fraccion_radio : float
        Fracción del radio del tubo que cubren los bins.
    fraccion_bin_r : float
        Ancho de bin en unidades de ``sigma_ref``.
    sigma_ref : float
        Sigma de referencia, en Å.
    """

    def __init__(self, especie, tubo, fraccion_region, fraccion_radio,
            fraccion_bin_r, sigma_ref):
        super().__init__(especie)
        self.tubo = tubo
        self.fraccion_region = fraccion_region
        self.fraccion_radio = fraccion_radio
        self.fraccion_bin_r = fraccion_bin_r
        self.sigma_ref = sigma_ref

    @property
    def ancho_bin(self):
        """float: Ancho de los bins radiales, en Å."""
        return self.fraccion_bin_r * self.sigma_ref

    def _preparar(self, trayectoria):
        radio_region = self.tubo.radio * self.fraccion_radio
        self._n_bins = int(np.ceil(radio_region / self.ancho_bin))
        self.bordes = self.ancho_bin * np.arange(self._n_bins + 1)
        self._largo_region = self.tubo.largo * self.fraccion_region

    @property
    def centros_bins(self):
        """numpy.ndarray: Punto medio de cada bin radial, en Å."""
        return (self.bordes[:-1] + self.bordes[1:]) / 2

    def _indices(self, posiciones_corregidas):
        """
        Bin radial de cada partícula, o ``n_bins`` si está fuera.
        """
        marco = self.tubo.marco
        indices = np.digitize(marco.radial(posiciones_corregidas),
                              self.bordes) - 1

        inferior, superior = self.tubo.limites(self.fraccion_region)
        longitudinal = marco.longitudinal(posiciones_corregidas)
        en_largo = (longitudinal >= inferior) & (longitudinal <= superior)

        return np.where(en_largo & (indices < self._n_bins), indices,
                        self._n_bins)

    def _metadatos(self):
        """Parámetros comunes de los análisis radiales."""
        return {
            'limites_region': self.tubo.limites(self.fraccion_region),
            'radio_tubo': self.tubo.radio,
            'fraccion_region': self.fraccion_region,
            'fraccion_radio': self.fraccion_radio,
            'fraccion_bin_r': self.fraccion_bin_r,
            'sigma_ref': self.sigma_ref,
            'ancho_bin': self.ancho_bin,
            'bordes_bins': self.bordes,
            'n_cuadros': self._n_cuadros}


class DensidadRadial(_AnalisisRadial):
    """
    Densidad numérica de una especie en función de la distancia al eje.

    En cada cuadro se cuentan las partículas de cada cáscara cilíndrica
    coaxial con el eje curvo del tubo. La densidad es el conteo dividido por
    el volumen de la cáscara; se informan su media y su desvío entre
    cuadros.

    Parameters
    ----------
    especie : Especie
        Especie de partículas.
    tubo : Tubo
        Tubo medido.
    fraccion_region : float, optional
        Fracción del largo del tubo. Por defecto 0.9.
    fraccion_radio : float, optional
        Fracción del radio del tubo que cubren los bins. Por defecto 1.
    fraccion_bin_r : float, optional
        Ancho de bin en unidades de ``sigma_ref``. Por defecto 0.25.
    sigma_ref : float, optional
        Sigma de referencia, en Å. Por defecto 3.5.
    """

    calculo = 'dens.num.radial.promedio'

    def __init__(self, especie, tubo,
            fraccion_region=FRACCION_REGION_PREDETERMINADA,
            fraccion_radio=FRACCION_RADIO_PREDETERMINADA,
            fraccion_bin_r=FRACCION_BIN_R_PREDETERMINADA,
            sigma_ref=SIGMA_REF_PREDETERMINADO):
        super().__init__(especie, tubo, fraccion_region, fraccion_radio,
                         fraccion_bin_r, sigma_ref)

    def _preparar(self, trayectoria):
        super()._preparar(trayectoria)
        self._suma = np.zeros(self._n_bins)
        self._suma_cuadrados = np.zeros(self._n_bins)

    def _acumular(self, bloque):
        posiciones = self.tubo.corregir(self._seleccion(bloque.posiciones))
        conteo = _conteo_por_cuadro(self._indices(posiciones), self._n_bins)

        self._suma += conteo.sum(axis=0)
        self._suma_cuadrados += (conteo**2).sum(axis=0)

    def resultado(self):
        """
        Densidad radial media y su desvío entre cuadros.

        Returns
        -------
        Resultado
            Columnas ``centro_bin`` (Å), ``n`` (conteo total sobre todos los
            cuadros), ``rho_n_media`` y ``rho_n_desvio`` (n/Å³), una fila
            por bin radial.

        Raises
        ------
        ErrorAnalisis
            Si no se procesó ningún cuadro.
        """
        self._verificar_datos()

        media, desvio = _media_y_desvio(
            self._suma, self._suma_cuadrados, self._n_cuadros)
        volumenes = self._largo_region * np.pi * np.diff(self.bordes**2)

        return Resultado(
            self.calculo, self.especie,
            columnas={
                'centro_bin': self.centros_bins,
                'n': self._suma,
                'rho_n_media': media / volumenes,
                'rho_n_desvio': desvio / volumenes},
            unidades={
                'centro_bin': 'Å', 'n': '-', 'rho_n_media': 'n / Å^3',
                'rho_n_desvio': 'n / Å^3'},
            metadatos=self._metadatos())


class PerfilVelocidad(_AnalisisRadial):
    """
    Velocidad longitudinal de una especie en función de la distancia al eje.

    Por defecto se acumulan, para cada bin radial, el conteo de muestras y
    la suma de velocidades y de sus cuadrados sobre todos los cuadros; el
    resultado es la media y el desvío sobre todas las muestras del bin. A
    diferencia de :class:`DensidadRadial`, el desvío no es entre cuadros.

    Con ``por_cuadro=True`` el resultado es, en cambio, la velocidad media de
    cada bin en cada cuadro, para seguir la evolución del perfil.

    Parameters
    ----------
    especie : Especie
        Especie de partículas.
    tubo : Tubo
        Tubo medido.
    fraccion_region : float, optional
        Fracción del largo del tubo. Por defecto 0.9.
    fraccion_radio : float, optional
        Fracción del radio del tubo que cubren los bins. Por defecto 1.
    fraccion_bin_r : float, optional
        Ancho de bin en unidades de ``sigma_ref``. Por defecto 0.25.
    sigma_ref : float, optional
        Sigma de referencia, en Å. Por defecto 3.5.
    por_cuadro : bool, optional
        Si es verdadero, el resultado es un perfil por cuadro. Por defecto
        False.
    """

    calculo = 'perfil.velocidad.promedio'

    def __init__(self, especie, tubo,
            fraccion_region=FRACCION_REGION_PREDETERMINADA,
            fraccion_radio=FRACCION_RADIO_PREDETERMINADA,
            fraccion_bin_r=FRACCION_BIN_R_PREDETERMINADA,
            sigma_ref=SIGMA_REF_PREDETERMINADO,
            por_cuadro=False):
        super().__init__(especie, tubo, fraccion_region, fraccion_radio,
                         fraccion_bin_r, sigma_ref)
        self.por_cuadro = por_cuadro

        if por_cuadro:
            self.calculo = 'perfil.velocidad.serie'

    def _preparar(self, trayectoria):
        super()._preparar(trayectoria)
        self._conteo = np.zeros(self._n_bins)
        self._suma = np.zeros(self._n_bins)
        self._suma_cuadrados = np.zeros(self._n_bins)
        self._serie_conteo = []
        self._serie_suma = []

    def _acumular(self, bloque):
        posiciones = self.tubo.corregir(self._seleccion(bloque.posiciones))
        velocidades = self.tubo.marco.longitudinal(
            self._seleccion(bloque.velocidades)).astype('f8')
        indices = self._indices(posiciones)

        conteo = _conteo_por_cuadro(indices, self._n_bins)
        suma = _conteo_por_cuadro(indices, self._n_bins, velocidades)

        self._conteo += conteo.sum(axis=0)
        self._suma += suma.sum(axis=0)
        self._suma_cuadrados += _conteo_por_cuadro(
            indices, self._n_bins, velocidades**2).sum(axis=0)

        if self.por_cuadro:
            self._serie_conteo.append(conteo)
            self._serie_suma.append(suma)

    def resultado(self):
        """
        Perfil de velocidad longitudinal medio, o por cuadro.

        Returns
        -------
        Resultado
            Por defecto, columnas ``centro_bin`` (Å), ``n`` (muestras),
            ``v_media`` y ``v_desvio`` (Å/fs), una fila por bin radial.
            Con ``por_cuadro=True``, columnas ``tiempo`` (fs), ``n`` y
            ``v_media``, donde las dos últimas tienen forma
            ``(n_cuadros, n_bins)``; los centros de los bins están en los
            metadatos. Los bins sin muestras tienen velocidad NaN.

        Raises
        ------
        ErrorAnalisis
            Si no se procesó ningún cuadro.
        """
        self._verificar_datos()

        metadatos = self._metadatos()
        unidades = {'centro_bin': 'Å', 'tiempo': 'fs', 'n': '-',
                    'v_media': 'Å/fs', 'v_desvio': 'Å/fs'}

        if self.por_cuadro:
            conteo = np.concatenate(self._serie_conteo)
            media = np.full(conteo.shape, np.nan)
            np.divide(np.concatenate(self._serie_suma), conteo, out=media,
                      where=conteo > 0)
            metadatos['centros_bins'] = self.centros_bins

            columnas = {'tiempo': self.tiempos, 'n': conteo,
                        'v_media': media}
        else:
            media, desvio = _media_y_desvio(
                self._suma, self._suma_cuadrados, self._conteo)

            columnas = {'centro_bin': self.centros_bins, 'n': self._conteo,
                        'v_media': media, 'v_desvio': desvio}

        return Resultado(
            self.calculo, self.especie, columnas,
            {nombre: unidades[nombre] for nombre in columnas}, metadatos)


#/ ---- DENSIDAD LONGITUDINAL -------------------------------------------

class DensidadLongitudinal(Analisis):
    """
    Densidad numérica de una especie a lo largo del eje, como en LAMMPS.

    Reproduce el cálculo del input de LAMMPS de las simulaciones de
    referencia (comandos ``compute chunk/atom`` y ``fix ave/chunk``), como
    respaldo para simulaciones en las que no se registró. El sistema se
    divide en tres regiones, cada una calculada por una instancia distinta:

    - ``'izq'``: el reservorio entre el extremo inferior de la celda y
      ``-l/2``, en láminas que abarcan toda la sección de la celda;
    - ``'tubo'``: el tramo entre ``-l/2`` y ``+l/2``, dentro de un cilindro
      recto de radio ``factor_radio * r`` sobre el eje de coordenadas;
    - ``'der'``: el reservorio entre ``+l/2`` y el extremo superior.

    Los bins tienen ancho ``l / numero_bins`` y empiezan en el extremo
    inferior de cada región. Como en LAMMPS, el último bin de un reservorio
    puede pasar el límite de la región, y las partículas en ese sobrante se
    cuentan.

    Las coordenadas son absolutas, como en LAMMPS: se supone el tubo
    centrado en el origen, y no se corrige la curvatura.

    Parameters
    ----------
    especie : Especie
        Especie de partículas.
    region : {'izq', 'tubo', 'der'}
        Región del sistema.
    tubo : Tubo, optional
        Tubo medido. Da el marco de referencia y los valores por defecto de
        ``largo`` y ``radio``.
    largo : float, optional
        Largo del tubo ``l``, en Å. Por defecto el largo medido del tubo.
        Para reproducir una corrida de LAMMPS, el valor de ``v__l``.
    radio : float, optional
        Radio del tubo ``r``, en Å. Por defecto el radio medido del tubo.
        Para reproducir una corrida de LAMMPS, el valor de ``v__r``.
    factor_radio : float, optional
        Radio del cilindro en unidades de ``radio``. Por defecto 1.5.
    numero_bins : int, optional
        Cantidad de bins sobre el largo del tubo. Por defecto 32.
    por_cuadro : bool, optional
        Si es verdadero, el resultado es un perfil por cuadro. Por defecto
        False.

    Raises
    ------
    ErrorAnalisis
        Si la región no es válida, o si falta el tubo y no se dieron los
        valores que lo reemplazan.
    """

    REGIONES = ('izq', 'tubo', 'der')
    """tuple of str: Regiones del sistema."""

    def __init__(self, especie, region, tubo=None, largo=None, radio=None,
            factor_radio=FACTOR_RADIO_CILINDRO,
            numero_bins=NUMERO_BINS_LONGITUDINAL, por_cuadro=False):
        super().__init__(especie)

        if region not in self.REGIONES:
            raise ErrorAnalisis(
                f'Región inválida: {region!r}. Válidas: {self.REGIONES}')

        if tubo is None and largo is None:
            raise ErrorAnalisis('Se requiere el tubo o el largo.')

        if region == 'tubo' and tubo is None and radio is None:
            raise ErrorAnalisis('Se requiere el tubo o el radio.')

        self.region = region
        self.tubo = tubo
        self.largo = tubo.largo if largo is None else largo
        self.radio = (radio if radio is not None
                      else tubo.radio if tubo is not None else None)
        self.factor_radio = factor_radio
        self.numero_bins = numero_bins
        self.por_cuadro = por_cuadro

        sufijo = 'serie' if por_cuadro else 'promedio'
        self.calculo = f'dens.num.longitudinal.{region}.{sufijo}'

    @property
    def ancho_bin(self):
        """float: Ancho de los bins, en Å."""
        return self.largo / self.numero_bins

    def _preparar(self, trayectoria):
        celda = trayectoria.celda

        if self.tubo is None:
            self._marco = MarcoTubo.desde_celda(celda)
        else:
            self._marco = self.tubo.marco

        inferior, superior = celda.limites[self._marco.indice_longitudinal]
        semilargo = self.largo / 2

        self._origen, extremo = {
            'izq': (inferior, -semilargo),
            'tubo': (-semilargo, semilargo),
            'der': (semilargo, superior)}[self.region]

        self._n_bins = _cantidad_de_bins(extremo - self._origen,
                                         self.ancho_bin)

        if self.region == 'tubo':
            self._radio_cilindro = self.factor_radio * self.radio
            area = np.pi * self._radio_cilindro**2
        else:
            self._radio_cilindro = None
            transversales = list(self._marco.indices_transversales)
            area = np.prod(celda.longitudes[transversales])

        self._volumen_bin = self.ancho_bin * area
        self._suma = np.zeros(self._n_bins)
        self._suma_cuadrados = np.zeros(self._n_bins)
        self._serie = []

    def _acumular(self, bloque):
        posiciones = self._seleccion(bloque.posiciones)
        longitudinal = self._marco.longitudinal(posiciones)

        indices = np.floor(
            (longitudinal - self._origen) / self.ancho_bin).astype(int)
        fuera = (indices < 0) | (indices >= self._n_bins)

        if self._radio_cilindro is not None:
            fuera |= self._marco.radial(posiciones) > self._radio_cilindro

        conteo = _conteo_por_cuadro(np.where(fuera, self._n_bins, indices),
                                    self._n_bins)

        self._suma += conteo.sum(axis=0)
        self._suma_cuadrados += (conteo**2).sum(axis=0)

        if self.por_cuadro:
            self._serie.append(conteo)

    @property
    def centros_bins(self):
        """numpy.ndarray: Coordenada del punto medio de cada bin, en Å."""
        return self._origen + self.ancho_bin * (np.arange(self._n_bins)
                                                + 0.5)

    def resultado(self):
        """
        Densidad longitudinal media y su desvío entre cuadros, o por cuadro.

        Returns
        -------
        Resultado
            Por defecto, columnas ``centro_bin`` (Å), ``n_media`` (conteo
            medio por cuadro), ``rho_n_media`` y ``rho_n_desvio`` (n/Å³),
            una fila por bin. Con ``por_cuadro=True``, columnas ``tiempo``
            (fs), ``n`` y ``rho_n``, donde las dos últimas tienen forma
            ``(n_cuadros, n_bins)``; los centros de los bins están en los
            metadatos.

        Raises
        ------
        ErrorAnalisis
            Si no se procesó ningún cuadro.
        """
        self._verificar_datos()

        metadatos = {
            'region': self.region,
            'largo': self.largo,
            'radio': self.radio,
            'radio_cilindro': self._radio_cilindro,
            'numero_bins': self.numero_bins,
            'ancho_bin': self.ancho_bin,
            'origen': self._origen,
            'volumen_bin': self._volumen_bin,
            'n_cuadros': self._n_cuadros}

        if self.por_cuadro:
            conteo = np.concatenate(self._serie)
            metadatos['centros_bins'] = self.centros_bins
            columnas = {'tiempo': self.tiempos, 'n': conteo,
                        'rho_n': conteo / self._volumen_bin}
            unidades = {'tiempo': 'fs', 'n': '-', 'rho_n': 'n / Å^3'}
        else:
            media, desvio = _media_y_desvio(
                self._suma, self._suma_cuadrados, self._n_cuadros)
            columnas = {'centro_bin': self.centros_bins, 'n_media': media,
                        'rho_n_media': media / self._volumen_bin,
                        'rho_n_desvio': desvio / self._volumen_bin}
            unidades = {'centro_bin': 'Å', 'n_media': '-',
                        'rho_n_media': 'n / Å^3',
                        'rho_n_desvio': 'n / Å^3'}

        return Resultado(self.calculo, self.especie, columnas, unidades,
                         metadatos)


#/ ---- PERVAPORACIÓN ---------------------------------------------------

class Pervaporacion(Analisis):
    """
    Conteo de partículas de una especie que atravesaron el tubo.

    Herramienta de diagnóstico: cuenta, en cada cuadro, las partículas más
    allá de un extremo del tubo, es decir, en el reservorio de destino. Como
    en LAMMPS, la coordenada es absoluta y el umbral es ``l/2``.

    El resultado identifica cada cuadro por su número de paso de
    integración, como el archivo que escribe LAMMPS.

    Parameters
    ----------
    especie : Especie
        Especie de partículas.
    tubo : Tubo, optional
        Tubo medido. Da el marco de referencia y el largo por defecto.
    largo : float, optional
        Largo del tubo ``l``, en Å. Por defecto el largo medido del tubo.
    lado : {1, -1}, optional
        Reservorio de destino: 1 para el de coordenada positiva, -1 para el
        de coordenada negativa. Por defecto 1.
    paso_integracion : float, optional
        Paso de integración en fs, para numerar los cuadros. Por defecto el
        que registra la trayectoria; es obligatorio para trayectorias que no
        lo registran, como las reescritas por OVITO.

    Raises
    ------
    ErrorAnalisis
        Si el lado no es 1 o -1, si falta el tubo y no se dio el largo, o
        si no se conoce el paso de integración.
    """

    calculo = 'conteo.pervaporacion'

    def __init__(self, especie, tubo=None, largo=None, lado=1,
            paso_integracion=None):
        super().__init__(especie)

        if lado not in (1, -1):
            raise ErrorAnalisis(f'Lado inválido: {lado!r}. Válidos: 1, -1')

        if tubo is None and largo is None:
            raise ErrorAnalisis('Se requiere el tubo o el largo.')

        self.tubo = tubo
        self.largo = tubo.largo if largo is None else largo
        self.lado = lado
        self.paso_integracion = paso_integracion

    def _preparar(self, trayectoria):
        if self.tubo is None:
            self._marco = MarcoTubo.desde_celda(trayectoria.celda)
        else:
            self._marco = self.tubo.marco

        self._paso = (self.paso_integracion if self.paso_integracion
                      is not None else trayectoria.paso_integracion)

        if self._paso is None:
            raise ErrorAnalisis(
                'La trayectoria no registra el paso de integración (por '
                'ejemplo, si fue reescrita por OVITO). Indicarlo con '
                'paso_integracion.')

        self._conteo = []

    def _acumular(self, bloque):
        longitudinal = self._marco.longitudinal(
            self._seleccion(bloque.posiciones))

        self._conteo.append(
            (self.lado * longitudinal > self.largo / 2).sum(axis=1))

    def resultado(self):
        """
        Serie temporal del conteo de partículas pervaporadas.

        Returns
        -------
        Resultado
            Columnas ``paso`` (número de paso de integración) y ``n``
            (partículas en el reservorio de destino), una fila por cuadro.

        Raises
        ------
        ErrorAnalisis
            Si no se procesó ningún cuadro.
        """
        self._verificar_datos()

        pasos = np.rint(self.tiempos / self._paso).astype(int)

        return Resultado(
            self.calculo, self.especie,
            columnas={'paso': pasos, 'n': np.concatenate(self._conteo)},
            unidades={'paso': '-', 'n': '-'},
            metadatos={'umbral': self.lado * self.largo / 2,
                       'largo': self.largo, 'lado': self.lado,
                       'paso_integracion': self._paso,
                       'n_cuadros': self._n_cuadros},
            formatos={'paso': '%d'})


#/ ---- RECORRIDO --------------------------------------------------------

def recorrer(trayectoria, analisis,
        tamanio_bloque=TAMANIO_BLOQUE_PREDETERMINADO, inicio=0, fin=None):
    """
    Ejecución de varios análisis con una sola pasada sobre la trayectoria.

    Parameters
    ----------
    trayectoria : Trayectoria
        Trayectoria abierta.
    analisis : iterable of Analisis
        Análisis a ejecutar.
    tamanio_bloque : int, optional
        Cantidad de cuadros por bloque. Por defecto
        :data:`TAMANIO_BLOQUE_PREDETERMINADO`.
    inicio : int, optional
        Primer cuadro a procesar. Por defecto 0.
    fin : int, optional
        Cuadro siguiente al último a procesar. Por defecto el final.

    Returns
    -------
    list of Resultado
        Resultado de cada análisis, en el orden en que se dieron.

    Raises
    ------
    ErrorTrayectoria
        Si el rango de cuadros o el tamaño de bloque son inválidos.
    """
    analisis = list(analisis)

    for uno in analisis:
        uno.iniciar(trayectoria)

    for bloque in trayectoria.bloques(tamanio_bloque, inicio, fin):
        for uno in analisis:
            uno.procesar(bloque)

    return [uno.resultado() for uno in analisis]


#/ ---- FIN MÓDULO --------------------------------------------------------
