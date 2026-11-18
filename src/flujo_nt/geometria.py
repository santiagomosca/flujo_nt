#! /usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Geometría del nanotubo: marco de referencia y medición del tubo curvo.

Este módulo contiene las dos piezas que permiten analizar una trayectoria sin
rotarla:

- :class:`MarcoTubo` responde qué dirección cartesiana es longitudinal al tubo
  y cuáles son transversales. Reemplaza a la detección del eje y a las seis
  copias del bloque que traducía el eje a índices.
- :class:`Tubo` mide el largo, el radio y la curvatura del tubo, y expresa
  posiciones en el sistema de referencia del tubo: coordenada longitudinal
  respecto de su centro y coordenadas transversales respecto de su eje curvo.

El tubo se modela siempre como curvo. Se lo corta en anillos a lo largo del
eje y se mide el centro y el radio de cada uno; un tubo recto es el caso en
que todos los centros coinciden, y no requiere tratamiento aparte.

Clases
------
MarcoTubo
    Asignación de direcciones longitudinal y transversales.
Tubo
    Medición del tubo y corrección de posiciones por su curvatura.

Examples
--------
>>> configuracion = Configuracion('sistema.final.data')
>>> with Trayectoria('sistema.nc', configuracion=configuracion) as tr:
...     tubo = Tubo.desde_trayectoria(tr)
...     oxigenos = tr.mascara_tipo(1)
...     for bloque in tr.bloques():
...         posiciones = tubo.corregir(bloque.posiciones[:, oxigenos])
...         en_tubo = tubo.en_region(posiciones, fraccion_largo=0.9)

@author: Santiago Mosca <santiagomosca85@gmail.com>

fecha: noviembre de 2026
"""

import numpy as np


#/ ---- CONSTANTES DEL MÓDULO --------------------------------------------

DISTANCIA_ENLACE_CC = 1.42
"""float: Distancia de enlace carbono-carbono en el grafeno, en Å."""

TOLERANCIA_EJE = 2 * DISTANCIA_ENLACE_CC
"""float: Diferencia mínima, en Å, entre el lado mayor de la celda y el
siguiente para considerar que el eje del tubo está bien definido."""

TIPOS_TUBO_PREDETERMINADOS = (6,)
"""tuple of int: Tipos de partícula que forman el tubo y sus paredes
cosidas."""

FRACCION_BORDE_PREDETERMINADA = 0.975
"""float: Fracción del semilado transversal de la celda a partir de la cual
un átomo se considera en contacto con el borde."""

FRACCION_TUBO_PREDETERMINADA = 0.95
"""float: Fracción del largo del tubo sobre la que se miden los anillos, para
excluir la unión con las paredes."""

ANCHO_ANILLO_PREDETERMINADO = 1.5 * DISTANCIA_ENLACE_CC
"""float: Ancho aproximado de cada anillo, en Å."""

MINIMO_ATOMOS_ANILLO = 3
"""int: Cantidad mínima de átomos para que un anillo defina centro y radio."""

FRACCION_COMPACIDAD_PARED = 0.1
"""float: Extensión longitudinal máxima de cada pared, como fracción de la
separación entre paredes."""


#/ ---- EXCEPCIONES -------------------------------------------------------

class ErrorGeometria(Exception):
    """
    Error en la determinación de la geometría del tubo.

    Se levanta cuando la geometría no puede determinarse sin ambigüedad. Se
    prefiere fallar a devolver un valor plausible pero incorrecto.
    """


#/ ---- MARCO DEL TUBO ----------------------------------------------------

class MarcoTubo:
    """
    Asignación de direcciones longitudinal y transversales al tubo.

    Responde una sola pregunta: en este sistema, qué componente cartesiana es
    longitudinal al tubo y cuáles son las dos transversales. Se construye una
    vez y se pasa a todo cálculo que necesite distinguirlas, de modo que una
    trayectoria con el tubo sobre el eje Z se analiza en Z, sin rotarla.

    Las direcciones transversales siguen una permutación cíclica de los ejes:
    el eje X tiene por transversales a (Y, Z), el eje Y a (Z, X) y el eje Z a
    (X, Y). Es la convención de la biblioteca original, y conserva la
    orientación derecha del sistema de coordenadas.

    Parameters
    ----------
    eje : {'X', 'Y', 'Z'}
        Dirección cartesiana paralela al eje del tubo.

    Attributes
    ----------
    eje : str
        Dirección cartesiana paralela al eje del tubo.
    indice_longitudinal : int
        Índice de la componente longitudinal: 0, 1 o 2.
    indices_transversales : tuple of int
        Índices de las dos componentes transversales, en orden cíclico.

    Raises
    ------
    ErrorGeometria
        Si el eje no es uno de 'X', 'Y' o 'Z'.

    Examples
    --------
    >>> marco = MarcoTubo('Z')
    >>> marco.indice_longitudinal, marco.indices_transversales
    (2, (0, 1))
    """

    EJES = ('X', 'Y', 'Z')
    """tuple of str: Ejes cartesianos, en el orden de sus índices."""

    def __init__(self, eje):
        eje = str(eje).upper()

        if eje not in self.EJES:
            raise ErrorGeometria(
                f'Eje de tubo inválido: {eje!r}. Válidos: {self.EJES}')

        self.eje = eje
        self.indice_longitudinal = self.EJES.index(eje)
        self.indices_transversales = (
            (self.indice_longitudinal + 1) % 3,
            (self.indice_longitudinal + 2) % 3)

    @classmethod
    def desde_celda(cls, celda, tolerancia=TOLERANCIA_EJE):
        """
        Detección del eje del tubo a partir de la celda de simulación.

        El eje del tubo es la dirección de mayor longitud de celda. Se exige
        que el lado mayor supere al siguiente en al menos ``tolerancia``; si
        no, el eje es ambiguo y se levanta un error en lugar de elegir uno.

        Parameters
        ----------
        celda : Celda
            Celda de simulación. No puede ser degenerada.
        tolerancia : float, optional
            Diferencia mínima entre el lado mayor y el siguiente, en Å. Por
            defecto :data:`TOLERANCIA_EJE`, el doble de la distancia de
            enlace C-C.

        Returns
        -------
        MarcoTubo
            Marco con el eje detectado.

        Raises
        ------
        ErrorGeometria
            Si la celda es degenerada o si el eje es ambiguo.

        Notes
        -----
        Una celda con una longitud nula, como la que escribe LAMMPS en
        trayectorias con ``kspace_modify slab``, haría que la dirección del
        tubo pareciera la más corta. Por eso se rechaza en lugar de procesarla:
        :class:`Trayectoria` ya corrige la celda al construirse, y esta
        verificación protege a quien construya una celda por otra vía.
        """
        if celda.degenerada:
            raise ErrorGeometria(
                f'No se puede detectar el eje del tubo en una celda con '
                f'longitudes nulas: {celda.longitudes}. Corregir la celda '
                f'con el archivo de configuración.')

        orden = np.argsort(celda.longitudes)
        mayor, segunda = celda.longitudes[orden[2]], celda.longitudes[orden[1]]

        if mayor - segunda < tolerancia:
            raise ErrorGeometria(
                f'Eje del tubo ambiguo: el lado mayor de la celda '
                f'({mayor:.4f} Å) no supera al siguiente ({segunda:.4f} Å) en '
                f'al menos {tolerancia:.4f} Å. Especificar el eje '
                f'explícitamente.')

        return cls(cls.EJES[orden[2]])

    def longitudinal(self, arreglo):
        """
        Componente longitudinal de un arreglo de vectores.

        Parameters
        ----------
        arreglo : numpy.ndarray
            Arreglo con las componentes cartesianas en el último eje, con
            forma ``(..., 3)``.

        Returns
        -------
        numpy.ndarray
            Componente longitudinal, con forma ``(...)``. Es una vista del
            arreglo de entrada, no una copia.
        """
        return arreglo[..., self.indice_longitudinal]

    def transversales(self, arreglo):
        """
        Componentes transversales de un arreglo de vectores.

        Parameters
        ----------
        arreglo : numpy.ndarray
            Arreglo con las componentes cartesianas en el último eje, con
            forma ``(..., 3)``.

        Returns
        -------
        numpy.ndarray
            Componentes transversales, con forma ``(..., 2)``, en el orden de
            :attr:`indices_transversales`. Es una copia.
        """
        return arreglo[..., list(self.indices_transversales)]

    def radial(self, arreglo):
        """
        Distancia de cada vector al eje longitudinal.

        Sobre posiciones ya corregidas con :meth:`Tubo.corregir`, es la
        distancia al eje curvo del tubo.

        Parameters
        ----------
        arreglo : numpy.ndarray
            Arreglo con las componentes cartesianas en el último eje, con
            forma ``(..., 3)``.

        Returns
        -------
        numpy.ndarray
            Norma de las componentes transversales, con forma ``(...)``.
        """
        return np.linalg.norm(self.transversales(arreglo), axis=-1)

    def __repr__(self):
        return f'MarcoTubo({self.eje!r})'


#/ ---- TUBO --------------------------------------------------------------

class Tubo:
    """
    Nanotubo curvo, medido a partir de las posiciones de sus átomos.

    Es un objeto de valor: guarda el resultado de una medición y no mantiene
    referencia a la trayectoria de la que proviene. Se obtiene normalmente con
    :meth:`medir` o :meth:`desde_trayectoria`, no con el constructor.

    El tubo se corta en anillos a lo largo de su eje, y para cada anillo se
    guarda su centro transversal y su radio. Los centros describen el eje
    curvo; :meth:`corregir` interpola entre ellos para expresar posiciones
    relativas a ese eje.

    Parameters
    ----------
    marco : MarcoTubo
        Marco de referencia del tubo.
    largo : float
        Distancia entre las posiciones medias de las dos paredes, en Å.
    centro : float
        Coordenada longitudinal del punto medio entre las paredes, en Å.
    bordes_anillos : array_like
        Bordes de los anillos sobre el eje longitudinal, en coordenadas
        absolutas, con forma ``(n_anillos + 1,)``, en Å.
    centros_anillos : array_like
        Centro transversal de cada anillo, con forma ``(n_anillos, 2)``, en
        Å, en el orden de :attr:`MarcoTubo.indices_transversales`.
    radios_anillos : array_like
        Radio medio de cada anillo, con forma ``(n_anillos,)``, en Å.

    Attributes
    ----------
    marco : MarcoTubo
        Marco de referencia del tubo.
    largo : float
        Largo del tubo, en Å.
    centro : float
        Coordenada longitudinal del centro del tubo, en Å.
    bordes_anillos : numpy.ndarray
        Bordes de los anillos, con forma ``(n_anillos + 1,)``, en Å.
    centros_anillos : numpy.ndarray
        Centros transversales de los anillos, con forma ``(n_anillos, 2)``,
        en Å.
    radios_anillos : numpy.ndarray
        Radios de los anillos, con forma ``(n_anillos,)``, en Å.

    Raises
    ------
    ErrorGeometria
        Si las formas de los arreglos de anillos no son consistentes.

    See Also
    --------
    MarcoTubo : Marco de referencia del tubo.
    """

    def __init__(self, marco, largo, centro, bordes_anillos, centros_anillos,
            radios_anillos):
        self.marco = marco
        self.largo = float(largo)
        self.centro = float(centro)
        self.bordes_anillos = np.asarray(bordes_anillos, dtype='f8')
        self.centros_anillos = np.asarray(centros_anillos, dtype='f8')
        self.radios_anillos = np.asarray(radios_anillos, dtype='f8')

        n_anillos = self.bordes_anillos.size - 1

        if n_anillos < 1:
            raise ErrorGeometria('El tubo debe tener al menos un anillo.')

        if self.centros_anillos.shape != (n_anillos, 2):
            raise ErrorGeometria(
                f'Centros de anillos con forma {self.centros_anillos.shape}; '
                f'se esperaba {(n_anillos, 2)}.')

        if self.radios_anillos.shape != (n_anillos,):
            raise ErrorGeometria(
                f'Radios de anillos con forma {self.radios_anillos.shape}; '
                f'se esperaba {(n_anillos,)}.')

        if np.any(np.diff(self.bordes_anillos) <= 0):
            raise ErrorGeometria(
                'Los bordes de los anillos deben ser estrictamente '
                'crecientes.')

    #/ ---- valores derivados ----------------------------------------------

    @property
    def radio(self):
        """
        float: Radio medio del tubo, en Å.

        Es el promedio de los radios de los anillos. Cada radio se mide desde
        el centro de su propio anillo, de modo que la curvatura del tubo no
        lo infla.
        """
        return float(self.radios_anillos.mean())

    @property
    def n_anillos(self):
        """int: Cantidad de anillos en que se midió el tubo."""
        return self.radios_anillos.size

    @property
    def medios_anillos(self):
        """
        numpy.ndarray: Posición longitudinal del punto medio de cada anillo.

        Tiene forma ``(n_anillos,)``, en Å, en coordenadas absolutas. Son los
        nodos sobre los que se interpolan los centros del eje curvo.
        """
        return (self.bordes_anillos[:-1] + self.bordes_anillos[1:]) / 2

    #/ ---- medición -------------------------------------------------------

    @classmethod
    def medir(cls, posiciones, marco, celda,
            fraccion_borde=FRACCION_BORDE_PREDETERMINADA,
            fraccion_tubo=FRACCION_TUBO_PREDETERMINADA,
            ancho_anillo=ANCHO_ANILLO_PREDETERMINADO):
        """
        Medición del tubo a partir de las posiciones de sus átomos.

        La medición tiene dos etapas.

        El largo se obtiene de las paredes cosidas al tubo. Se toman los
        átomos que tocan el borde transversal de la celda: el tubo nunca se
        acerca a ese borde, y las paredes, que abarcan toda la sección,
        siempre lo hacen. Esos átomos se separan en dos paredes, y el largo es
        la distancia entre sus posiciones longitudinales medias. Se mide sobre
        el borde de la celda porque las paredes están combadas: cerca del tubo
        se adelantan unos 2.5 Å respecto del borde.

        Los anillos se miden sobre la fracción central del tubo, para excluir
        la unión con las paredes. Esa región se divide en anillos de ancho
        uniforme, y de cada uno se obtiene el centro transversal como
        promedio de sus átomos y el radio como distancia media de sus átomos
        a ese centro.

        Parameters
        ----------
        posiciones : numpy.ndarray
            Posiciones de los átomos del tubo y sus paredes cosidas en un
            cuadro, con forma ``(n, 3)``, en Å. No deben incluirse otras
            estructuras, como las paredes sueltas que cierran los
            reservorios.
        marco : MarcoTubo
            Marco de referencia del tubo.
        celda : Celda
            Celda de simulación, ya corregida si era degenerada.
        fraccion_borde : float, optional
            Fracción del semilado transversal de la celda a partir de la cual
            un átomo se considera en contacto con el borde. Por defecto
            :data:`FRACCION_BORDE_PREDETERMINADA`.
        fraccion_tubo : float, optional
            Fracción del largo sobre la que se miden los anillos. Por defecto
            :data:`FRACCION_TUBO_PREDETERMINADA`.
        ancho_anillo : float, optional
            Ancho aproximado de cada anillo, en Å. Se ajusta para que un
            número entero de anillos cubra exactamente la región medida. Por
            defecto :data:`ANCHO_ANILLO_PREDETERMINADO`.

        Returns
        -------
        Tubo
            Tubo medido.

        Raises
        ------
        ErrorGeometria
            Si no hay átomos en contacto con el borde, si no se distinguen dos
            paredes compactas, o si algún anillo tiene menos de
            :data:`MINIMO_ATOMOS_ANILLO` átomos.

        Notes
        -----
        La medición opera sólo sobre los átomos del tubo y está
        completamente vectorizada: sobre un tubo de unos 4800 átomos cuesta
        alrededor de medio milisegundo, la mitad de lo que cuesta leer un
        cuadro de cien mil partículas. Para un tubo que se deforma durante la
        simulación puede repetirse en cada cuadro, sobre posiciones que ya
        están en memoria, sin lecturas adicionales.

        Respecto de ``obtener_dimensiones_tubo`` de la biblioteca original se
        corrigieron tres defectos en la construcción de los anillos: el
        anillo central tenía el doble de ancho, los átomos de los extremos de
        la región podían quedar fuera de todo anillo, y un anillo vacío
        desplazaba los centros de todos los siguientes.
        """
        posiciones = np.asarray(posiciones)

        if posiciones.ndim != 2 or posiciones.shape[1] != 3:
            raise ErrorGeometria(
                f'Se esperaban posiciones con forma (n, 3); se recibió '
                f'{posiciones.shape}.')

        largo, centro = cls._medir_paredes(
            posiciones, marco, celda, fraccion_borde)

        #/ Anillos de ancho uniforme sobre la región central del tubo
        largo_anillos = largo * fraccion_tubo
        n_anillos = max(1, int(round(largo_anillos / ancho_anillo)))
        bordes = centro + np.linspace(
            -largo_anillos / 2, largo_anillos / 2, n_anillos + 1)

        centros, radios, conteos = cls._medir_anillos(
            marco.longitudinal(posiciones),
            marco.transversales(posiciones),
            bordes)

        insuficientes = np.flatnonzero(conteos < MINIMO_ATOMOS_ANILLO)
        if insuficientes.size:
            raise ErrorGeometria(
                f'Anillos con menos de {MINIMO_ATOMOS_ANILLO} átomos: '
                f'{insuficientes.tolist()} (de {n_anillos}). Aumentar '
                f'ancho_anillo o verificar los tipos de partícula del tubo.')

        return cls(marco, largo, centro, bordes, centros, radios)

    @classmethod
    def desde_trayectoria(cls, trayectoria, marco=None,
            tipos=TIPOS_TUBO_PREDETERMINADOS, cuadro=0, **opciones):
        """
        Medición del tubo en un cuadro de una trayectoria.

        Parameters
        ----------
        trayectoria : Trayectoria
            Trayectoria abierta.
        marco : MarcoTubo, optional
            Marco de referencia. Si es None se detecta a partir de la celda
            de la trayectoria.
        tipos : iterable of int, optional
            Tipos de partícula que forman el tubo y sus paredes cosidas. Por
            defecto :data:`TIPOS_TUBO_PREDETERMINADOS`. No deben incluirse
            los tipos de las paredes sueltas de los reservorios.
        cuadro : int, optional
            Cuadro en el que se mide el tubo. Por defecto el primero.
        **opciones
            Argumentos adicionales para :meth:`medir`.

        Returns
        -------
        Tubo
            Tubo medido.

        Raises
        ------
        ErrorGeometria
            En los casos descriptos en :meth:`medir`, o si el eje es ambiguo.
        ErrorTrayectoria
            Si alguno de los tipos no está presente o el cuadro está fuera de
            rango.
        """
        if marco is None:
            marco = MarcoTubo.desde_celda(trayectoria.celda)

        mascara = trayectoria.mascara_tipos(tipos)
        posiciones = trayectoria.cuadro(cuadro).posiciones[0][mascara]

        return cls.medir(posiciones, marco, trayectoria.celda, **opciones)

    @staticmethod
    def _medir_paredes(posiciones, marco, celda, fraccion_borde):
        """
        Largo y centro longitudinal del tubo a partir de sus paredes.

        Parameters
        ----------
        posiciones : numpy.ndarray
            Posiciones de los átomos del tubo, con forma ``(n, 3)``.
        marco : MarcoTubo
            Marco de referencia del tubo.
        celda : Celda
            Celda de simulación.
        fraccion_borde : float
            Fracción del semilado transversal que define el contacto con el
            borde.

        Returns
        -------
        largo : float
            Distancia entre las posiciones medias de las paredes, en Å.
        centro : float
            Punto medio entre las paredes, en Å.

        Raises
        ------
        ErrorGeometria
            Si no hay átomos en contacto con el borde, o si no se distinguen
            dos paredes compactas.
        """
        transversales = list(marco.indices_transversales)
        centro_celda = (celda.origen + celda.longitudes / 2)[transversales]
        umbral = (celda.longitudes / 2 * fraccion_borde)[transversales]

        en_borde = np.any(
            np.abs(marco.transversales(posiciones) - centro_celda) > umbral,
            axis=1)

        if not np.any(en_borde):
            raise ErrorGeometria(
                'Ningún átomo del tubo toca el borde transversal de la celda: '
                'no se encontraron las paredes cosidas al tubo.')

        longitudinal = marco.longitudinal(posiciones)[en_borde]

        #/ Separación de las dos paredes por el punto medio de su extensión
        corte = (longitudinal.min() + longitudinal.max()) / 2
        paredes = (longitudinal[longitudinal < corte],
                   longitudinal[longitudinal >= corte])

        if paredes[0].size == 0 or paredes[1].size == 0:
            raise ErrorGeometria(
                'Los átomos en contacto con el borde no forman dos paredes.')

        separacion = paredes[1].mean() - paredes[0].mean()

        #/ Cada pared debe ser compacta frente a la separación entre ambas. Si
        #/ no lo es, se incluyeron átomos de otras estructuras, como las
        #/ paredes sueltas de los reservorios, que también tocan el borde.
        for pared in paredes:
            extension = pared.max() - pared.min()
            if extension > FRACCION_COMPACIDAD_PARED * separacion:
                raise ErrorGeometria(
                    f'Una pared se extiende {extension:.2f} Å sobre el eje, '
                    f'frente a {separacion:.2f} Å de separación entre '
                    f'paredes. Probablemente se incluyeron tipos de partícula '
                    f'que no pertenecen al tubo, como las paredes sueltas de '
                    f'los reservorios.')

        return float(separacion), float((paredes[0].mean()
                                         + paredes[1].mean()) / 2)

    @staticmethod
    def _medir_anillos(longitudinal, transversales, bordes):
        """
        Centro, radio y cantidad de átomos de cada anillo.

        Está completamente vectorizada: las sumas por anillo se hacen con
        ``numpy.bincount`` en lugar de recorrer los anillos.

        Parameters
        ----------
        longitudinal : numpy.ndarray
            Coordenada longitudinal de cada átomo, con forma ``(n,)``.
        transversales : numpy.ndarray
            Coordenadas transversales de cada átomo, con forma ``(n, 2)``.
        bordes : numpy.ndarray
            Bordes de los anillos, crecientes, con forma ``(n_anillos + 1,)``.
            Los átomos fuera de ``[bordes[0], bordes[-1]]`` se descartan.

        Returns
        -------
        centros : numpy.ndarray
            Centro transversal de cada anillo, con forma ``(n_anillos, 2)``.
            Es NaN en los anillos vacíos.
        radios : numpy.ndarray
            Radio medio de cada anillo, con forma ``(n_anillos,)``. Es NaN en
            los anillos vacíos.
        conteos : numpy.ndarray
            Cantidad de átomos de cada anillo, con forma ``(n_anillos,)``.
        """
        n_anillos = bordes.size - 1

        dentro = (longitudinal >= bordes[0]) & (longitudinal <= bordes[-1])
        transversales = transversales[dentro]

        #/ Índice de anillo de cada átomo. El borde superior se incluye en el
        #/ último anillo.
        anillo = np.clip(
            np.digitize(longitudinal[dentro], bordes) - 1, 0, n_anillos - 1)

        conteos = np.bincount(anillo, minlength=n_anillos)

        with np.errstate(invalid='ignore', divide='ignore'):
            centros = np.stack([
                np.bincount(anillo, weights=transversales[:, k],
                            minlength=n_anillos) / conteos
                for k in range(2)], axis=1)

            distancias = np.linalg.norm(transversales - centros[anillo],
                                        axis=1)
            radios = np.bincount(anillo, weights=distancias,
                                 minlength=n_anillos) / conteos

        return centros, radios, conteos

    #/ ---- uso ------------------------------------------------------------

    def corregir(self, posiciones):
        """
        Expresión de posiciones en el sistema de referencia del tubo.

        La coordenada longitudinal pasa a medirse desde el centro del tubo, y
        las transversales desde el eje curvo: a cada posición se le resta el
        centro del tubo a su altura, interpolado linealmente entre los
        centros de los anillos vecinos. Más allá del primer y del último
        anillo se usa el centro del anillo extremo.

        Parameters
        ----------
        posiciones : numpy.ndarray
            Posiciones, con forma ``(..., 3)``, en Å. Típicamente
            ``bloque.posiciones[:, mascara]``, con forma
            ``(n_cuadros, n_seleccionados, 3)``.

        Returns
        -------
        numpy.ndarray
            Posiciones en el sistema del tubo, con la misma forma y tipo que
            la entrada. Las componentes conservan su orden cartesiano, de
            modo que :attr:`marco` sigue indicando cuál es cada una. Es una
            copia: la entrada no se modifica.

        Notes
        -----
        Se devuelven las tres componentes y no sólo la coordenada radial
        porque la radial se obtiene de las transversales con
        :meth:`MarcoTubo.radial`, pero las transversales no pueden
        recuperarse de la radial.

        Reemplaza a las cuatro copias del bloque de interpolación de la
        biblioteca original, que además calculaban los pesos contra los
        bordes de los anillos en lugar de contra sus puntos medios: el centro
        interpolado quedaba corrido medio anillo, y en la mitad exterior de
        los anillos extremos no se aplicaba corrección.
        """
        corregidas = np.array(posiciones, copy=True)
        longitudinal = self.marco.longitudinal(corregidas)
        medios = self.medios_anillos

        #/ La componente longitudinal no cambia hasta el final, así que puede
        #/ usarse como vista mientras se corrigen las transversales
        for k, indice in enumerate(self.marco.indices_transversales):
            corregidas[..., indice] -= np.interp(
                longitudinal, medios, self.centros_anillos[:, k])

        corregidas[..., self.marco.indice_longitudinal] -= self.centro

        return corregidas

    def limites(self, fraccion=1.0):
        """
        Límites longitudinales de una fracción del tubo.

        Parameters
        ----------
        fraccion : float, optional
            Fracción del largo del tubo, centrada en su centro. Por defecto 1.

        Returns
        -------
        numpy.ndarray
            Límites inferior y superior, con forma ``(2,)``, en Å, en el
            sistema del tubo: relativos a su centro.
        """
        semilargo = self.largo * fraccion / 2

        return np.array([-semilargo, semilargo])

    def en_region(self, posiciones_corregidas, fraccion_largo=1.0,
            fraccion_radio=1.0):
        """
        Pertenencia de posiciones a una región cilíndrica del tubo.

        La región es un cilindro coaxial con el eje curvo del tubo, de largo
        ``fraccion_largo * largo`` y radio ``fraccion_radio * radio``.

        Parameters
        ----------
        posiciones_corregidas : numpy.ndarray
            Posiciones en el sistema del tubo, obtenidas con
            :meth:`corregir`, con forma ``(..., 3)``.
        fraccion_largo : float, optional
            Fracción del largo del tubo. Por defecto 1.
        fraccion_radio : float, optional
            Fracción del radio del tubo. Por defecto 1.

        Returns
        -------
        numpy.ndarray
            Máscara booleana con forma ``(...)``.
        """
        inferior, superior = self.limites(fraccion_largo)
        longitudinal = self.marco.longitudinal(posiciones_corregidas)
        radial = self.marco.radial(posiciones_corregidas)

        return ((longitudinal >= inferior) & (longitudinal <= superior)
                & (radial <= self.radio * fraccion_radio))

    def __repr__(self):
        return (f'Tubo(eje={self.marco.eje!r}, largo={self.largo:.6f}, '
                f'radio={self.radio:.6f}, n_anillos={self.n_anillos})')


#/ ---- FIN MÓDULO --------------------------------------------------------
