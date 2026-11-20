# flujo_nt

Análisis de trayectorias de dinámica molecular de mezclas que fluyen a través
de nanotubos de carbono, generadas con LAMMPS en formato netCDF (convención
AMBER).

El paquete calcula el flujo a través del tubo, la densidad y el perfil de
velocidad en función de la distancia al eje, la densidad a lo largo del eje y
el conteo de moléculas pervaporadas. Trabaja en el marco de referencia del
tubo, cualquiera sea el eje de la celda sobre el que esté alineado, de modo
que no hace falta rotar las trayectorias. Las lee por bloques de cuadros, con
lo que la memoria necesaria no depende de su largo, y corrige las posiciones
por la curvatura que el tubo adquiere durante la simulación.

Es la reescritura de los scripts de análisis del trabajo:

> S. Mosca, C. M. Carlevaro y E. Lomba, "Molecular dynamics study of
> pervaporation of water–methanol mixtures through graphene nanotubes",
> *The Journal of Chemical Physics* **163**, 164501 (2025).
> <https://doi.org/10.1063/5.0299861>

## Instalación

Requiere Python 3.9 o posterior. Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[desarrollo]"
```

La opción `-e` instala el paquete en modo editable: los cambios en `src/` se
ven sin reinstalar. El grupo `desarrollo` agrega pytest, pycodestyle y
numpydoc. La instalación crea el comando `flujo-nt` en el entorno; si se
agrega un comando nuevo, hay que reinstalar una vez para que aparezca. Sin
instalar el comando, `python -m flujo_nt` hace lo mismo.

## Datos de entrada

### Trayectoria

Un archivo netCDF con convención AMBER, escrito por LAMMPS con
`dump netcdf` o reescrito por OVITO. El dialecto se detecta con el atributo
global `program`: OVITO llama `atom_types` e `identifier` a las variables que
LAMMPS llama `type` e `id`.

| Variable        | Uso                                                      |
|-----------------|----------------------------------------------------------|
| `coordinates`   | siempre                                                  |
| `type`          | siempre; deben ser constantes en el tiempo               |
| `time`          | siempre                                                  |
| `cell_lengths`, `cell_origin` | siempre; la celda debe ser constante (NVT) |
| `id`            | si está, se verifica que el orden de los átomos no cambie |
| `velocities`    | flujo y perfil de velocidad                              |

Las demás variables (`forces`, `ix`, `iy`, `iz`, `mol`…) pueden estar o no;
como la lectura es perezosa, las que ningún cálculo usa no se leen. LAMMPS
guarda en `time` el número de paso, con el paso de integración en el atributo
`scale_factor`; el paquete lo usa para informar los pasos en la
pervaporación. Si falta (por ejemplo, en archivos reescritos por OVITO), se
indica con `--paso-integracion`.

Al abrir la trayectoria se comprueba que los tipos, el orden de los átomos y
la celda no cambien entre cuadros. El estilo `dump netcdf` de LAMMPS ordena
los átomos por identificador; `netcdf/mpiio` los escribe en el orden de los
procesos, que puede cambiar entre cuadros, y esas trayectorias se rechazan.

### Configuración

El archivo de `write_data` de la misma simulación. Es obligatorio cuando la
trayectoria registra una longitud de celda nula, lo que ocurre al usar
`kspace_modify slab`: la longitud verdadera se toma de la configuración. Si se
da, sus longitudes de celda prevalecen sobre las de la trayectoria. Sólo se
lee su encabezado (límites de la celda, cantidad de átomos y tipos, masas).

### Sistema

| Tipo | Código | Partícula                              |
|------|--------|----------------------------------------|
| 1    | `Oa`   | oxígeno del agua                       |
| 2    | `Ha`   | hidrógeno del agua                     |
| 3    | `Om`   | oxígeno del metanol                    |
| 4    | `Cm`   | metilo del metanol (átomo unido)       |
| 5    | `Hm`   | hidrógeno del metanol                  |
| 6    | —      | tubo y paredes unidas a él             |
| 7    | —      | paredes sueltas (sólo repulsivas)      |

Los flujos se calculan para `Oa` y `Om`, que representan a sus moléculas: el
flujo másico usa la masa de la molécula entera.

El tubo se mide en el primer cuadro a partir de los átomos de tipo 6. Debe
estar alineado con un eje de la celda, que tiene que ser su longitud mayor; el
eje se detecta solo. Su largo es la distancia entre las dos paredes que tocan
el borde transversal de la celda. El tubo se divide en anillos, y de cada uno
se miden el centro y el radio; las posiciones de las partículas se refieren
luego al eje curvo que forman esos centros. El radio del tubo es el promedio
de los radios de los anillos.

## Uso

### Comando `flujo-nt`

`flujo-nt analizar` recibe una o más trayectorias y, con una sola pasada por
cada una, escribe los resultados junto a ellas como
`<base>.<cálculo>.<especie>.dat`, donde `<base>` es la ruta de la trayectoria
sin la extensión. La configuración se busca como `<base>.final.data`, o se
indica con `--configuracion`.

```bash
flujo-nt analizar sistema.1.nc
flujo-nt analizar sistema.1.nc -c flujo,pervaporacion -e Oa
```

| Cálculo (`-c`)                | Especies por defecto | Archivos                                      |
|-------------------------------|----------------------|-----------------------------------------------|
| `flujo` \*                    | Oa, Om               | `flujo.inst.<esp>`                            |
| `densidad-radial` \*          | Oa, Ha, Om, Cm, Hm   | `dens.num.radial.promedio.<esp>`              |
| `perfil-velocidad` \*         | Oa, Om               | `perfil.velocidad.promedio.<esp>`             |
| `perfil-velocidad-serie`      | Oa, Om               | `perfil.velocidad.serie.<esp>`                |
| `densidad-longitudinal`       | Oa, Om               | `dens.num.longitudinal.<región>.promedio.<esp>` |
| `densidad-longitudinal-serie` | Oa, Om               | `dens.num.longitudinal.<región>.serie.<esp>`  |
| `pervaporacion`               | Oa, Om               | `conteo.pervaporacion.<esp>`                  |

Los marcados con \* se ejecutan si no se indica `-c`. La opción `-e` restringe
cada cálculo a las especies dadas. `flujo-nt analizar -h` muestra todas las
opciones: el ancho de los bins radiales, el reservorio de destino de la
pervaporación, el tamaño de bloque y el directorio de salida, entre otras.

La selección de trayectorias queda a cargo del intérprete de comandos, de modo
que el comando no depende de cómo estén organizados los directorios:

```bash
RAIZ=/home/usuario/reservorios
shopt -s globstar
flujo-nt analizar $RAIZ/R{1-5,3-0}/**/*.398K.{1,2}.nc -q 2> avisos.log
```

Si una trayectoria falla, se informa y se sigue con las demás; el código de
salida es 1 si falló alguna. Los avisos (por ejemplo, cada archivo que se
sobrescribe) y los errores van a la salida de errores, y `-q` silencia todo lo
demás.

`flujo-nt recortar` copia los primeros cuadros de una trayectoria a un archivo
nuevo, cuadro por cuadro, opcionalmente excluyendo variables:

```bash
flujo-nt recortar sistema.1.nc sistema.1.recortado.nc -n 2 -x forces
flujo-nt recortar sistema.1.nc -l      # sólo lista el contenido
```

### Como biblioteca

```python
from flujo_nt import (Configuracion, DensidadRadial, Flujo, Trayectoria,
                      Tubo, buscar_especie, escribir_resultado, recorrer)

agua = buscar_especie('Oa')
configuracion = Configuracion('sistema.1.final.data')

with Trayectoria('sistema.1.nc', configuracion=configuracion) as tr:
    tubo = Tubo.desde_trayectoria(tr)
    flujo, densidad = recorrer(tr, [Flujo(agua, tubo),
                                    DensidadRadial(agua, tubo)])

flujo['Jn']                                  # arreglo de numpy
escribir_resultado(flujo, 'resultados/sistema.1')
```

`recorrer` alimenta todos los análisis con una sola pasada; cada bloque se lee
una vez y lo comparten. Los análisis aceptan más parámetros que el comando
(por ejemplo, las fracciones del tubo que abarca la región analizada); están
documentados en sus docstrings.

## Resultados

Cada resultado es un archivo de texto con un encabezado de líneas `#`: el
cálculo y la especie, las unidades de cada columna, los parámetros del cálculo
(región, bins, volúmenes…), la versión de `flujo_nt`, la fecha, la
trayectoria de origen y, en la última línea, los nombres de las columnas. Los
números tienen formato `%.5e`, como los de LAMMPS. Se leen con
`numpy.loadtxt(ruta)`, o con `flujo_nt.leer_resultado(ruta)`, que además
recupera los nombres y las unidades.

Las regiones dentro del tubo son coaxiales con su eje curvo y abarcan, por
defecto, el 90 % central de su largo, para excluir la entrada y la salida, y
todo su radio. Los bins radiales tienen ancho `fraccion_bin_r × sigma_ref`
(por defecto 0.25 × 3.5 Å = 0.875 Å).

| Cálculo | Columnas | Descripción |
|---------|----------|-------------|
| `flujo.inst` | `tiempo` [fs], `n`, `Jn` [1/(fs Å²)], `Jm` [g/(mol fs Å²)] | Por cuadro: partículas en la región y flujos numérico y másico (suma de velocidades longitudinales sobre el volumen de la región). |
| `dens.num.radial.promedio` | `centro_bin` [Å], `n`, `rho_n_media`, `rho_n_desvio` [1/Å³] | Por bin radial: conteo total, densidad media y su desvío entre cuadros. |
| `perfil.velocidad.promedio` | `centro_bin` [Å], `n`, `v_media`, `v_desvio` [Å/fs] | Por bin radial: muestras, velocidad longitudinal media y su desvío sobre todas las muestras. |
| `perfil.velocidad.serie` | `tiempo` [fs], `centro_bin` [Å], `n`, `v_media` [Å/fs] | Por cuadro y bin radial, en formato largo. |
| `dens.num.longitudinal.<región>.promedio` | `centro_bin` [Å], `n_media`, `rho_n_media`, `rho_n_desvio` [1/Å³] | Por bin a lo largo del eje: conteo medio, densidad media y su desvío entre cuadros. |
| `dens.num.longitudinal.<región>.serie` | `tiempo` [fs], `centro_bin` [Å], `n`, `rho_n` [1/Å³] | Por cuadro y bin, en formato largo. |
| `conteo.pervaporacion` | `paso`, `n` | Por cuadro: número de paso y partículas en el reservorio de destino. |

La densidad longitudinal reproduce el cálculo del input de LAMMPS de las
simulaciones (`compute chunk/atom` y `fix ave/chunk`), como respaldo para las
simulaciones en que no se registró. Con `l` el largo y `r` el radio del tubo,
las regiones son `izq` (desde el borde de la celda hasta `-l/2`, en láminas de
toda la sección), `tubo` (de `-l/2` a `+l/2`, dentro de un cilindro recto de
radio `1.5 r`) y `der` (de `+l/2` al borde), con bins de ancho `l/32`. Como en
LAMMPS, usa coordenadas absolutas con el tubo centrado en el origen. La
pervaporación, con el mismo criterio, cuenta las partículas más allá de
`+l/2`, o de `-l/2` con `--lado -1`.

En el formato largo de las series hay una fila por cada cuadro y bin: el
tiempo se repite para todos los bins de un cuadro, y los bins se recorren
completos antes de pasar al cuadro siguiente.

## Validación

Los tests comparan los resultados sobre los datos de ejemplo con los de la
biblioteca original y con los que escribe LAMMPS durante la simulación: la
densidad longitudinal y la pervaporación coinciden bin a bin con LAMMPS, y los
conteos del flujo con la biblioteca original. También se verifica, con
sistemas sintéticos de geometría conocida, que los análisis den los valores
exactos, y que los resultados no dependan del tamaño de bloque.

La biblioteca original se conserva en `legado/`, junto con los scripts que
generan los valores de referencia: `generar_referencia.py` ejecuta la
biblioteca original sobre los datos de ejemplo y escribe
`tests/referencia_original.json`, y `extraer_referencia_lammps.py` toma los
archivos que escribió LAMMPS durante la simulación y escribe
`tests/referencia_lammps.json`.

## Tests

```bash
pytest
```

Los tests que usan los datos de ejemplo se omiten si los archivos no están en
`datos/ejemplo/`.

## Estructura

```
src/flujo_nt/      paquete
  trayectoria.py     lectura perezosa de trayectorias y configuraciones
  geometria.py       marco de referencia y medición del tubo curvo
  especies.py        especies de partículas del sistema
  analisis.py        flujo, densidades, perfil de velocidad, pervaporación
  escritura.py       escritura y lectura de resultados en texto
  recorte.py         recorte de trayectorias
  cli.py             comando flujo-nt
tests/             tests y valores de referencia (biblioteca original, LAMMPS)
rendimiento/       medición de tiempo y memoria según el tamaño de bloque
datos/ejemplo/     dos cuadros de una trayectoria real y su configuración
legado/            biblioteca original y scripts que generan las referencias
```

## Datos de ejemplo

`datos/ejemplo/` contiene los dos primeros cuadros de una simulación de la
mezcla agua-metanol con fracción molar de metanol 0.115, tubo de radio
nominal 1.5 y largo nominal 37.5 (en unidades de sigma_ref), campo eléctrico
de valor 1e-5 en la dirección longitudinal, a 398 K:

- `reservorios_R1-5_L37-5_X0-115_Elong-1eneg5_398K_2cuadros.nc`: trayectoria
  escrita por LAMMPS con `dump netcdf`.
- `reservorios_R1-5_L37-5_X0-115_Elong-1eneg5_398K_final.data`: configuración
  final escrita con `write_data`. Es necesaria porque la simulación usa
  `kspace_modify slab` y la trayectoria registra longitud de celda nula en z.

Como los nombres no siguen la convención `<base>.nc` y `<base>.final.data`,
con el comando hay que indicar la configuración:

```bash
cd datos/ejemplo
flujo-nt analizar *_2cuadros.nc --configuracion *_final.data -o /tmp/ejemplo
```

## Rendimiento

`rendimiento/medir_rendimiento.py` mide el tiempo real, el tiempo de CPU y el
pico de memoria de los análisis para distintos tamaños de bloque, cada
medición en un proceso propio:

```bash
python rendimiento/medir_rendimiento.py sistema.1.nc -t 1,10,50,200,todo \
    -r 3 -o rendimiento.dat
```

El tamaño `todo` procesa la trayectoria en un solo bloque, con todos los
cuadros en memoria a la vez.

## Estilo

El código sigue PEP 8 con dos excepciones deliberadas, declaradas en
`setup.cfg`: los comentarios de sección con el marcador `#/` y la sangría de
continuación alineada. Las docstrings siguen la convención numpydoc.

```bash
pycodestyle src tests rendimiento
```

## Licencia

MIT; ver `LICENSE`.
