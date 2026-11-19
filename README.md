# flujo_nt

Análisis de trayectorias de dinámica molecular de mezclas que fluyen a través
de nanotubos de carbono, generadas con LAMMPS en formato netCDF (convención
AMBER).

> README provisorio: describe la instalación y los tests. La documentación de
> los datos esperados y de cada script se completa con las clases de análisis.

## Instalación

Requiere Python 3.9 o posterior. Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[desarrollo]"
```

La opción `-e` instala el paquete en modo editable: los cambios en `src/` se
ven sin reinstalar. El grupo `desarrollo` agrega pytest, pycodestyle y
numpydoc.

## Tests

```bash
pytest
```

Los tests que usan los datos de ejemplo se omiten si los archivos no están en
`datos/ejemplo/`.

## Estructura

```
src/flujo_nt/    paquete
  trayectoria.py   lectura perezosa de trayectorias y configuraciones
  geometria.py     marco de referencia y medición del tubo curvo
  especies.py      especies de partículas del sistema
  analisis.py      flujo, densidades, perfil de velocidad, pervaporación
  escritura.py     escritura y lectura de resultados en texto
scripts/         herramientas de línea de comandos
tests/           tests y valores de referencia (biblioteca original, LAMMPS)
datos/ejemplo/   dos cuadros de una trayectoria real y su configuración
legado/          biblioteca original y scripts que generan las referencias
```

## Datos de ejemplo

`datos/ejemplo/` contiene los dos primeros cuadros de una simulación de la
mezcla agua-metanol con fracción molar de metanol 0.115, tubo de radio nominal 1.5 y largo nominal
37.5 (en unidades de sigma_ref), campo eléctrico de valor 1e-5 en la
dirección longitudinal, a 398 K:

- `reservorios_R1-5_L37-5_X0-115_Elong-1eneg5_398K_2cuadros.nc`: trayectoria
  escrita por LAMMPS con `dump netcdf`.
- `reservorios_R1-5_L37-5_X0-115_Elong-1eneg5_398K_final.data`: configuración
  final escrita con `write_data`. Es necesaria porque la simulación usa
  `kspace_modify slab` y la trayectoria registra longitud de celda nula en z.

## Estilo

El código sigue PEP 8 con dos excepciones deliberadas, declaradas en
`setup.cfg`: los comentarios de sección con el marcador `#/` y la sangría de
continuación alineada. Las docstrings siguen la convención numpydoc.

```bash
pycodestyle src tests scripts
```
