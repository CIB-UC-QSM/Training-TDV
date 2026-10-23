# Qué hay en `docs/`

| entrada | qué es |
|---|---|
| [`bitacora.md`](bitacora.md) | el estado del entrenamiento 3D: lo fijado para la corrida larga y por qué, lo medido que sigue valiendo, lo abierto, lo que dejó el 2D y qué corridas quedan en `runs/`. Las entradas nuevas van al final, fechadas |
| [`porque-del-codigo.md`](porque-del-codigo.md) | el porqué de cada constante y decisión de `train3d.py`, `ddr/qsm3d.py` y la receta de `train.py` que hereda el 3D, por secciones; el código las cita por nombre («MESETA», «MEMORIA») |
| [`23-10-pasar-corrida-a-otro-pc.md`](23-10-pasar-corrida-a-otro-pc.md) | un ejemplo completo de cómo seguir en otro PC una corrida a medio entrenar (`--reanudar`), con la copia por red local al final |
| [`figuras/`](figuras/) | las figuras de las mediciones (locales: no se versionan, por el peso), con nombre `DD-MM-descripcion` (el día en que se generan), en una carpeta por red (`resultados-control/`, `resultados-sobreajuste/`) y, cuando hay varias semillas de ruido, una por semilla (`seed1/`, `seed2/`, `seed3/`); el script y los números de cada una, en su entrada de la bitácora |
| [`papers/`](papers/INDEX.md) | la bibliografía: `INDEX.md` y una nota destilada por paper en `notes/`, que se lee antes que el PDF (los PDF no se versionan; skill `papers`) |
