# Qué hay en `docs/`

| entrada | qué es |
|---|---|
| [`bitacora.md`](bitacora.md) | el estado del entrenamiento 3D: lo fijado para la corrida larga y por qué, lo medido que sigue valiendo, lo abierto, lo que dejó el 2D y qué corridas quedan en `runs/`. Las entradas nuevas van al final, fechadas |
| [`porque-del-codigo.md`](porque-del-codigo.md) | el porqué de cada constante y decisión de `train3d.py`, `ddr/qsm3d.py` y la receta de `train.py` que hereda el 3D, por secciones; el código las cita por nombre («MESETA», «MEMORIA») |
| [`23-10-pasar-corrida-a-otro-pc.md`](23-10-pasar-corrida-a-otro-pc.md) | un ejemplo completo de cómo seguir en otro PC una corrida a medio entrenar (`--reanudar`), con la copia por red local al final |
| [`Notas.txt`](Notas.txt) | notas del autor: los datos 3D, la carga en `../QSM-TDV`, la prueba de sobreajuste, qué es cada corrida `prep3d-*` y las ideas a futuro, de más a menos costo-efectivas |
| [`Sanity Check/`](Sanity%20Check/resumen/RESUMEN.md) | las figuras de las mediciones. Sólo `resumen/` se versiona: `RESUMEN.md` resume lo vigente con sus figuras e inventaría las viejas. El resto es local (~14 MB): las figuras del 21 y 22-10, con nombre `DD-MM-descripcion` (el día en que se generan), en una carpeta por red o medición (`resultados-control/`, `resultados-sobreajuste/`, `resultados-ciego-log-l32/`, `resultados-g10-s2-2250/`, `por-que-no-reescalar/`) y PNG sueltos del 21-10, más `26-10-auditoria-ruido/` (la auditoría del ruido y de los datos QSM, ya con SNR = máximo/σ); de las tres semillas de ruido, sólo `seed1/`, porque las otras dan lo mismo (≤ 0.01 pp). Las anteriores al 23-10 usan el SNR de la media del cerebro: SNR_pico = SNR_media / 0.316. El script y los números de cada una, en su entrada de la bitácora |
| [`papers/`](papers/INDEX.md) | la bibliografía: `INDEX.md` y una nota destilada por paper en `notes/`, que se lee antes que el PDF (los PDF no se versionan; skill `papers`) |
