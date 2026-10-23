# Pasar una corrida a medio entrenar a otro PC

Un ejemplo completo con la corrida real: `ciego-epocas-l32-10k`, que va de la 5070 Ti de este PC
(«origen») a la RTX PRO 6000 de 96 GB («destino»), y allá sigue hasta la 20 000. Con otra corrida,
se cambian el nombre y las banderas.

## Qué viaja

Todo lo que hace falta para seguir está en `estado-<it>.pth`, dentro de la carpeta de la corrida:
- los pesos;
- los momentos de Adam;
- los generadores de parches y de ruido;
- los contadores del disyuntor y su respaldo;
- el historial.

`train3d.py` lo escribe cada `--guardar-cada` iteraciones (acá, 496), y siempre en la última.
Reanudar desde un estado sigue la corrida en el mismo punto: los mismos parches y el mismo ruido.

El resto de la carpeta lo lee la reanudación o el seguimiento:
- `config.json`, `history.json`, `traza.npy`;
- `qsm.jsonl` y `qsm_sim2.jsonl`;
- `seguimiento.json`, que identifica la corrida de W&B;
- `best.pth`, `best_qsm.pth` y `last.pth`.

## 1. En el origen: dejar un estado y parar

1. **El último estado:**
   ```bash
   ls runs/ciego-epocas-l32-10k/estado-*.pth | tail -1      # p. ej. estado-005456.pth
   ```
2. **Si la corrida sigue viva,** se espera al estado siguiente y se para justo después: se pierde
   lo hecho desde el último estado. El estado se escribe después de la validación de esa
   iteración, y el log lo confirma con la línea de la iteración siguiente (`[  5475/10000]`).
   ```bash
   tail -f runs/ciego-epocas-l32-10k/salida.log      # esperar la línea de la iteración siguiente
   systemctl --user stop tdv-ciego-epocas-l32-10k
   ```
   Si termina sola, el estado final es `estado-010000.pth`.
3. **Nunca en los dos PC a la vez:** escribirían el mismo `--out` y la misma corrida de W&B.
4. **Anotar el comando entero:** está en `runs/ciego-epocas-l32-10k/plan-desde-4960.sh`. Al reanudar
   se repiten todas las banderas. Sólo pueden cambiar las de `LIBRES` en `train3d.py`:
   - iteraciones, cadencias, memoria y seguimiento;
   - las de la evaluación QSM;
   - `--val-snr`;
   - `--guard`, `--guard-abs` y `--guard-respaldo`.

   Si cambia otra, `--reanudar` se niega y la nombra.

## 2. En el destino: código y entorno

El mismo código que corrió en el origen: `main` con el commit «training 3D en wandb».

```bash
# en el origen, si todavía no está en GitHub (lo decide el autor):
git push origin main

# en el destino (README, «Llevar el entrenamiento 3D a otra máquina», pasos 1 y 2):
cat /proc/driver/nvidia/version                  # driver ≥ 580
git clone https://github.com/CIB-UC-QSM/Training-TDV.git && cd Training-TDV
git log -1 --format=%s                           # tiene que decir: training 3D en wandb
uv sync --locked --managed-python && mkdir -p runs
uv run python -c "import torch; print(torch.cuda.get_device_name())"
uv run wandb login                               # la misma cuenta (entity cienfueg0z-tdv)
```

Si el destino ya tiene el repositorio, alcanza con `git pull` y el `git log -1` de arriba.

## 3. Copiar datos y corrida

Desde la raíz del repositorio en el origen. `-R` conserva las rutas relativas y `-L` copia
archivos en vez de enlaces.

```bash
D=usuario@destino:Training-TDV/
rsync -aRL --info=progress2 datasets/ldm100k3d/images datasets/ldm100k3d/mascaras \
    datasets/qsm_cosmos datasets/qsm_sim2 $D
rsync -aRL --info=progress2 runs/ciego-epocas-l32-10k $D
```

Pesan ~32 GB los datos y ~1 GB la corrida. Las listas (`lista_train_1000.txt`, `lista_eval.txt`)
vienen con el clon. Los estados viejos se pueden excluir, por ejemplo con
`--exclude 'estado-00[0-4]*'`; el que se va a reanudar, no.

```bash
# en el destino, desde Training-TDV/:
find -L datasets/ldm100k3d/images datasets/ldm100k3d/mascaras datasets/qsm_cosmos \
    datasets/qsm_sim2 -type f | wc -l            # 4028
ls runs/ciego-epocas-l32-10k/estado-005456.pth
```

## 4. Reanudar en el destino

Es el comando del origen, con tres diferencias:
- `--reanudar` apunta al estado copiado;
- `--iterations` es el techo nuevo;
- corre el `train3d.py` del repositorio, no la copia en `codigo-disyuntor/`.

Las reglas 11 y 12 de AGENTS.md (la GPU 1, los hilos 16–31) son de este PC: allá se usa la GPU
que tenga.

```bash
systemd-run --user --collect --unit tdv-ciego-epocas-l32-10k -p WorkingDirectory=$PWD \
    bash -c ".venv/bin/python -u train3d.py \
    --out runs/ciego-epocas-l32-10k --iterations 20000 --lote 32 --epocas \
    --reanudar runs/ciego-epocas-l32-10k/estado-005456.pth \
    --hitos --meseta 0 --mascara-cerebro --seed 3 --guard 10 \
    --train-list datasets/ldm100k3d/lista_train_1000.txt \
    --sigma-min 0.81 --sigma-max 55 --sigma-log \
    --eval-every 496 --eval-qsm-cada 0 --qsm-target-std 0.25 --qsm-sim2 datasets/qsm_sim2 \
    --val-snr 25 50 100 --estabilidad-lado 0 \
    --guardar-cada 496 --log-every 25 --seguimiento online \
    >> runs/ciego-epocas-l32-10k/salida.log 2>&1"
```

La unidad de systemd usa `.venv/bin/python`, no `uv run`, porque el `PATH` de una unidad no
incluye `~/.local/bin`.

## 5. Verificar los primeros minutos

```bash
tail -f runs/ciego-epocas-l32-10k/salida.log
```

- **`AVISO: el micro-lote pasa de 3 a N`.** En 96 GB, `auto` elige trozos más grandes. Cambia el
  orden de suma del gradiente: desde ahí la corrida no es bit a bit la que hubiera seguido, pero
  la receta es la misma.
- **`seguimiento: W&B online ... corrida <id>, reabierta`.** Si se reanudó desde el último paso
  registrado, sigue la misma corrida de W&B. Desde un estado anterior abre otra (`nueva`), porque
  W&B no reescribe pasos. Le pasó a esta corrida al reanudar desde la 4 960.
- **La primera línea de iteración:** su pérdida tiene que ser del orden de la del origen en esa
  zona (20–30 en la 5 000), con `rech` y `reb` siguiendo los contadores.
- **La primera validación** va en el múltiplo de 496 siguiente: el PSNR en el cerebro tiene que
  seguir la curva de W&B, sin saltos.

## 6. Para traerla de vuelta

Lo mismo al revés:
1. parar después de un estado;
2. copiar `runs/ciego-epocas-l32-10k` al origen;
3. reanudar con `--vram-gb 16`, o dejar `auto`.

Vuelve el `AVISO` del micro-lote.

## 7. Variante: gráficos de W&B desde cero (una rama)

Con otro `--out` y el mismo `--reanudar`, la corrida es una rama (`ddr/seguimiento.py`, bloque
final): el entrenamiento sigue exacto (pesos, Adam, parches, ruido, contador de iteraciones), pero
abre una corrida nueva de W&B, en el grupo de la original, que empieza en la iteración siguiente al
estado. La historia vieja queda en las corridas viejas.

```bash
--out runs/ciego-epocas-l32-20k --reanudar runs/ciego-epocas-l32-10k/estado-010000.pth
```

- **El eje x sigue en 10 001, no en 0:** el contador es parte del estado, y de él dependen los
  hitos del lr.
- **`best.pth` y `best_qsm.pth` se heredan del estado:** la rama sólo escribe un `best_qsm.pth`
  nuevo si mejora el mejor de la corrida original.
- **De la corrida original basta el estado**, que lleva la traza, en su misma ruta. Con su
  `seguimiento.json` al lado, la rama queda enlazada a la corrida vieja de W&B (`padre_id`).

---

**Nota: por red local.** Si los dos PC están en la misma red, el `rsync` de arriba va directo por
SSH:
1. **En el destino:** `sudo apt install openssh-server`, y la IP con `hostname -I` (p. ej.
   `192.168.1.40`).
2. **En el origen:** `ssh-copy-id usuario@192.168.1.40`, una vez, para no escribir la clave.
   Después, `D=usuario@192.168.1.40:Training-TDV/` en el paso 3.
3. **Velocidad:** en una LAN gigabit no conviene `-z`, porque comprimir es más lento que la red:
   ~110 MB/s, unos 5–6 minutos para los 32 GB (estimado). Por Wi-Fi puede ser 5–10 veces más
   lento: ahí sí `-z`, y `--partial` para retomar una copia cortada.
4. **Sin SSH:** un disco externo, con el mismo `rsync -aRL` hacia el disco y desde el disco.
