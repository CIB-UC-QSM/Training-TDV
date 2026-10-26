# Pasar una corrida a medio entrenar a otro PC

Un ejemplo completo con la corrida en vuelo, `ciego-epocas-l32-20k`: cómo pasarla de la 5070 Ti de
este PC («origen») a la RTX PRO 6000 de 96 GB («destino»), para que allá siga hasta la 20 000. Con
otra corrida, se cambian el nombre y las banderas.

## Qué viaja

Todo lo que hace falta para seguir está en `estado-<it>.pth`, dentro de la carpeta de la corrida:
- los pesos;
- los momentos de Adam;
- los generadores de parches y de ruido;
- los contadores del disyuntor y su respaldo;
- el historial.

`train3d.py` lo escribe cada `--guardar-cada` iteraciones (aquí, 496: la 10 416, la 10 912…), y
siempre en la última. Reanudar desde un estado sigue la corrida en el mismo punto: los mismos
parches y el mismo ruido.

El resto de la carpeta lo lee la reanudación o el seguimiento:
- `config.json`, `history.json`, `traza.npy`;
- `qsm.jsonl` y `qsm_sim2.jsonl`;
- `seguimiento.json`, que identifica la corrida de W&B;
- `best.pth`, `best_qsm.pth` y `last.pth`;
- `codigo/`, la copia del código que lanza `plan.sh`.

## 1. En el origen: dejar un estado y parar

1. **El último estado:**
   ```bash
   ls runs/ciego-epocas-l32-20k/estado-*.pth | tail -1      # p. ej. estado-010912.pth
   ```
2. **Si la corrida sigue viva,** se espera al estado siguiente y se para en cuanto aparece: se
   pierde lo hecho desde el último estado. El estado se escribe después de la validación de esa
   iteración, a un temporal que se renombra al final: si `ls` lo muestra, está entero.
   ```bash
   ls runs/ciego-epocas-l32-20k/estado-*.pth | tail -1      # repetir hasta que cambie
   systemctl --user stop tdv-ciego-epocas-l32-20k
   ```
   Parada antes de la línea de log siguiente al estado (`[ 10925/20000]` tras el de la 10 912),
   W&B sigue la misma corrida; parada después, abre otra (paso 5). Si termina sola, el estado final
   es `estado-020000.pth`.
3. **Nunca en los dos PC a la vez:** escribirían el mismo `--out` y la misma corrida de W&B.
4. **Anotar el comando entero:** está en `runs/ciego-epocas-l32-20k/plan.sh`. Al reanudar se
   repiten todas las banderas. Sólo pueden cambiar las de `LIBRES` en `train3d.py`:
   - iteraciones, cadencias, memoria y seguimiento;
   - las de la evaluación QSM, también `--qsm-sim2`;
   - `--val-snr`;
   - `--guard`, `--guard-abs` y `--guard-respaldo`.

   Si cambia otra, `--reanudar` se niega y la nombra.

## 2. En el destino: código y entorno

El mismo código que corre en el origen: el de `runs/ciego-epocas-l32-20k/codigo/`, que el paso 3
compara con el clon.

```bash
# en el origen, si todavía no está en GitHub (lo decide el autor):
git push origin main

# en el destino (README, «Llevar el entrenamiento 3D a otra máquina», pasos 1 y 2):
cat /proc/driver/nvidia/version                  # driver ≥ 580
git clone https://github.com/CIB-UC-QSM/Training-TDV.git && cd Training-TDV
uv sync --locked --managed-python && mkdir -p runs
uv run python -c "import torch; print(torch.cuda.get_device_name())"
uv run wandb login                               # la misma cuenta (entity cienfueg0z-tdv)
loginctl enable-linger                           # una vez
```

Sin `enable-linger`, systemd para las unidades del usuario cuando se cierra su última sesión, y
con ellas la corrida. Si el destino ya tiene el repositorio, alcanza con `git pull`.

## 3. Copiar datos y corrida

Desde la raíz del repositorio en el origen. `-R` conserva las rutas relativas y `-L` copia
archivos en vez de enlaces.

```bash
D=usuario@destino:Training-TDV/
rsync -aRL --info=progress2 datasets/ldm100k3d/images datasets/ldm100k3d/mascaras \
    datasets/qsm_cosmos datasets/qsm_sim2 $D
rsync -aRL --info=progress2 runs/ciego-epocas-l32-20k $D
```

Los datos pesan ~32 GB, y cada estado de la corrida, ~48 MB. Las listas (`lista_train_1000.txt`,
`lista_eval.txt`) vienen con el clon. Los estados viejos se pueden excluir, por ejemplo con
`--exclude 'estado-010*'`; el que se va a reanudar, no.

```bash
# en el destino, desde Training-TDV/:
find -L datasets/ldm100k3d/images datasets/ldm100k3d/mascaras datasets/qsm_cosmos \
    datasets/qsm_sim2 -type f | wc -l            # 4028
ls runs/ciego-epocas-l32-20k/estado-010912.pth
C=runs/ciego-epocas-l32-20k/codigo                # sin salida: el mismo código
diff -q train3d.py $C/train3d.py; diff -q train.py $C/train.py
diff -rq --exclude=__pycache__ ddr $C/ddr
```

## 4. Reanudar en el destino

Es el comando de `plan.sh`, con tres diferencias:
- `--reanudar` apunta al estado copiado de esta corrida, no al de la 10 000 del que salió la rama;
- corre el `train3d.py` del repositorio, no la copia en `codigo/` (son iguales: paso 3);
- sin `CUDA_VISIBLE_DEVICES=1` ni `taskset -c 16-31`.

Las reglas 11 y 12 de AGENTS.md (la GPU 1, los hilos 16–31) son de este PC: allá se usa la GPU
que tenga. `--iterations` queda en 20 000; para seguir más allá, se sube el techo.

```bash
systemd-run --user --collect --unit tdv-ciego-epocas-l32-20k -p WorkingDirectory=$PWD \
    bash -c ".venv/bin/python -u train3d.py \
    --out runs/ciego-epocas-l32-20k --iterations 20000 --lote 32 --epocas \
    --reanudar runs/ciego-epocas-l32-20k/estado-010912.pth \
    --hitos --meseta 0 --mascara-cerebro --seed 3 --guard 10 \
    --train-list datasets/ldm100k3d/lista_train_1000.txt \
    --sigma-min 0.81 --sigma-max 55 --sigma-log \
    --eval-every 496 --eval-qsm-cada 0 --qsm-target-std 0.15 0.02 0.05 --qsm-sim2 datasets/qsm_sim2 \
    --val-snr 25 50 100 --estabilidad-lado 0 \
    --guardar-cada 496 --log-every 25 --seguimiento online \
    >> runs/ciego-epocas-l32-20k/salida.log 2>&1"
```

La unidad de systemd usa `.venv/bin/python`, no `uv run`, porque el `PATH` de una unidad no
incluye `~/.local/bin`.

## 5. Verificar los primeros minutos

```bash
tail -f runs/ciego-epocas-l32-20k/salida.log
```

- **`AVISO: el micro-lote pasa de 3 a N`.** En 96 GB, `auto` elige trozos más grandes. Cambia el
  orden de suma del gradiente, y con ruido ciego también el ruido de cada parche: cada micro-lote
  sortea sus σ antes que su ruido. Desde ahí la corrida no es bit a bit la que hubiera seguido,
  pero la receta es la misma.
- **`seguimiento: W&B online ... corrida <id>, reabierta`.** Si se paró antes de la línea de log
  siguiente al estado (paso 1), sigue la misma corrida de W&B. Si no, W&B ya tiene pasos después
  del estado y no los reescribe: abre otra con el mismo nombre y en el mismo grupo (`nueva`), y el
  id viejo queda en `descartadas` de `seguimiento.json`.
- **La primera línea de iteración:** su pérdida tiene que ser del orden de la del origen en esa
  zona (15–25 en la 11 000), con `rech` y `reb` siguiendo los contadores.
- **La primera validación** va en el múltiplo de 496 siguiente: el PSNR en el cerebro tiene que
  seguir la curva de W&B (33.75 dB en la 10 912), sin saltos.

## 6. Para traerla de vuelta

Lo mismo al revés:
1. parar después de un estado;
2. copiar `runs/ciego-epocas-l32-20k` al origen;
3. reanudar con `--vram-gb 16`, o dejar `auto`.

Vuelve el `AVISO` del micro-lote.

## 7. Variante: gráficos de W&B desde cero (una rama)

Con otro `--out` y el mismo `--reanudar`, la corrida es una rama (`ddr/seguimiento.py`, bloque
final): el entrenamiento sigue exacto (pesos, Adam, parches, ruido, contador de iteraciones), pero
abre una corrida nueva de W&B, en el grupo de la original, que empieza en la iteración siguiente al
estado. La historia vieja queda en las corridas viejas. Así salió `ciego-epocas-l32-20k` de
`ciego-epocas-l32-10k`:

```bash
--out runs/ciego-epocas-l32-20k --reanudar runs/ciego-epocas-l32-10k/estado-010000.pth
```

- **El eje x sigue en 10 001, no en 0:** el contador es parte del estado, y de él dependen los
  hitos del lr.
- **`best.pth` y `best_qsm.pth` se heredan del estado:** la rama sólo escribe un `best_qsm.pth`
  nuevo si mejora el mejor de la corrida original.
- **Si la rama cambia `--qsm-target-std`,** el `best_qsm.pth` heredado y las filas viejas de
  `qsm.jsonl` son de otro criterio: se compara desde la primera evaluación de la rama. En la 20k, el
  heredado era el de la 1 488, elegido a 0.25; lo reemplazó en la 10 416 el de 26.21 % a 0.02.
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
