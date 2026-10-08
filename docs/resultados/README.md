# Resultados por corrida

Una ficha por corrida en esta carpeta, generada con `scripts/resumen.py`. Ordenadas por
mejor PSNR en CBSD68. El estado sale de `estado.json`; donde dice «deducido» es que la
corrida es vieja y no lo tiene, y se infirió contando filas de traza.

| corrida | mejor PSNR | en la it | última | estado | `‖g‖` > 1e4 | rechazos |
|---|---|---|---|---|---|---|
| [t8delta-100k-s1](t8delta-100k-s1.md) | **31.547** | 100,000 | 31.547 | terminada | 0.1 % | 163 |
| [t8delta-100k-s0](t8delta-100k-s0.md) | **31.526** | 64,000 | 31.524 | MATADA | 0.0 % | 18 |
| [replica-v4](replica-v4.md) | **31.478** | 68,000 | 30.955 | antigua, solo history.json | — | — |
| [t8delta-100k-corte-s1](t8delta-100k-corte-s1.md) | **31.419** | 18,000 | 31.419 | MATADA | 0.0 % | 4 |
| [t8delta-100k-corte-s0](t8delta-100k-corte-s0.md) | **31.419** | 16,000 | 31.393 | MATADA | 0.0 % | 11 |
| [tinit-10delta-10k](tinit-10delta-10k.md) | **31.352** | 10,000 | 31.352 | terminada | 0.0 % | 4 |
| [tinit-7p9delta-10k](tinit-7p9delta-10k.md) | **31.343** | 10,000 | 31.343 | terminada | 0.2 % | 93 |
| [x-bf16-det](x-bf16-det.md) | **31.194** | 5,000 | 31.194 | antigua, solo history.json | — | — |
| [tinit/A-seed0](tinit__A-seed0.md) | **31.190** | 4,000 | 31.190 | antigua, solo history.json | — | — |
| [precision/rapido-seed0](precision__rapido-seed0.md) | **31.190** | 4,000 | 31.190 | terminada (deducido) | 0.0 % | — |
| [tinit01/seed0](tinit01__seed0.md) | **31.188** | 4,000 | 31.188 | terminada (deducido) | 0.0 % | — |
| [disyuntor-100k](disyuntor-100k.md) | **31.182** | 4,000 | 27.631 | MATADA | 91.8 % | 45,925 |
| [precision/intermedio-seed3](precision__intermedio-seed3.md) | **31.178** | 4,000 | 31.178 | terminada (deducido) | 0.0 % | — |
| [clip/base-seed2](clip__base-seed2.md) | **31.177** | 4,000 | 31.177 | terminada (deducido) | 0.0 % | — |
| [nodDP/seed2](nodDP__seed2.md) | **31.175** | 4,000 | 31.175 | terminada (deducido) | 0.0 % | — |
| [lambdainit/B-seed3](lambdainit__B-seed3.md) | **31.167** | 4,000 | 31.167 | terminada (deducido) | 0.0 % | — |
| [lambdainit/B-seed1](lambdainit__B-seed1.md) | **31.165** | 4,000 | 31.165 | terminada (deducido) | 0.0 % | — |
| [nodDP/seed5](nodDP__seed5.md) | **31.163** | 4,000 | 31.163 | terminada (deducido) | 0.0 % | — |
| [precision/intermedio-seed1](precision__intermedio-seed1.md) | **31.161** | 4,000 | 31.161 | terminada (deducido) | 0.0 % | — |
| [precision/intermedio-seed4](precision__intermedio-seed4.md) | **31.159** | 4,000 | 31.159 | terminada (deducido) | 0.0 % | — |
| [precision/intermedio-seed10](precision__intermedio-seed10.md) | **31.158** | 4,000 | 31.158 | terminada (deducido) | 0.0 % | — |
| [precision/intermedio-seed2](precision__intermedio-seed2.md) | **31.157** | 4,000 | 31.157 | terminada (deducido) | 0.0 % | — |
| [precision/rapido-seed6](precision__rapido-seed6.md) | **31.155** | 4,000 | 31.155 | terminada (deducido) | 0.0 % | — |
| [tinit-2delta-50k](tinit-2delta-50k.md) | **31.155** | 4,000 | 31.155 | MATADA | 1.2 % | 79 |
| [replica-bf16](replica-bf16.md) | **31.155** | 4,000 | 20.173 | antigua, solo history.json | — | — |
| [precision/rapido-seed5](precision__rapido-seed5.md) | **31.152** | 4,000 | 31.152 | terminada (deducido) | 0.0 % | — |
| [tinit/A-seed5](tinit__A-seed5.md) | **31.141** | 4,000 | 31.141 | antigua, solo history.json | — | — |
| [precision/rapido-seed7](precision__rapido-seed7.md) | **31.139** | 4,000 | 31.139 | terminada (deducido) | 0.0 % | — |
| [precision/rapido-seed11](precision__rapido-seed11.md) | **31.138** | 4,000 | 31.138 | terminada (deducido) | 0.0 % | — |
| [precision/intermedio-seed7](precision__intermedio-seed7.md) | **31.127** | 4,000 | 31.127 | terminada (deducido) | 0.0 % | — |
| [ablacion/seed0](ablacion__seed0.md) | **31.127** | 3,000 | 17.392 | terminada (deducido) | 18.8 % | — |
| [clip/base-seed4](clip__base-seed4.md) | **31.114** | 3,000 | 9.956 | terminada (deducido) | 19.6 % | — |
| [traza/seed4](traza__seed4.md) | **31.108** | 3,000 | 12.850 | terminada (deducido) | 17.6 % | — |
| [tinit01/seed2](tinit01__seed2.md) | **31.105** | 3,000 | -7.750 | terminada (deducido) | 2.8 % | — |
| [tinit/A-seed3](tinit__A-seed3.md) | **31.102** | 4,000 | 31.102 | antigua, solo history.json | — | — |
| [clip/base-seed0](clip__base-seed0.md) | **31.099** | 3,000 | 22.462 | terminada (deducido) | 14.2 % | — |
| [clip/clip-seed0](clip__clip-seed0.md) | **31.099** | 3,000 | 20.173 | terminada (deducido) | 13.4 % | — |
| [tinit01/seed5](tinit01__seed5.md) | **31.095** | 3,000 | -4.106 | terminada (deducido) | 9.2 % | — |
| [precision/rapido-seed4](precision__rapido-seed4.md) | **31.078** | 4,000 | 31.078 | terminada (deducido) | 0.0 % | — |
| [precision/intermedio-seed8](precision__intermedio-seed8.md) | **31.077** | 3,000 | 20.806 | terminada (deducido) | 6.5 % | — |
| [tinit/B-larga](tinit__B-larga.md) | **31.074** | 2,500 | 20.173 | antigua, solo history.json | — | — |
| [clip/base-seed3](clip__base-seed3.md) | **31.070** | 4,000 | 31.070 | terminada (deducido) | 0.0 % | — |
| [clip/base-seed5](clip__base-seed5.md) | **31.057** | 3,000 | 20.173 | terminada (deducido) | 7.8 % | — |
| [precision/rapido-seed10](precision__rapido-seed10.md) | **31.056** | 3,000 | -2.567 | terminada (deducido) | 24.4 % | — |
| [precision/rapido-seed2](precision__rapido-seed2.md) | **31.005** | 2,000 | 30.937 | terminada (deducido) | 0.0 % | — |
| [nodDP/seed4](nodDP__seed4.md) | **31.005** | 2,000 | -1.601 | terminada (deducido) | 32.1 % | — |
| [tinit01/seed1](tinit01__seed1.md) | **31.004** | 2,000 | -5.433 | terminada (deducido) | 25.4 % | — |
| [lambdainit/A-seed2](lambdainit__A-seed2.md) | **31.003** | 4,000 | 31.003 | terminada (deducido) | 0.0 % | — |
| [precision/rapido-seed1](precision__rapido-seed1.md) | **31.002** | 2,000 | nan | terminada (deducido) | 3.1 % | — |
| [tinit/A-seed4](tinit__A-seed4.md) | **30.993** | 2,000 | 20.173 | antigua, solo history.json | — | — |
| [precision/intermedio-seed0](precision__intermedio-seed0.md) | **30.993** | 2,000 | 12.956 | terminada (deducido) | 45.4 % | — |
| [lambdainit/A-seed5](lambdainit__A-seed5.md) | **30.993** | 2,000 | 15.484 | terminada (deducido) | 41.5 % | — |
| [lambdainit/A-seed0](lambdainit__A-seed0.md) | **30.992** | 2,000 | 21.459 | terminada (deducido) | 13.8 % | — |
| [lambdainit/A-seed3](lambdainit__A-seed3.md) | **30.991** | 2,000 | nan | terminada (deducido) | 13.7 % | — |
| [nodDP/seed0](nodDP__seed0.md) | **30.989** | 2,000 | 25.012 | terminada (deducido) | 9.5 % | — |
| [tinit/A-seed2](tinit__A-seed2.md) | **30.988** | 2,000 | 20.804 | antigua, solo history.json | — | — |
| [lambdainit/B-seed2](lambdainit__B-seed2.md) | **30.985** | 2,000 | 12.780 | terminada (deducido) | 41.2 % | — |
| [tinit01/seed3](tinit01__seed3.md) | **30.982** | 2,000 | 3.147 | terminada (deducido) | 41.2 % | — |
| [precision/rapido-seed3](precision__rapido-seed3.md) | **30.982** | 2,000 | 16.713 | terminada (deducido) | 40.1 % | — |
| [precision/rapido-seed8](precision__rapido-seed8.md) | **30.972** | 2,000 | 9.835 | terminada (deducido) | 41.1 % | — |
| [precision/intermedio-seed6](precision__intermedio-seed6.md) | **30.943** | 2,000 | 15.530 | terminada (deducido) | 30.7 % | — |
| [replica-v5](replica-v5.md) | **30.941** | 2,000 | 20.173 | antigua, solo history.json | — | — |
| [v7-tmin](v7-tmin.md) | **30.935** | 2,000 | 26.569 | antigua, solo history.json | — | — |
| [traza/seed1](traza__seed1.md) | **30.894** | 1,500 | 30.894 | terminada (deducido) | 0.0 % | — |
| [traza/seed2](traza__seed2.md) | **30.868** | 1,500 | 30.681 | terminada (deducido) | 0.1 % | — |
| [s1-literal](s1-literal.md) | **30.852** | 1,500 | 30.852 | antigua, solo history.json | — | — |
| [lambdainit/A-seed4](lambdainit__A-seed4.md) | **30.748** | 1,000 | 20.131 | terminada (deducido) | 2.4 % | — |
| [precision/intermedio-seed11](precision__intermedio-seed11.md) | **30.742** | 1,000 | -33.806 | terminada (deducido) | 61.1 % | — |
| [lambdainit/B-seed0](lambdainit__B-seed0.md) | **30.738** | 1,000 | -30.190 | terminada (deducido) | 51.0 % | — |
| [lambdainit/A-seed1](lambdainit__A-seed1.md) | **30.728** | 1,000 | 30.605 | terminada (deducido) | 0.1 % | — |
| [clip/base-seed1](clip__base-seed1.md) | **30.728** | 1,000 | 4.525 | terminada (deducido) | 56.8 % | — |
| [clip/clip-seed1](clip__clip-seed1.md) | **30.728** | 1,000 | 30.728 | incompleta (deducido) | 0.0 % | — |
| [nodDP/seed1](nodDP__seed1.md) | **30.726** | 1,000 | 20.744 | terminada (deducido) | 40.5 % | — |
| [lambdainit/B-seed4](lambdainit__B-seed4.md) | **30.717** | 1,000 | 17.541 | terminada (deducido) | 54.9 % | — |
| [tinit01/seed4](tinit01__seed4.md) | **30.716** | 1,000 | 17.225 | terminada (deducido) | 72.3 % | — |
| [ab-clip](ab-clip.md) | **30.712** | 1,500 | 30.712 | antigua, solo history.json | — | — |
| [precision/rapido-seed9](precision__rapido-seed9.md) | **30.695** | 1,000 | nan | terminada (deducido) | 6.5 % | — |
| [nodDP/seed3](nodDP__seed3.md) | **30.690** | 1,000 | 20.583 | terminada (deducido) | 60.0 % | — |
| [precision/intermedio-seed5](precision__intermedio-seed5.md) | **30.619** | 1,000 | 6.427 | terminada (deducido) | 53.5 % | — |
| [repeat5/run0](repeat5__run0.md) | **30.395** | 2,000 | 27.756 | terminada (deducido) | 16.3 % | — |
| [repeat5/run1](repeat5__run1.md) | **30.395** | 2,000 | 27.756 | terminada (deducido) | 16.3 % | — |
| [repeat5/run2](repeat5__run2.md) | **30.395** | 2,000 | 27.756 | terminada (deducido) | 16.3 % | — |
| [repeat5/run3](repeat5__run3.md) | **30.395** | 2,000 | 27.756 | terminada (deducido) | 16.3 % | — |
| [repeat5/run4](repeat5__run4.md) | **30.395** | 2,000 | 27.756 | terminada (deducido) | 16.3 % | — |
| [s1-nockpt](s1-nockpt.md) | **30.377** | 1,000 | 30.377 | antigua, solo history.json | — | — |
| [b2-A-control](b2-A-control.md) | **30.340** | 600 | 30.340 | antigua, solo history.json | — | — |
| [replica-v2](replica-v2.md) | **30.315** | 2,000 | 18.408 | antigua, solo history.json | — | — |
| [b2-C-poison](b2-C-poison.md) | **30.310** | 600 | 30.310 | antigua, solo history.json | — | — |
| [nodDP/solo-B](nodDP__solo-B.md) | **30.084** | 300 | 30.084 | terminada (deducido) | 0.0 % | — |
| [ab-lr](ab-lr.md) | **30.084** | 1,500 | 30.084 | antigua, solo history.json | — | — |
| [nodDP/solo-A](nodDP__solo-A.md) | **30.082** | 300 | 30.082 | terminada (deducido) | 0.0 % | — |
| [nodDP/det-A](nodDP__det-A.md) | **30.034** | 300 | 30.034 | terminada (deducido) | 0.0 % | — |
| [ablacion/sinckpt-B](ablacion__sinckpt-B.md) | **30.026** | 300 | 30.026 | terminada (deducido) | 0.0 % | — |
| [ablacion/sinbench-A](ablacion__sinbench-A.md) | **30.025** | 300 | 30.025 | terminada (deducido) | 0.0 % | — |
| [ablacion/sinbench-B](ablacion__sinbench-B.md) | **30.025** | 300 | 30.025 | terminada (deducido) | 0.0 % | — |
| [nodDP/det-B](nodDP__det-B.md) | **30.022** | 300 | 30.022 | terminada (deducido) | 0.0 % | — |
| [ablacion/sinckpt-A](ablacion__sinckpt-A.md) | **30.005** | 300 | 30.005 | terminada (deducido) | 0.0 % | — |
| [s1-noclip](s1-noclip.md) | **25.943** | 1,500 | 25.943 | antigua, solo history.json | — | — |
| [lambdainit/B-seed5](lambdainit__B-seed5.md) | **22.085** | 3,000 | 21.909 | terminada (deducido) | 58.0 % | — |
| [referencia-100k](referencia-100k.md) | **21.735** | 38,000 | -16.147 | terminada (deducido) | 97.3 % | — |
| [v6-fp32-ieee](v6-fp32-ieee.md) | **20.173** | 2,000 | 20.173 | antigua, solo history.json | — | — |
| [precision/intermedio-seed9](precision__intermedio-seed9.md) | **10.467** | 4,000 | 10.467 | terminada (deducido) | 80.5 % | — |
| [tinit/A-seed1](tinit__A-seed1.md) | **-3.407** | 4,000 | -3.407 | antigua, solo history.json | — | — |
| [caza-adjunta/seed7](caza-adjunta__seed7.md) | — | — | — | terminada (deducido) | 55.0 % | — |
| [caza-adjunta/seed7-fino](caza-adjunta__seed7-fino.md) | — | — | — | terminada (deducido) | 0.1 % | — |
| [caza-adjunta/seed8](caza-adjunta__seed8.md) | — | — | — | terminada (deducido) | 0.9 % | — |
| [caza-adjunta/seed8-fino](caza-adjunta__seed8-fino.md) | — | — | — | terminada (deducido) | 0.0 % | — |
| [contrafactual/disyuntor](contrafactual__disyuntor.md) | — | — | — | terminada (deducido) | 0.0 % | — |
| [contrafactual/skip2694](contrafactual__skip2694.md) | — | — | — | terminada (deducido) | 0.0 % | — |
| [traza-adjunta](traza-adjunta.md) | — | — | — | terminada (deducido) | 0.0 % | — |
