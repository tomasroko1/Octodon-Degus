Análisis de células hipocampales en *Octodon degus*. Todo el código se corre y prueba desde el notebook `playground.ipynb`.

---

## 1. Mapas (`maps.py`)
`firing_map` grafica el recorrido del animal en gris y le pone puntos rojos donde la neurona disparó, filtrando los momentos en los que el animal estaba quieto. `rate_map` toma esos datos y genera un mapa de calor dividiendo los spikes por el tiempo que el animal pasó en cada zona.

## 2. GLM Manual (`glm_posicion_manual`)
Dividimos la caja en una grilla de campanas de gauss y ajustamos el peso de cada una asumiendo una distribución de poisson. Sirve para ver cómo responde la neurona al espacio, pero al ser una grilla rígida, el mapa queda bastante "pixelado".

## 3. GAM Espacial (`get_gam_posicion`)
Para evitar lo pixelado del GLM, usamos Modelos Aditivos Generalizados (GAMs) con splines. El modelo ajusta una superficie suave y continua que representa de forma mucho más natural el "place field".

Para armarlo, cortamos el tiempo en ventanitas (ej. 0.1s) y alineamos la trayectoria con la cantidad de spikes de ese momento. Usamos 5x5 splines: si le damos demasiados, el modelo se sobreajusta y termina memorizando exactamente por dónde caminó el degú.

Métricas del GAM: el *Pseudo R-Squared* nos dice qué porcentaje de los disparos se explican exclusivamente por la posición, y los grados de libertad efectivos (EDoF) muestran qué tanta complejidad realmente necesitó el modelo.

## 4. GAM de Mirada / Viewpoint (`get_gam_viewpoint_1d`)
Probamos si la neurona responde a hacia dónde está mirando el animal. Calculamos el ángulo de la cabeza, tiramos un rayo imaginario hasta la pared, y "desenrollamos" las 4 paredes en una línea 1D continua. Sobre ese perímetro ajustamos un GAM cíclico para ver si la célula tiene preferencia por mirar a alguna pared en particular.
