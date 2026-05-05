### Análisis Espacial (`firing_map`)
Dentro de los scripts herramientas (`utils.py`), el método principal es `firing_map(sesion, tetrodo, neurona)`. Este método nos permite visualizar la actividad de una neurona individual durante una sesión en el Open Field (OF). 

El método realiza lo siguiente:
- Grafica la **trayectoria completa del animal** en gris claro (con un suavizado gaussiano).
- Superpone en **puntos rojos** las posiciones exactas donde la neurona seleccionada disparó un potencial de acción.
- Aplica un **filtro de velocidad** (por defecto > 2 cm/s) para omitir los disparos que ocurren cuando el animal está quieto.

Podemos ver el resultado devuelto por esta función. Esta es nuestra **place cell** candidata (sesión 2, tetrodo 3, célula 3) 
![Figura 1](Figure_1.png)

### Modelado Estadístico: GLM y GAM (`glm_analysis.py`)
Para evaluar la codificación espacial de la neurona, utilizamos Modelos Lineales Generalizados (GLM) y Modelos Aditivos Generalizados (GAM) asumiendo una distribución de Poisson para los spikes.

#### GLM Manual (Figura 3)
Ajustamos el modelo de manera manual dividiendo el espacio mediante una grilla de funciones base independientes.

El espacio se fracciona en muchísimos cuadraditos, y el modelo ajusta el peso de cada región por separado. El algoritmo cuenta con regularización (ridge) para que los bines que el animal nunca pisó no "exploten".
![Figura 3](Figure_3.png)

#### GAM Automático con Splines (GAM 1 y GAM 2)
En otro acercamiento, implementamos un GAM utilizando **Splines** que encuentra la penalización óptima mediante cross validation.

- **GAM 1**: Serie temporal ("Predicción vs Realidad"). En negro se ven las barras que representan los disparos (spikes) discretos reales que emitió la neurona. La línea roja continua superpuesta es lo que el modelo GAM predice que debería haber disparado basándose exclusivamente en la posición exacta del degú en ese momento.
![GAM 1](GAM1.png)

- **GAM 2**: Es el place field predictivo del GAM. El place field se expresa en este caso como un gradiente suave y continuo, reflejando de manera mucho más natural la verdadera probabilidad espacial de la célula (comparar con modelos lineales).
![GAM 2](GAM2.png)
