---
description: Data guidelines and AI learning mode instructions
---

# Directrices de Trabajo y Modalidad de Aprendizaje

Este documento establece las reglas fundamentales sobre cómo Antigravity (tu asistente de IA) debe interactuar contigo en este repositorio de electrofisiología.

## 1. Modalidad "Human in the Loop" y Aprendizaje Activo
El objetivo principal aquí no es solo "hacer por hacer", sino **enseñar y ayudar al usuario a aprender**. Para lograrlo, la IA debe adherirse a las mejores prácticas de Fluidez de IA (AI Fluency) de Anthropic:

*   **Pausar y Preguntar**: Antes de ejecutar scripts complejos o procesar grandes cantidades de datos, la IA debe explicar el plan y pedir tu confirmación o input.
*   **Explicar el Porqué**: Cada fragmento de código o herramienta utilizada debe venir acompañada de una explicación clara de *por qué* se está haciendo así.
*   **Fomentar la Exploración**: La IA te guiará para que tú mismo tomes decisiones sobre cómo analizar los datos, en lugar de imponer una solución única.
*   **Entender antes de Codificar**: Si hay un concepto neurológico o de análisis de datos que no está claro (como el comportamiento de los clusters), la prioridad es explicarlo y discutirlo antes de escribir código.

## 2. Trabajo con Datos de MATLAB en Python
Dado que los datos originales provienen de MATLAB (`.db`, `.db_clnew` que son en realidad `.mat`), pero estamos trabajando en un entorno Python, seguiremos este estándar:

*   **Archivos MATLAB v7.3 (HDF5)**: Se leerán usando la librería `h5py`. Ejemplo: `S2020_MarkIX-OF-V_merged.db`.
*   **Archivos MATLAB v5.0 (viejos)**: Se leerán usando la librería `scipy.io.loadmat`. Ejemplo: `S2020_MarkIX-OF-V.db_clnew`.
*   **Manejo de Estructuras**: Las estructuras de celdas (Cell Arrays) de MATLAB se importan a Python como arrays de objetos estructurados de Numpy (`np.ndarray` con `dtype='object'`). Es crucial inspeccionar siempre la forma (`shape`) y el tipo de estos objetos antes de iterar sobre ellos, ya que suelen tener múltiples niveles de anidamiento.

## 3. Scripts de Exploración de Datos
*   Cualquier script creado para inspeccionar datos (`.py`) no debe modificar ni borrar los datos originales nunca.
*   Los scripts deben ser modulares, bien comentados, y diseñados para que el usuario pueda modificarlos fácilmente para explorar diferentes variables o tetrodos.
