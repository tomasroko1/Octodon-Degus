import json
import os

notebook_path = r"c:\Users\tomas\OneDrive\Escritorio\Octodon-Degus\playground.ipynb"

# 1. Cargar el notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
new_cells = []

# Recorrer celdas y modificarlas
for i, cell in enumerate(cells):
    # Celda 1: Actualizar imports de scripts.utils.angulo_preferido
    if i == 1:
        source_text = "".join(cell["source"])
        target_import = (
            "from scripts.utils.angulo_preferido import (\n"
            "    calcular_angulo_preferido_coseno,\n"
            "    analizar_mirada_hacia_punto,\n"
            "    encontrar_punto_preferido_mirada,\n"
            "    evaluar_confianza_punto_preferido,\n"
            "    evaluar_confianza_angulo_preferido,\n"
            "    analizar_sintonizacion_perspectiva_continua,\n"
            "    evaluar_confianza_sintonizacion_perspectiva,\n"
            ")"
        )
        replacement_import = (
            "from scripts.utils.angulo_preferido import (\n"
            "    calcular_angulo_preferido_coseno,\n"
            "    evaluar_confianza_angulo_preferido,\n"
            "    analizar_sintonizacion_perspectiva_continua,\n"
            "    evaluar_confianza_sintonizacion_perspectiva,\n"
            ")"
        )
        if target_import in source_text:
            source_text = source_text.replace(target_import, replacement_import)
        else:
            # Fallback en caso de que los saltos de línea difieran levemente
            # Busquemos la parte y la reemplacemos de forma robusta
            import_start = source_text.find("from scripts.utils.angulo_preferido import (")
            if import_start != -1:
                import_end = source_text.find(")", import_start) + 1
                source_text = (
                    source_text[:import_start] +
                    "from scripts.utils.angulo_preferido import (\n"
                    "    calcular_angulo_preferido_coseno,\n"
                    "    evaluar_confianza_angulo_preferido,\n"
                    "    analizar_sintonizacion_perspectiva_continua,\n"
                    "    evaluar_confianza_sintonizacion_perspectiva,\n"
                    ")" +
                    source_text[import_end:]
                )
        cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in source_text.splitlines()]
        new_cells.append(cell)
        
    elif i == 15:
        # Renombrar Sección 6
        cell["source"] = [
            "## 6. Sintonización de Perspectiva Circular Continua (Ancla de Mirada)\n",
            "Identifica el punto óptimo de referencia (ancla) en la arena que maximiza la sintonización de la neurona a la dirección de la cabeza corregida por perspectiva. Este análisis sienta las bases matemáticas para modelar el Viewpoint en el GAM de forma continua."
        ]
        new_cells.append(cell)
        
    elif i == 16:
        # Mantener la carga de datos
        new_cells.append(cell)
        
    elif i in [17, 18, 19, 20]:
        # Eliminar las celdas del modelo de mirada discreto
        print(f"Eliminando celda {i}: {cell['cell_type']} - {cell['source'][0][:30]}...")
        
    elif i == 21:
        # Promover y renombrar la sección 6.2 a 6.1
        cell["source"] = [
            "### 6.1 Búsqueda y Validación de la Coordenada de Referencia Óptima\n",
            "Búsqueda exhaustiva en una grilla de 2D para encontrar el ancla espacial que maximiza la modulación de primer armónico complejo (proyección de coseno), validada estadísticamente con 100 repeticiones de spike shifting."
        ]
        new_cells.append(cell)
        
    else:
        # Mantener el resto de las celdas sin cambios
        new_cells.append(cell)

nb["cells"] = new_cells

# Guardar el notebook simplificado
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n¡Notebook modificado y simplificado con éxito!")
