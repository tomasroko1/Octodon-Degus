import json

with open("playground.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Modificar importaciones en la celda 1 (índice 1 es el código de imports)
cell_imports = nb["cells"][1]
source = cell_imports["source"]

new_source = []
i = 0
while i < len(source):
    line = source[i]
    if "from scripts.utils.angulo_preferido import (" in line:
        new_source.append(line)
        new_source.append("    calcular_angulo_preferido_coseno,\n")
        new_source.append("    analizar_mirada_hacia_punto,\n")
        new_source.append("    encontrar_punto_preferido_mirada,\n")
        new_source.append("    evaluar_confianza_punto_preferido,\n")
        new_source.append("    evaluar_confianza_angulo_preferido,\n")
        # Saltar las líneas de importación antiguas que correspondían a esto
        while i < len(source) and ")" not in source[i]:
            i += 1
        new_source.append(")\n")
    else:
        new_source.append(line)
    i += 1

cell_imports["source"] = new_source

# 2. Definir las dos nuevas celdas
markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 6.1 Test de Control de Significancia Estadística (Spike Shifting)\n",
        "Para validar científicamente si la sintonización del ángulo preferido (Head Direction) y de la mirada (Viewpoint) son estadísticamente significativos o si son producto del azar, realizamos un análisis de control mediante el **desplazamiento circular de los spikes** (*spike shifting*).\n",
        "\n",
        "Este test:\n",
        "1. Desplaza circularmente el vector de spikes por una cantidad de tiempo aleatoria (mínimo de 10 segundos de cada lado para evitar solapamientos inmediatos).\n",
        "2. Rompe la correlación temporal exacta entre los eventos de disparo de la neurona y el comportamiento en tiempo real del animal (dirección de cabeza y posición en la grilla).\n",
        "3. Preserva de manera idéntica todas las propiedades intrínsecas de disparo de la célula (bursting, tasa de disparo promedio, autocorrelación temporal).\n",
        "\n",
        "Repitiendo este procedimiento $N$ veces (por ejemplo, 100 veces), construimos una **distribución nula** para evaluar si nuestra selectividad/amplitud experimental es significativamente mayor de lo esperado por mero azar."
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ==========================================\n",
        "# 1. EVALUAR CONFIANZA DE HEAD DIRECTION (HD)\n",
        "# ==========================================\n",
        "print(\"Ejecutando test de control para Head Direction (100 desplazamientos)... (esto tomará unos segundos)\")\n",
        "confianza_hd = evaluar_confianza_angulo_preferido(\n",
        "    ang_bins_deg, conteo_spikes, tiempos_bin=bin_size, tolerancia_deg=30, n_shifts=100, min_shift_sec=10.0\n",
        ")\n",
        "\n",
        "print(\"\\n=== RESULTADOS CONFIANZA HEAD DIRECTION ===\")\n",
        "print(f\"Amplitud Original: {confianza_hd['original_amplitude']:.4f} Hz\")\n",
        "print(f\"Media Distribución Nula: {confianza_hd['mean_shuffled_amplitude']:.4f} Hz ± {confianza_hd['std_shuffled_amplitude']:.4f} Hz\")\n",
        "print(f\"Z-Score: {confianza_hd['z_score']:.2f}\")\n",
        "print(f\"P-Valor: {confianza_hd['p_value']:.4f}\")\n",
        "print(f\"¿Es significativo?: {'SÍ (p < 0.05)' if confianza_hd['confianza_significativa'] else 'NO (p >= 0.05)'}\")\n",
        "\n",
        "# ==========================================\n",
        "# 2. EVALUAR CONFIANZA DE MIRADA (VIEWPOINT)\n",
        "# ==========================================\n",
        "print(\"\\nEjecutando test de control para Viewpoint de Mirada (100 desplazamientos)... (esto tomará unos segundos)\")\n",
        "confianza_gaze = evaluar_confianza_punto_preferido(\n",
        "    x_bins, y_bins, ang_bins_rad, spikes_gaze, tiempos_bin=bin_size, \n",
        "    grid_res=25, tolerancia_deg=30, n_shifts=100, min_shift_sec=10.0\n",
        ")\n",
        "\n",
        "print(\"\\n=== RESULTADOS CONFIANZA VIEWPOINT (MIRADA) ===\")\n",
        "print(f\"Coordenada Preferida: {confianza_gaze['original_coord']}\")\n",
        "print(f\"Selectividad ΔR Original: {confianza_gaze['original_delta_R']:.4f} Hz\")\n",
        "print(f\"Media Distribución Nula: {confianza_gaze['mean_shuffled_delta_R']:.4f} Hz ± {confianza_gaze['std_shuffled_delta_R']:.4f} Hz\")\n",
        "print(f\"Z-Score: {confianza_gaze['z_score']:.2f}\")\n",
        "print(f\"P-Valor: {confianza_gaze['p_value']:.4f}\")\n",
        "print(f\"¿Es significativo?: {'SÍ (p < 0.05)' if confianza_gaze['confianza_significativa'] else 'NO (p >= 0.05)'}\")\n",
        "\n",
        "# ==========================================\n",
        "# 3. VISUALIZACIÓN DE AMBAS DISTRIBUCIONES NULAS\n",
        "# ==========================================\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))\n",
        "\n",
        "# Panel izquierda: HD Amplitudes\n",
        "sns.histplot(confianza_hd['shuffled_amplitudes'], ax=ax1, color='gray', alpha=0.6, kde=True, label='Shuffled (Nula)')\n",
        "ax1.axvline(confianza_hd['original_amplitude'], color='red', linestyle='--', linewidth=2.5, \n",
        "            label=f'Original: {confianza_hd[\"original_amplitude\"]:.3f} Hz\\np={confianza_hd[\"p_value\"]:.3f} (Z={confianza_hd[\"z_score\"]:.1f})')\n",
        "ax1.set_title('Test de Control: Amplitud de Head Direction', fontweight='bold')\n",
        "ax1.set_xlabel('Amplitud del Ajuste Coseno (Hz)')\n",
        "ax1.set_ylabel('Frecuencia')\n",
        "ax1.legend()\n",
        "\n",
        "# Panel derecha: Viewpoint Delta R\n",
        "sns.histplot(confianza_gaze['shuffled_delta_Rs'], ax=ax2, color='gray', alpha=0.6, kde=True, label='Shuffled (Nula)')\n",
        "ax2.axvline(confianza_gaze['original_delta_R'], color='red', linestyle='--', linewidth=2.5, \n",
        "            label=f'Original: {confianza_gaze[\"original_delta_R\"]:.3f} Hz\\np={confianza_gaze[\"p_value\"]:.3f} (Z={confianza_gaze[\"z_score\"]:.1f})')\n",
        "ax2.set_title('Test de Control: Selectividad de Mirada (ΔR)', fontweight='bold')\n",
        "ax2.set_xlabel('Selectividad Máxima ΔR (Hz)')\n",
        "ax2.set_ylabel('Frecuencia')\n",
        "ax2.legend()\n",
        "\n",
        "plt.suptitle(f'Validación de Preferencias mediante Spike Shifting (100 shuffles, shift > 10s)', fontsize=14, fontweight='bold', y=0.98)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
}

# 3. Insertar las celdas después de la celda 17
nb["cells"].insert(18, markdown_cell)
nb["cells"].insert(19, code_cell)

# 4. Guardar los cambios en el archivo del notebook
with open("playground.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("¡Modificaciones aplicadas con éxito a playground.ipynb!")
