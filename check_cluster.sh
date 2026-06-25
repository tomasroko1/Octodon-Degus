#!/bin/bash
# =============================================
# DIAGNÓSTICO RÁPIDO
# Pegá esto en la terminal de la máquina Linux
# =============================================

echo ""
echo "========== 1. QUIEN SOY Y DONDE ESTOY =========="
echo "Usuario: $(whoami)"
echo "Hostname: $(hostname)"
echo "Home: $HOME"
echo ""

echo "========== 2. MIS CARPETAS =========="
echo "--- /data ---"
ls -la /data/ 2>/dev/null | head -15
echo ""
echo "--- /mnt/NAS (tu carpeta) ---"
# Mostrá las carpetas de primer nivel del NAS
ls -la /mnt/NAS/ 2>/dev/null | head -20
echo ""

echo "========== 3. DATOS DE DEGUS =========="
echo "--- merged_files ---"
ls /mnt/NAS/Degus/merged_files/*_merged.db 2>/dev/null | head -15
echo "Total merged: $(ls /mnt/NAS/Degus/merged_files/*_merged.db 2>/dev/null | wc -l)"
echo ""
echo "--- AllData2.db ---"
find /mnt/NAS -name "AllData2.db" 2>/dev/null
echo ""

echo "========== 4. PYTHON =========="
which python3 2>/dev/null || which python 2>/dev/null || echo "NO HAY PYTHON"
python3 --version 2>/dev/null || python --version 2>/dev/null
python3 -c "import numpy; print('numpy:', numpy.__version__)" 2>/dev/null || echo "numpy: NO"
python3 -c "import scipy; print('scipy:', scipy.__version__)" 2>/dev/null || echo "scipy: NO"
python3 -c "import h5py; print('h5py:', h5py.__version__)" 2>/dev/null || echo "h5py: NO"
python3 -c "import pandas; print('pandas:', pandas.__version__)" 2>/dev/null || echo "pandas: NO"
echo ""

echo "========== 5. MATLAB =========="
which matlab 2>/dev/null || echo "matlab: NO en PATH"
echo ""

echo "========== 6. SISTEMA DE COLAS (CLUSTER) =========="
which sbatch 2>/dev/null && echo "→ SLURM detectado" || echo "sbatch: no"
which qsub 2>/dev/null && echo "→ PBS/SGE detectado" || echo "qsub: no"
echo ""

echo "========== 7. ESPACIO EN DISCO =========="
df -h /data 2>/dev/null | tail -1
df -h /mnt/NAS 2>/dev/null | tail -1
df -h $HOME 2>/dev/null | tail -1
echo ""

echo "========== FIN =========="
