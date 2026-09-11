#!/bin/bash
#SBATCH --job-name=fci_parallel
#SBATCH --nodes=1
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --ntasks=18
#SBATCH --time=03:00:00

echo "Criando ambiente virtual..."
python3 -m venv venv_fci
source venv_fci/bin/activate

echo "Instalando dependências..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Iniciando cálculos FCI paralelos..."
python calcular_fci_paralelo.py

echo "Limpando ambiente virtual..."
deactivate
rm -rf venv_fci

echo "Job finalizado em $(date)"
