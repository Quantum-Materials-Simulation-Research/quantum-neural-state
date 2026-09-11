import csv
import multiprocessing
from pathlib import Path
from pyscf import gto, scf, fci
from filelock import FileLock
import os
import multiprocessing
import time

# Dados das moléculas
moleculas = [
    ("H2", "H 0 0 0; H 0 0 0.74", 0, 0),
    ("LiH", "Li 0 0 0; H 0 0 1.6", 0, 0),
    ("H2O", "O 0 0 0; H 0.7586 0.0 0.5043; H -0.7586 0.0 0.5043", 0, 0),
    ("CH2", "C 0 0 0; H 0 0 1.1; H 1.1 0 0", 0, 0),
    ("BeH2", "Be 0 0 0; H 0 0 1.3; H 0 0 -1.3", 0, 0),
    ("NH3", "N 0 0 0; H 0 0.94 0.34; H -0.81 -0.47 0.34; H 0.81 -0.47 0.34", 0, 0),
    ("CH4", "C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629", 0, 0),
    ("C2", "C 0 0 0; C 0 0 1.25", 0, 2),
    ("F2", "F 0 0 0; F 0 0 1.41", 0, 0),
    ("N2", "N 0 0 0; N 0 0 1.1", 0, 0),
    ("O2", "O 0 0 0; O 0 0 1.2", 0, 2),
    ("LiF", "Li 0 0 0; F 0 0 1.5", 0, 0),
    ("HCl", "H 0 0 0; Cl 0 0 1.27", 0, 0),
    ("H2S", "S 0 0 0; H 0.96 0 0.3; H -0.96 0 0.3", 0, 0),
    ("CH2O", "C 0 0 0; H 0.0 0.0 1.1; O 1.2 0.0 0.0; H -0.9 0.9 0.0", 0, 0),
    ("PH3", "P 0 0 0; H 0.96 0 0.3; H -0.96 0 0.3; H 0 0.96 -0.3", 0, 0),
    ("LiCl", "Li 0 0 0; Cl 0 0 2.0", 0, 0),
    ("CH4O", "C 0 0 0; O 1.43 0 0; H 0.63 0.9 0; H -0.63 0.9 0; H 0.63 -0.9 0; H -0.63 -0.9 0", 0, 0),
    # 21. Lithium Oxide (Li2O): 3 + 3 + 8 = 14 elétrons → par → spin 0
    ("Li2O", "Li 0.000 0.000 0.000; O 0.000 0.000 1.75; Li 0.000 0.000 3.50", 0, 0),

    # 22. Ethylene Oxide (C2H4O): 6*2 + 4*1 + 8 = 24 elétrons → par → spin 0
    ("C2H4O", "C -0.676 0.000 0.000; C 0.676 0.000 0.000; O 0.000 1.000 0.000; H -1.213 0.000 0.943; H -1.213 0.000 -0.943; H 1.213 0.000 0.943; H 1.213 0.000 -0.943", 0, 0),

    # 23. Propene (C3H6): 6*3 + 1*6 = 24 elétrons → par → spin 0
    ("C3H6", "C -0.667 0.000 0.000; C 0.667 0.000 0.000; C 1.877 0.000 0.000; H -1.221 0.935 0.000; H -1.221 -0.935 0.000; H 2.421 0.935 0.000; H 2.421 -0.935 0.000; H 0.667 0.000 1.100; H 0.667 0.000 -1.100", 0, 0),

    # 24. Acetic Acid (C2H4O2): 6*2 + 1*4 + 8*2 = 30 elétrons → par → spin 0
    ("C2H4O2", "C 0.000 0.000 0.000; C 1.540 0.000 0.000; O 2.140 1.200 0.000; O 2.140 -1.200 0.000; H -0.500 0.900 0.000; H -0.500 -0.900 0.000; H 1.540 0.000 1.090; H 1.540 0.000 -1.090", 0, 0),

    # 25. Sulfuric Acid (H2SO4): 1*2 + 16 + 8*4 = 2 + 16 + 32 = 50 elétrons → par → spin 0
    ("H2SO4", "S 0.000 0.000 0.000; O 1.440 0.000 0.000; O -1.440 0.000 0.000; O 0.000 1.440 0.000; O 0.000 -1.440 0.000; H 1.800 0.000 0.900; H -1.800 0.000 0.900", 0, 0),

    # 26. Sodium Carbonate (CNa2O3): 6 + 11*2 + 8*3 = 6 + 22 + 24 = 52 elétrons → par → spin 0
    ("CNa2O3", "C 0.000 0.000 0.000; O 1.200 0.000 0.000; O -1.200 0.000 0.000; O 0.000 1.200 0.000; Na 2.200 0.000 0.000; Na -2.200 0.000 0.000", 0, 0)
]

OUTPUT_CSV = "resultados_fci_parallel.csv"
LOCK_FILE = "resultados_fci_parallel.lock"

# Função que calcula a energia FCI de uma molécula
def calcula_fci(args):
    nome, atom_str, carga, spin = args
    inicio = time.time()

    try:
        mol = gto.Mole()
        mol.atom = atom_str
        mol.basis = 'sto-3g'
        mol.charge = carga
        mol.spin = spin
        mol.build()

        mf = scf.RHF(mol)
        hf_energy = mf.kernel()

        cisolver = fci.FCI(mol, mf.mo_coeff)
        cisolver.conv_tol = 1e-12
        fci_energy, _ = cisolver.kernel()
        
        print(f"✅ {nome}calculado com sucesso")
        fim = time.time()
        duracao = round(fim - inicio, 2)

        with FileLock(LOCK_FILE):
            file_exists = Path(OUTPUT_CSV).exists()
            with open(OUTPUT_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Molecule", "HF Energy", "FCI Energy", "Tempo (s)"])
                writer.writerow([nome, f"{hf_energy:.8f}", f"{fci_energy:.8f}", duracao])

    except Exception as e:
        print(f"❌ Erro ao calcular {nome}: {e}")

# Execução paralela
if __name__ == "__main__":
    print("CPUs disponiveis para esse processo: ", os.cpu_count(), flush=True)
    print("Multiprocessing detecta:", multiprocessing.cpu_count(), flush=True)
    with multiprocessing.Pool(processes=len(moleculas)) as pool:
        pool.map(calcula_fci, moleculas)
