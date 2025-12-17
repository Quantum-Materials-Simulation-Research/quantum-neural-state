"""=================================================================================================================================================
**                                                   Copyright © 2025 Chanah Yocheved Bat Sarah                                                   **
**                                                                                                                                                **
**                                                       Author: Chanah Yocheved Bat Sarah                                                        **
**                                                          Contact: contact@chanah.dev                                                           **
**                                                                Date: 2025-05-25                                                                **
**                                                      License: Custom Attribution License                                                       **
**                                                                                                                                                **
**    Este projeto reúne temas da pesquisa de mestrado em física de materiais, com o objetivo de conseguir descrever a matéria usando métodos     **
**                                 computacionais do estado da arte, como computação quântica e machine learning.                                 **
**                                                                                                                                                **
**   Permission is granted to use, copy, modify, and distribute this file, provided that this notice is retained in full and that the origin of   **
**    the software is clearly and explicitly attributed to the original author. Such attribution must be preserved not only within the source     **
**       code, but also in any accompanying documentation, public display, distribution, or derived work, in both digital or printed form.        **
**                                                  For licensing inquiries: contact@chanah.dev                                                   **
====================================================================================================================================================
"""

import numpy as np
from openfermion import FermionOperator, jordan_wigner
from scipy.linalg import eigh
from itertools import combinations

# ------------------ Função para gerar configurações binárias ------------------
# Garante que as configurações foram geradas
def generate_binary_configs(n, k):
    """Summary
    """
    from itertools import combinations
    configs = []
    for occ in combinations(range(n), k):
        state = np.zeros(n, dtype=int)
        state[list(occ)] = 1
        configs.append(state)
    return np.array(configs)

# ------------------ Carrega os integrais ------------------
data = np.load("h2_integrals.npz")
h1 = data["h1"]
eri = data["eri"]
n_orb = int(data["n_orb"])
nelec = int(data["nelec"])

# ------------------ Cria Hamiltoniano ------------------
H = FermionOperator()

# Termos de 1 elétron
for p in range(n_orb):
    for q in range(n_orb):
        coef = h1[p, q]
        if abs(coef) > 1e-12:
            H += FermionOperator(f"{p}^ {q}", coef)

# Termos de 2 elétrons
for p in range(n_orb):
    for q in range(n_orb):
        for r in range(n_orb):
            for s in range(n_orb):
                coef = 0.5 * eri[p, q, r, s]
                if abs(coef) > 1e-12:
                    H += FermionOperator(f"{p}^ {q}^ {s} {r}", coef)

# ------------------ Aplica Jordan-Wigner ------------------
H_jw = jordan_wigner(H)

# ------------------ Gera configurações ------------------
configs = generate_binary_configs(4, 2)
H_proj = np.zeros((len(configs), len(configs)), dtype=complex)

# ------------------ Monta Hamiltoniano projetado ------------------
for i, sigma in enumerate(configs):
    for j, sigma_p in enumerate(configs):
        contrib = 0
        for term, coeff in H_jw.terms.items():
            new_sigma = sigma_p.copy()
            phase = 1.0 + 0j
            for qubit, op in term:
                if op == 'Z':
                    phase *= (-1)**int(new_sigma[qubit])
                elif op in ('X', 'Y'):
                    new_sigma[qubit] = 1 - new_sigma[qubit]
                    if op == 'Y':
                        phase *= 1j if sigma_p[qubit] == 0 else -1j
            if np.array_equal(new_sigma, sigma):
                contrib += coeff * phase
        H_proj[i, j] = contrib

# ------------------ Diagonaliza ------------------
eigvals, eigvecs = eigh(H_proj)

# ------------------ Resultado ------------------
print("Autovalores (energias dos estados):")
for i, val in enumerate(eigvals):
    print(f"Estado {i+1}: {val:.6f} Ha")

print(f"\n✨ Energia do estado fundamental: {eigvals[0]:.6f} Ha")


# ------------------ Identifica o estado fundamental ------------------
# Encontra o índice do menor autovalor
idx_min = np.argmin(eigvals)
energia_min = eigvals[idx_min]
estado_min = configs[idx_min]

# ------------------ Resultado ------------------
print(f"\n🌌 Estado fundamental encontrado:")
print(f"Configuração binária σ = {estado_min.astype(int)}")
print(f"Energia correspondente: {energia_min:.6f} Ha")