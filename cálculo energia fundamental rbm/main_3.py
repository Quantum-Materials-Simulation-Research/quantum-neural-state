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
import matplotlib.pyplot as plt
from rbm_3 import RBM, generate_configurations, train_rbm_sr
from openfermion import FermionOperator, jordan_wigner

# --- Carrega integrais do LiH ---
data = np.load("lih_integrals.npz")
h1 = data["h1"]
eri = data["eri"]
n_orb = int(data["n_orb"])
nelec = int(data["nelec"])

# --- Hamiltoniano em segunda quantização ---
H = FermionOperator()
for p in range(n_orb):
    for q in range(n_orb):
        coef = h1[p, q]
        if abs(coef) > 1e-12:
            H += FermionOperator(f"{p}^ {q}", coef)
for p in range(n_orb):
    for q in range(n_orb):
        for r in range(n_orb):
            for s in range(n_orb):
                coef = 0.5 * eri[p, q, r, s]
                if abs(coef) > 1e-12:
                    H += FermionOperator(f"{p}^ {q}^ {s} {r}", coef)
theta = jordan_wigner(H)

# --- Gera configurações ---
configs = generate_configurations(n_sites=n_orb, n_electrons=nelec)

# --- Inicializa RBM ---
rbm = RBM(n_visible=n_orb, n_hidden=n_orb)
np.random.seed(42)
rbm.a = np.random.normal(0, 0.01, size=n_orb)
rbm.b = np.random.normal(0, 0.01, size=n_orb)
rbm.W = np.random.normal(0, 0.01, size=(n_orb, n_orb))

# --- Treinamento com SR ---
energias = train_rbm_sr(rbm, configs, theta, epochs=50, lr=0.001)

fci_energy=-8.509289
# --- Gráfico ---
plt.plot(energias, label="Energy ⟨H⟩")
plt.axhline(fci_energy, linestyle="--", color="gray", label="FCI energy")
plt.text(len(energias) - 50, fci_energy + 0.01,
                 f"FCI = {fci_energy:.6f} Ha", color='gray', fontsize=9)
plt.xlabel("Epoch")
plt.ylabel("Energy (Ha)")
plt.title("RBM LiH energy relaxation with Stochastic Reconfiguration")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()