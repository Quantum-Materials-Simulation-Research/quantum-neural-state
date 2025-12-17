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

from pyscf import gto, scf, ao2mo
import numpy as np

# Define a molécula H₂ (distância internuclear ~0.74 Å)
mol = gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 0.734'
mol.basis = 'sto-3g'
mol.spin = 0  # número de elétrons alfa - beta
mol.charge = 0
mol.build()

# SCF (Hartree-Fock) calculation
mf = scf.RHF(mol)
mf.kernel()

# Número de orbitais espaciais
n_orb = mf.mo_coeff.shape[1]

# Matriz de um elétron no espaço de orbitais moleculares (MO)
h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff

# Integrais de dois elétrons transformados para base MO (antisymmetrized physicist notation)
eri = ao2mo.kernel(mol, mf.mo_coeff)
eri = ao2mo.restore(1, eri, n_orb)  # (pq|rs)

# Salva resultados
np.savez("h2_integrals.npz", h1=h1_mo, eri=eri, n_orb=n_orb, nelec=mol.nelectron)

print(f"Integrals salvos: {n_orb} orbitais, {mol.nelectron} elétrons.")