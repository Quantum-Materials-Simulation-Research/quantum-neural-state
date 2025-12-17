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
from pyscf import gto, scf, ao2mo

# 1. Define a molécula de LiH
mol = gto.Mole()
mol.atom = 'Li 0 0 0; H 0 0 1.6'  # distância típica em angstroms
mol.basis = 'sto-3g'
mol.spin = 0  # número de elétrons alfa - beta
mol.build()

# 2. Hartree-Fock
mf = scf.RHF(mol)
mf.kernel()

# 3. Integrais de 1 e 2 elétrons
n_orb = mf.mo_coeff.shape[1]  # número de orbitais moleculares
h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff  # integrais 1-elétron
eri = ao2mo.kernel(mol, mf.mo_coeff)              # 2-elétrons (formato compactado)
eri = ao2mo.restore(1, eri, n_orb)                 # restaurar formato (pqrs)

# 4. Número de elétrons
nelec = mol.nelectron

# 5. Salva os dados
np.savez("lih_integrals.npz", h1=h1, eri=eri, n_orb=n_orb, nelec=nelec)
print(f"✔️ Integrais de LiH (STO-3G) salvos em 'lih_integrals.npz'")