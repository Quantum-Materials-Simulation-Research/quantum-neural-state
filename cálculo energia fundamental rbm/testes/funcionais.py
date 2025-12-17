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

from pyscf import gto, dft
import pandas as pd

# Define a molécula de água com base STO-3G
mol = gto.Mole()
mol.atom = 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587'
mol.basis = 'sto-3g'
mol.build()

# Lista de funcionais para comparar
functionals = ['lda', 'pbe', 'b3lyp']
energies = {}

# Roda os cálculos para cada funcional
for xc in functionals:
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.kernel()
    energies[xc] = mf.e_tot

# Mostra os resultados
df = pd.DataFrame.from_dict(energies, orient='index', columns=['Energia Total (Hartree)'])
df.index.name = 'Funcional XC'
print(df.sort_values(by='Energia Total (Hartree)'))