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
from itertools import combinations

# ----------------- RBM -----------------
class RBM:

    """Summary
    """
    
    def __init__(self, n_visible, n_hidden):
        """Summary
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.a = np.random.randn(n_visible) * 0.01
        self.b = np.random.randn(n_hidden) * 0.01
        self.W = np.random.randn(n_visible, n_hidden) * 0.01

    def psi(self, sigma):
        """Summary
        """
        preactivation = self.b + np.dot(sigma, self.W)
        preactivation = np.clip(preactivation, -10, 10)
        hidden_product = np.prod(2 * np.cosh(preactivation))
        a_dot = np.dot(self.a, sigma)
        a_dot = np.clip(a_dot, -20, 20)
        return np.exp(a_dot) * hidden_product

    def log_psi(self, sigma):
        """Summary
        """
        a_dot = np.dot(self.a, sigma)
        preactivation = self.b + np.dot(sigma, self.W)
        cosh_terms = np.log(2 * np.cosh(preactivation))
        return a_dot + np.sum(cosh_terms)

    def log_derivatives(self, sigma):
        """Summary
        """
        h_input = self.b + np.dot(sigma, self.W)
        tanh_h = np.tanh(h_input)
        return sigma, tanh_h, np.outer(sigma, tanh_h)

# ----------------- Configurações -----------------
def generate_configurations(n_sites, n_electrons):
    """Summary
    """
    configs = []
    for occ in combinations(range(n_sites), n_electrons):
        config = np.zeros(n_sites)
        config[list(occ)] = 1
        configs.append(config)
    return np.array(configs)
