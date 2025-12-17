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

import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np

# Dados sintéticos
X = np.linspace(0, np.pi, 20)
X_norm = qml.numpy.array(X / np.pi)                  # entrada normalizada
Y = qml.numpy.array(2 * X + 1)                       # saída esperada

# Dispositivo com 1 qubit
dev = qml.device("default.qubit", wires=1)

# Circuito quântico variacional
@qml.qnode(dev)
def circuit(x, weights):
    """Summary
    """
    qml.RX(x * np.pi, wires=0)      # codificação da entrada
    qml.RY(weights[0], wires=0)     # parâmetro treinável
    return qml.expval(qml.PauliZ(0))  # ⟨Z⟩ ∈ [-1, 1]

# Modelo escalonado para saída linear
scale = -4.0
shift = 5.0

def model(x, weights):
    """Summary
    """
    return scale * circuit(x, weights) + shift

# Função custo
def cost(weights):
    """Summary
    """
    predictions = qml.numpy.array([model(x, weights) for x in X_norm])
    return qml.numpy.mean((predictions - Y) ** 2)

# Inicialização do parâmetro
weights = qml.numpy.array([0.1], requires_grad=True)

# Otimizador
opt = qml.optimize.NesterovMomentumOptimizer(stepsize=0.1)
losses = []

# Treinamento
for i in range(100):
    weights = opt.step(cost, weights)
    l = cost(weights)
    losses.append(l)
    if i % 10 == 0:
        print(f"Iteraction {i:3d} | Loss: {l:.6f} | Weight: {weights[0]:.4f}")

# Predição final
preds = [model(x, weights) for x in X_norm]

# Plot do resultado
plt.figure(figsize=(8, 4))
plt.plot(X, Y, "o", label="Target data")
plt.plot(X, preds, "-", label="Learnded Function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression Using a Variational Quantum Circuit")
plt.grid(False)
plt.tight_layout()
plt.show()