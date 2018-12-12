from matplotlib import pyplot as plt
import numpy as np


x = np.linspace(-3,3)
y_tanh = np.tanh(x)
y_sigmoid = 1. / (1. + np.exp(-x))
y_relu = np.clip(x, 0, None)

fig, ax = plt.subplots(figsize=(10,6))

ax.plot(x, y_tanh, label='tanh')
ax.plot(x, y_sigmoid, label='sigmoid')
ax.plot(x, y_relu, label='ReLU')
ax.set_ylabel('a(x)')
ax.set_xlabel('x')
plt.legend(title='Activation Function')
plt.savefig('../writing/figures/activation_functions.pdf', bbox_inches='tight')