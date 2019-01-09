from matplotlib import pyplot as plt
import numpy as np


x = np.linspace(-3,3, num=100)
yd_tanh = 1-(np.tanh(x))**2
yd_sigmoid = (1./(1.+np.exp(-x)))*(1-1./(1.+np.exp(-x)))
yd_relu = (x>0).astype(np.float32)

plt.rcParams.update({'font.size': 20})

fig, ax = plt.subplots(figsize=(10,6))

ax.plot(x, yd_tanh, label='tanh')
ax.plot(x, yd_sigmoid, label='sigmoid')
ax.step(x, yd_relu, label='ReLU', )
ax.set_ylabel("a'(x)")
ax.set_xlabel('x')
plt.legend(title='Activation Function')
plt.savefig('../writing/figures/activation_functions_derivative.pdf', bbox_inches='tight')