import matplotlib.pyplot as plt
import numpy as np


x = np.array([0, 5, 10, 15, 20, 25])
y = np.array([21, 39, 51, 63, 70, 90])

a = 18/7
b = 494/21

y_pred = a * x + b

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='red', label='Експериментальні точки', zorder=5)
plt.plot(x, y_pred, color='blue', label=f'Апроксимація: y = {a:.2f}x + {b:.2f}')

plt.title('Апроксимація методом найменших квадратів')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, linestyle='--')
plt.legend()
plt.show()