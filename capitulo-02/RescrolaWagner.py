import numpy as np
import matplotlib.pyplot as plt


def simular_rescorla_wagner(n_ensayos, alpha, valor_inicial=0.0, lambda_=1.0):
    valores = [valor_inicial]
    errores = []

    for _ in range(n_ensayos):
        error = lambda_ - valores[-1]
        nuevo_valor = valores[-1] + alpha * error

        errores.append(error)
        valores.append(nuevo_valor)

    return np.array(valores), np.array(errores)


# 1. Adquisición
valores, errores = simular_rescorla_wagner(30, alpha=0.2)

plt.figure()
plt.plot(range(31), valores, marker="o", markersize=3)
plt.xlabel("Ensayo")
plt.ylabel("Fuerza asociativa")
plt.title("Adquisición del condicionamiento")
plt.grid(True)
plt.show()


# 2. Comparación de tasas de aprendizaje
plt.figure()

for alpha in [0.05, 0.2, 0.8]:
    valores, _ = simular_rescorla_wagner(30, alpha)
    plt.plot(range(31), valores, label=f"alpha = {alpha}")

plt.xlabel("Ensayo")
plt.ylabel("Fuerza asociativa")
plt.title("Efecto de la tasa de aprendizaje")
plt.legend()
plt.grid(True)
plt.show()


# 3. Adquisición y extinción
adquisicion, _ = simular_rescorla_wagner(30, alpha=0.2, lambda_=1.0)

extincion, _ = simular_rescorla_wagner(
    20,
    alpha=0.2,
    valor_inicial=adquisicion[-1],
    lambda_=0.0
)

# Evitar duplicar el valor del ensayo 30
trayectoria = np.concatenate([adquisicion, extincion[1:]])

plt.figure()
plt.plot(range(51), trayectoria)
plt.axvline(30, linestyle="--", label="Inicio de extinción")
plt.xlabel("Ensayo")
plt.ylabel("Fuerza asociativa")
plt.title("Adquisición y extinción")
plt.legend()
plt.grid(True)
plt.show()