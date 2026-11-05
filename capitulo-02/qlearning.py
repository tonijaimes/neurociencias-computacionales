import numpy as np
import matplotlib.pyplot as plt

NUM_ESTADOS = 5
NUM_ACCIONES = 2

IZQUIERDA = 0
DERECHA = 1

ALPHA = 0.1
GAMMA = 0.9

EPISODIOS = 500
MAX_PASOS = 100

EPSILONS = [0.0, 0.1, 0.5]


def ejecutar_accion(estado, accion):
    if accion == IZQUIERDA:
        siguiente = max(0, estado - 1)
    else:
        siguiente = min(NUM_ESTADOS - 1, estado + 1)

    terminado = siguiente == NUM_ESTADOS - 1
    recompensa = 10 if terminado else -1

    return siguiente, recompensa, terminado


def seleccionar_accion(Q, estado, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(NUM_ACCIONES))

    valores = Q[estado]
    mejores = np.flatnonzero(np.isclose(valores, valores.max()))

    return int(rng.choice(mejores))


def entrenar(epsilon, semilla=42):
    rng = np.random.default_rng(semilla)

    Q = np.zeros((NUM_ESTADOS, NUM_ACCIONES))

    recompensas = []
    pasos_episodio = []
    exitos = []

    for episodio in range(EPISODIOS):
        estado = 0
        recompensa_total = 0
        terminado = False

        for paso in range(MAX_PASOS):
            accion = seleccionar_accion(Q, estado, epsilon, rng)

            siguiente, recompensa, terminado = ejecutar_accion(
                estado, accion
            )

            if terminado:
                valor_futuro = 0
            else:
                valor_futuro = np.max(Q[siguiente])

            error_td = (
                recompensa
                + GAMMA * valor_futuro
                - Q[estado, accion]
            )

            Q[estado, accion] += ALPHA * error_td

            recompensa_total += recompensa
            estado = siguiente

            if terminado:
                break

        recompensas.append(recompensa_total)
        pasos_episodio.append(paso + 1)
        exitos.append(terminado)

    return Q, np.array(recompensas), np.array(pasos_episodio), np.array(exitos)


def evaluar_politica(Q):
    estado = 0
    trayectoria = [estado]
    recompensa_total = 0

    for _ in range(MAX_PASOS):
        # Para evaluar, se elige la mejor acción sin exploración.
        accion = int(np.argmax(Q[estado]))

        estado, recompensa, terminado = ejecutar_accion(estado, accion)

        trayectoria.append(estado)
        recompensa_total += recompensa

        if terminado:
            return trayectoria, recompensa_total, True

    return trayectoria, recompensa_total, False


resultados = {}

for epsilon in EPSILONS:
    Q, recompensas, pasos, exitos = entrenar(
        epsilon=epsilon,
        semilla=42
    )

    trayectoria, recompensa_eval, objetivo = evaluar_politica(Q)

    resultados[epsilon] = {
        "Q": Q,
        "recompensas": recompensas,
        "pasos": pasos,
        "exitos": exitos
    }

    print(f"\nEpsilon = {epsilon}")
    print("Tabla Q:")
    print(np.round(Q, 3))

    print("Recompensa media, últimos 100 episodios:",
          np.mean(recompensas[-100:]))

    print("Episodios exitosos:", np.sum(exitos), "/", EPISODIOS)
    print("Trayectoria de evaluación:", trayectoria)
    print("Recompensa de evaluación:", recompensa_eval)
    print("¿Alcanzó el objetivo?", objetivo)


# Gráfica 1: recompensa acumulada por episodio
plt.figure(figsize=(10, 5))

for epsilon in EPSILONS:
    recompensas = resultados[epsilon]["recompensas"]

    plt.plot(
        recompensas,
        alpha=0.45,
        label=f"epsilon = {epsilon}"
    )

plt.xlabel("Episodio")
plt.ylabel("Recompensa acumulada")
plt.title("Aprendizaje por refuerzo: comparación de exploración")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Gráfica 2: media móvil de las recompensas
plt.figure(figsize=(10, 5))

VENTANA = 20

for epsilon in EPSILONS:
    recompensas = resultados[epsilon]["recompensas"]

    media = np.convolve(
        recompensas,
        np.ones(VENTANA) / VENTANA,
        mode="valid"
    )

    plt.plot(
        np.arange(VENTANA, EPISODIOS + 1),
        media,
        label=f"epsilon = {epsilon}"
    )

plt.xlabel("Episodio")
plt.ylabel("Recompensa media")
plt.title("Evolución del aprendizaje (media móvil)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()