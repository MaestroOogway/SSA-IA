import random as rnd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from copy import deepcopy

# reproducibilidad
SEED = 42
rnd.seed(SEED)
np.random.seed(SEED)

# constante usada en la transformación de fitness
C = 0.01

class Problem:
    """
    Modelado del problema (variables, límites, restricciones y función objetivo).
    Variables: x1..x5 (número de anuncios por canal)
    """
    def __init__(self):
        self.dimension = 5
        # límites máximos por variable 
        # orden: x1, x2, x3, x4, x5
        self.max_values = [15, 10, 25, 4, 30]
        # peso/alcance por variable (para restricción de cobertura / clientes)
        self.customer_weights = [1000, 2000, 1500, 2500, 300]
        # Otros parámetros de la restricción 
        # restricciones lineales:
        # x1 + x2 <= 20
        # 150*x1 + 300*x2 <= 1800
        # x2 == 0 or x3 == 0  (no pueden coexistir diarios y TV noche)
        self.coverage_limit = 50000

    def checkConstraint(self, x):
        """
        Devuelve True si x cumple todas las restricciones, False en caso contrario.
        """
        if len(x) != self.dimension:
            return False

        # comprobar límites por variable
        for i, xi in enumerate(x):
            if xi is None:
                return False
            if xi < 0 or xi > self.max_values[i]:
                return False

        x1, x2, x3, x4, x5 = x

        # restricciones específicas
        if (x1 + x2) > 20:
            return False
        if (150 * x1 + 300 * x2) > 1800:
            return False
        # la restricción "o" original interpretada como no coexistencia
        if (x2 != 0) and (x3 != 0):
            return False
        # restricción de cobertura (clientes potenciales)
        coverage = sum(w * xi for w, xi in zip(self.customer_weights, x))
        if coverage > self.coverage_limit:
            return False

        return True

    def eval(self, x):
        """
        Evaluación de la función objetivo. Retorna un valor numérico (cuanto mayor -> mejor).
        La expresión se deja similar al código original, pero se puede ajustar.
        """
        x1, x2, x3, x4, x5 = x
        # Nota: la fórmula original devolvía un valor (posible negativo).
        # Aseguramos que la función devuelva un float.
        value = (-0.0178591 * x1
                 -0.0431486 * x2
                 -0.000551901 * x3
                 -0.0088101 * x4
                 +0.00171961 * x5
                 + 0.6)
        return float(value)


class Spider:
    """
    Representa una araña/individuo del enjambre.
    Contiene la solución actual (vector x) y operaciones básicas.
    """
    def __init__(self, problem: Problem = None):
        self.p = problem if problem is not None else Problem()
        self.x = []
        # inicializar dentro de límites por variable
        for vmax in self.p.max_values:
            self.x.append(float(rnd.randint(0, vmax)))
        # asegurar factibilidad ocasionalmente (se validará en el Swarm)
    
    def isFeasible(self):
        return self.p.checkConstraint(self.x)

    def fit(self):
        return self.p.eval(self.x)

    def copy(self, other: "Spider"):
        """
        Copia el estado de otra araña (deep copy para evitar aliasing).
        """
        self.x = deepcopy(other.x)

    def isBetterThan(self, other: "Spider"):
        """
        Comparador robusto: transforma la fitness a una escala (sigmoide) y compara.
        Se asume que mayor fit es mejor.
        """
        eps = 1e-9
        f_self = self.fit()
        f_other = other.fit()
        # sigmoide centrada en C para mapear a (0,1)
        def vib(f):
            # protección contra overflow
            z = (f - C)
            if z > 700:
                return 1.0
            if z < -700:
                return 0.0
            return 1.0 / (1.0 + math.exp(-z))
        return vib(f_self) > vib(f_other)

    def move(self, g: "Spider", o: float, ra: float, pc: float, pm: float):
        """
        Mover la araña hacia la mejor global g con parámetros:
        o: medida de dispersión (std)
        ra: tasa de atenuación
        pc: probabilidad de cambio de máscara en cada dimensión
        pm: probabilidad de que la máscara sea 1 cuando se decide cambiar
        """
        # construir máscara 0/1 para cada dimensión
        mask = []
        for _ in range(len(self.x)):
            if rnd.random() < pc:
                mask.append(1 if rnd.random() < pm else 0)
            else:
                mask.append(0)

        # movimiento por dimensión
        for i in range(len(self.x)):
            direction = (g.x[i] - self.x[i]) * mask[i]
            # paso con componente aleatoria y con ruido proporcional a 'o'
            step = direction * rnd.uniform(0, 1) * ra + rnd.gauss(0, 0.05) * (o if o > 0 else 1.0)
            new_val = self.x[i] + step

            # reflejo / clamp a límites por variable
            vmax = self.p.max_values[i]
            if new_val < 0:
                new_val = 0.0
            elif new_val > vmax:
                new_val = vmax
            self.x[i] = float(new_val)

    def __str__(self):
        rounded = [int(round(v)) for v in self.x]
        return f"fit: {self.fit():.6f} x: {rounded}"

    def toListInt(self):
        return [int(round(v)) for v in self.x]


class Swarm:
    """
    Enjambre que contiene múltiples arañas y la lógica evolutiva.
    """
    def __init__(self, nSpiders: int = 5, maxIter: int = 30):
        self.maxIter = maxIter
        self.nSpiders = nSpiders
        self.swarm = []
        self.g = None  # mejor global (Spider)
        self.iterations = []
        self.best_fitness = []
        self.best_solutions = []

        # parámetros por defecto (pueden pasarse a solve)
        self.ra = 0.8
        self.pc = 0.5
        self.pm = 0.8

    def standard_deviation(self):
        """
        Desviación estándar promedio por dimensión (medida de dispersión).
        """
        if len(self.swarm) == 0:
            return 0.0
        pop = np.array([spider.x for spider in self.swarm], dtype=float)
        std_per_dim = np.std(pop, axis=0)
        return float(np.mean(std_per_dim))

    def solve(self, ra: float = None, pc: float = None, pm: float = None):
        """
        Ejecuta el algoritmo: inicializa y evoluciona.
        """
        if ra is not None:
            self.ra = ra
        if pc is not None:
            self.pc = pc
        if pm is not None:
            self.pm = pm

        print(f"Parametros -> ra: {self.ra}, pc: {self.pc}, pm: {self.pm}")
        self.initRand()
        self.evolve()

    def initRand(self):
        """
        Inicializar la población con soluciones factibles.
        """
        print("  -->  initRand  <-- ")
        self.swarm = []
        attempts_limit = 2000
        for i in range(self.nSpiders):
            attempts = 0
            while True:
                s = Spider()
                attempts += 1
                if s.isFeasible():
                    break
                if attempts >= attempts_limit:
                    # si no encontramos factible por alguna razón, forzamos una factible por clamping
                    # clamp each var to min(max, current)
                    for j in range(len(s.x)):
                        s.x[j] = min(max(s.x[j], 0.0), s.p.max_values[j])
                    if s.isFeasible():
                        break
            self.swarm.append(s)

        # inicializar mejor global (copia profunda)
        self.g = Spider()
        self.g.copy(self.swarm[0])
        for i in range(1, self.nSpiders):
            if self.swarm[i].isBetterThan(self.g):
                self.g.copy(self.swarm[i])

        self.swarmToConsole()
        self.bestToConsole()

    def evolve(self):
        """
        Bucle principal de evolución.
        """
        print("  -->  evolve  <-- ")
        t = 1
        while t <= self.maxIter:
            o = self.standard_deviation()
            # mover cada araña
            for i in range(self.nSpiders):
                a = Spider()
                a.copy(self.swarm[i])
                attempts = 0
                while True:
                    a.move(self.g, o, self.ra, self.pc, self.pm)
                    attempts += 1
                    # si es factible, lo aceptamos; si no lo conseguimos tras N intentos, descartamos
                    if a.isFeasible() or attempts >= 15:
                        break
                if a.isFeasible() and a.isBetterThan(self.swarm[i]):
                    self.swarm[i].copy(a)

            # actualizar mejor actual en la población
            best_in_pop = max(self.swarm, key=lambda s: s.fit())
            # guardar copia de la mejor encontrada en esta iteración
            best_copy = Spider()
            best_copy.copy(best_in_pop)
            self.best_solutions.append(best_copy)

            # actualizar mejor global self.g
            if best_in_pop.isBetterThan(self.g):
                self.g.copy(best_in_pop)

            # registrar fitness para la curva de convergencia
            current_best_fitness = best_in_pop.fit()
            if len(self.best_fitness) == 0:
                self.best_fitness.append(current_best_fitness)
            else:
                # guardar el mayor entre el histórico y el actual (monótono no decreciente)
                self.best_fitness.append(max(self.best_fitness[-1], current_best_fitness))

            self.iterations.append(t)

            # feedback por consola (compacto)
            print(f"Iter {t:02d} | best_pop_fit: {current_best_fitness:.6f} | global_best_fit: {self.g.fit():.6f}")
            t += 1

        print("Evolución finalizada.")
        self.swarmToConsole()
        self.bestToConsole()

    def plotConvergence(self):
        """
        Grafica la convergencia del mejor fitness a lo largo de las iteraciones.
        """
        plt.figure(figsize=(8, 4))
        plt.plot(self.iterations, self.best_fitness, marker='o')
        plt.xlabel("Iterations")
        plt.ylabel("Best Fitness (acumulado)")
        plt.title("Convergence Plot")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plotBoxPerVariable(self):
        """
        Boxplot por variable usando la población final.
        """
        data_points = [spider.toListInt() for spider in self.swarm]
        df = pd.DataFrame(data_points, columns=['TV Tarde', 'TV Noche', 'Diarios', 'Revistas', 'Radio'])
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=df)
        plt.xlabel('Tipo de Anuncio')
        plt.ylabel('Cantidad de Anuncios')
        plt.title('Distribución de Cantidades de Anuncios por Tipo (población final)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def swarmToConsole(self):
        print(" -- Swarm --")
        for i, spider in enumerate(self.swarm):
            rounded_values = [int(round(val)) for val in spider.x]
            print(f"[{i}] fit: {spider.fit():.6f} x: {rounded_values}")

    def bestToConsole(self):
        print(" -- Best Global --")
        if self.g is not None:
            print(f"fit: {self.g.fit():.6f} x: {[int(round(v)) for v in self.g.x]}")
        else:
            print("No hay solución global aún.")

    def get_best_quality_scores(self):
        """
        Devuelve las fitness de las mejores soluciones guardadas (lista de floats).
        """
        return [s.fit() for s in self.best_solutions]


if __name__ == "__main__":
    import sys

    MAX_SPIDERS = 30
    MAX_ITER = 100
    n_runs = 20

    # Validar argumentos
    if len(sys.argv) != 3:
        print("Uso: python SSA.py <num_arañas> <num_iteraciones>")
        sys.exit(1)

    try:
        nSpiders = int(sys.argv[1])
        maxIter = int(sys.argv[2])
    except ValueError:
        print("Error: los argumentos deben ser números enteros.")
        sys.exit(1)

    # Restricciones
    if nSpiders < 1 or nSpiders > MAX_SPIDERS:
        print(f"Error: la cantidad de arañas debe ser entre 1 y {MAX_SPIDERS}.")
        sys.exit(1)

    if maxIter < 1 or maxIter > MAX_ITER:
        print(f"Error: la cantidad de iteraciones debe ser entre 1 y {MAX_ITER}.")
        sys.exit(1)

    print(f"Número de arañas: {nSpiders}")
    print(f"Número de iteraciones: {maxIter}")
    print(f"El algoritmo se ejecutará {n_runs} veces\n")

    # parámetros del algoritmo
    ra = 0.5  # tasa de atenuación
    pc = 0.2  # probabilidad de cambio de máscara
    pm = 0.8  # probabilidad de 1 en la máscara

    # inicializar y ejecutar enjambre
    s = Swarm(nSpiders=nSpiders, maxIter=maxIter)
    s.solve(ra=ra, pc=pc, pm=pm)

    # graficas
    s.plotConvergence()
    s.plotBoxPerVariable()

    # Estadísticas descriptivas de las mejores soluciones por iteración
    quality_scores = s.get_best_quality_scores()
    if len(quality_scores) > 0:
        mejor = max(quality_scores)
        peor = min(quality_scores)
        promedio = np.mean(quality_scores)
        mediana = np.median(quality_scores)
        desviacion_estandar = np.std(quality_scores)
        rango_intercuartilico = np.percentile(quality_scores, 75) - np.percentile(quality_scores, 25)

        print("\nResumen estadístico sobre las mejores soluciones por iteración:")
        print(f"Mejor: {mejor:.6f}")
        print(f"Peor: {peor:.6f}")
        print(f"Promedio: {promedio:.6f}")
        print(f"Mediana: {mediana:.6f}")
        print(f"Desviación Estándar: {desviacion_estandar:.6f}")
        print(f"Rango Intercuartílico: {rango_intercuartilico:.6f}")
    else:
        print("No se tienen mejores soluciones registradas.")
