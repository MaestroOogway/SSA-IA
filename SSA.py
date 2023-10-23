import pandas as pd
import random as rnd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

C = 0.01

class Problem:
  def __init__(self):

    self.dimension=5

  def checkConstraint(self, x):
    self.x1, self.x2, self.x3, self.x4, self.x5 = x
    if (self.x1 >= 0 and
      self.x2 >= 0 and
      self.x3 >= 0 and
      self.x4 >= 0 and
      self.x5 >= 0 and
      self.x1 <= 12 and
      self.x2 <= 6 and
      self.x3 <= 25 and
      self.x4 <= 4 and
      self.x5 <= 30 and
      self.x1 + self.x2 <= 20 and
      (150 * self.x1 + 300 * self.x2) <= 1800 and
      (self.x2 == 0 or self.x3 == 0)) :
      return True

  def eval(self, x):
    self.x1, self.x2, self.x3, self.x4, self.x5 = x
    self.max = -0.0178591* self.x1 - 0.0431486* self.x2 - 0.000551901* self.x3 - 0.0088101* self.x4 + 0.00171961* self.x5 + 0.6
    return self.max

class Spider(Problem):
  def __init__(self):#A la araña se le atribuye el problema
    self.p = Problem()
    self.x = []

    for j in range(self.p.dimension):
      self.x.append(rnd.randint(0,25)) # Cambiar, dependiendo del algoritmo

  def isFeasible(self):
    return self.p.checkConstraint(self.x)

  def isBetterThan(self, g):
    #Posicion Objetivo Actual
    vibration_intensity = math.log((1 / (self.fit() - C)) + 1)
    #Posicion Objetivo del agente
    g_vibration_intensity = math.log((1 / (g.fit() - C)) + 1)
    return (vibration_intensity < g_vibration_intensity) # La intensidad de la vibracion del mejor agente es mejor que la actual.

  def fit(self):
    return self.p.eval(self.x)

  def move(self, g , o):
    ra = 0.8  # Tasa de atenuación de vibración que se propaga en la red.
    pc = 0.5  # Probabilidad de cambio de máscara de dimensión en una caminata aleatoria.
    pm = 0.8  # Probabilidad de cada valor en la máscara de dimensión para ser uno.
    mask = [self.toBinary(pi) if rnd.random() < pc else mi for pi, mi in zip(g.x, self.x)]    # Determinar la máscara de dimensión para guiar el movimiento
    # Calcular el nuevo siguiente movimiento de la araña
    following_position = [gi + maski * (self.toBinary(rnd.randint(0, 1)) * ri - self.toBinary(rnd.randint(0, 1)) * xi)
                              for gi, maski, ri, xi in zip(g.x, mask, rnd.choices([-1, 1], k=len(g.x)), self.x)]
    # Realizar la caminata aleatoria y manejar las restricciones
    for i in range(len(self.x)):
      self.x[i] = g.x[i] + following_position[i] * rnd.uniform(0, 1) * ra
      # Aplicar el reflejo si la nueva posición viola alguna restricción
      if self.x[i] > 25:
        self.x[i] = 25 - rnd.uniform(0, 1) * ra
      elif self.x[i] < 0:
        self.x[i] = rnd.uniform(0, 1) * ra

  def toBinary(self, x):
    return 1 if (1 / (1 + math.pow(math.e, -(x)))) > rnd.random() else 0

  def __str__(self) -> str:
    return f"fit: {self.fit()} x: {self.x}"

  def copy(self, a):
    self.x = a.x.copy()

  def toString(self):
    return self.x

class Swarm:
  def __init__(self):
    self.maxIter = 30
    self.nSpiders = 5 # Agregar los parametros del algoritmo
    self.swarm = []
    self.g = Spider()
    self.iterations = []  # Lista para almacenar las iteraciones
    self.best_fitness = []  # Lista para almacenar el me
    self.best_solutions = []


  def standard_deviation(self):
    pop = [self.g.x for spider in self.swarm]
    return np.std(pop)

  def solve(self,ra,pc,pm):
    print(ra,pc,pm)
    self.initRand()
    self.evolve()

  def initRand(self):
    print("  -->  initRand  <-- ")
    for i in range(self.nSpiders):#Aqui se agregan las Arañas al enjambre.
      while True:
        s = Spider()
        if s.isFeasible():
          break
      self.swarm.append(s)

    self.g.copy(self.swarm[0])
    for i in range(1, self.nSpiders):
      if self.swarm[i].isBetterThan(self.g):
        self.g.copy(self.swarm[i])

    self.swarmToConsole()
    self.bestToConsole()

  def evolve(self):
    print("  -->  evolve  <-- ")
    o = self.standard_deviation()
    t = 1
    while t <= self.maxIter:
      for i in range(self.nSpiders):
        a = Spider()
        while True:
          a.copy(self.swarm[i])
          a.move(self.g, o)
          if a.isFeasible(): #Si a es factible
            break
        self.swarm[i].copy(a)
      best_solution = max(self.swarm, key=lambda spider: spider.fit())  # Encontrar la mejor solución actual
      self.best_solutions.append(best_solution)
      for i in range(self.nSpiders):
        if self.swarm[i].isBetterThan(self.g):
          self.g.copy(self.swarm[i])

      best_fitness = max(spider.fit() for spider in self.swarm)
      if len(self.best_fitness) == 0:
        self.best_fitness.append(best_fitness)
      else:
        if best_fitness > self.best_fitness[-1]:
          self.best_fitness.append(best_fitness)
        else:
          self.best_fitness.append(self.best_fitness[-1])
      self.iterations.append(t)


      self.swarmToConsole()
      self.bestToConsole()
      t = t + 1

  def plotConvergence(self):
    plt.figure()
    plt.plot(self.iterations, self.best_fitness, 'b-')
    plt.xlabel("Iterations")
    plt.ylabel("Best Fitness")
    plt.title("Convergence Plot")
    plt.show()

  def plotScatter(self):
    data_points = [(round(spider.x[0]), round(spider.x[1]), round(spider.x[2]), round(spider.x[3]), round(spider.x[4])) for spider in s.swarm]

    # Extraer los valores de X y Y en listas separadas para cada variable
    x1_values = [point[0] for point in data_points]
    x2_values = [point[1] for point in data_points]
    x3_values = [point[2] for point in data_points]
    x4_values = [point[3] for point in data_points]
    x5_values = [point[4] for point in data_points]
    data = pd.DataFrame({'TV Tarde': x1_values, 'TV Noche': x2_values, 'Diarios': x3_values, 'Revistas': x4_values, 'Radio': x5_values})
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.xlabel('Tipo de Anuncio')
    plt.ylabel('Cantidad de Anuncios')
    plt.title('Distribución de Cantidades de Anuncios por Tipo')
    plt.grid(True)
    plt.show()

  def swarmToConsole(self):
    print(" -- Swarm --")
    for i in range(self.nSpiders):
        rounded_values = [int(round(val)) for val in self.swarm[i].x]  # Redondear los valores a enteros
        print(f"fit: {round(self.swarm[i].fit(),4)} " f"x: {rounded_values}")
  def bestToConsole(self):
    print(" -- Best --")
    print(f"{round(self.g.fit(), 4)}")  # Redondear el

try:
  ra = 0.5  # Tasa de atenuación de vibración que se propaga en la red.
  pc = 0.2  # Probabilidad de cambio de máscara de dimensión en una caminata aleatoria.
  pm = 0.8  # Probabilidad de cada valor en la máscara de dimensión para ser uno.
  s = Swarm()
  s.solve(ra,pc,pm)
  s.plotConvergence()
  s.plotScatter()
  # Obtener las calificaciones de calidad de las mejores soluciones
  quality_scores = [best_solution.fit() for best_solution in s.best_solutions]

    # Calcular las medidas de resumen descriptivo
  mejor = max(quality_scores)
  peor = min(quality_scores)
  promedio = np.mean(quality_scores)
  mediana = np.median(quality_scores)
  desviacion_estandar = np.std(quality_scores)
  rango_intercuartilico = np.percentile(quality_scores, 75) - np.percentile(quality_scores, 25)
  print("Mejor:", mejor)
  print("Peor:", peor)
  print("Promedio:", promedio)
  print("Mediana:", mediana)
  print("Desviación Estándar:", desviacion_estandar)
  print("Rango Intercuartílico:", rango_intercuartilico)
except Exception as e:
  print(f"{e} \nCaused by {e.__cause__}")
