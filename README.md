Spider Swarm Optimizer

Breve: Implementación en Python de un algoritmo tipo spider swarm para resolver un problema de asignación de anuncios (5 variables). El código incluye restricciones del problema, movimiento de individuos, control de factibilidad, visualizaciones y estadísticas básicas.

Archivos

spider_swarm_clean.py — Código principal listo para ejecutar.

Requisitos

Python 3.8+

Paquetes: numpy, matplotlib, seaborn, pandas

Instalación rápida:

pip install numpy matplotlib seaborn pandas

Cómo ejecutar

Desde la terminal:

python SSA.py


Opcional: edita parámetros dentro del if __name__ == "__main__": (al inicio del bloque final):

ra — tasa de atenuación de la propagación.

pc — probabilidad de cambiar máscara por dimensión.

pm — probabilidad de que la máscara sea 1 cuando se decide cambiar.

nSpiders — tamaño del enjambre.

maxIter — número de iteraciones.

SEED — semilla para reproducibilidad (definida en la cabecera del archivo).

Qué hace el código

Modela el problema con 5 variables (x1..x5), límites específicos por variable y estas restricciones ejemplo:

x1 + x2 ≤ 20

150*x1 + 300*x2 ≤ 1800

x2 == 0 o x3 == 0 (no coexistencia)

restricción de cobertura: suma(pesos·xi) ≤ 50000

Evalúa una función objetivo (lineal combinada con constantes — igual que en la versión original).

Inicializa un enjambre de arañas con soluciones factibles.

Itera el algoritmo: cada araña intenta moverse hacia la mejor global (g) respetando máscaras y parámetros; se controla factibilidad.

Guarda la mejor solución de cada iteración y mantiene la mejor global.

Al final muestra:

Mensajes por consola con estado y mejores soluciones.

Gráfica de convergencia (mejor fitness por iteración).

Boxplot de la población final por variable.

Estadísticas descriptivas de las mejores soluciones por iteración.

Interpretación rápida de resultados

Convergence Plot: muestra la evolución del mejor fitness. Una curva plana y alta indica convergencia exitosa.

Boxplot por variable: visualiza la dispersión final de soluciones en cada canal (útil para ver si todas las arañas convergieron al mismo valor o hay diversidad).

Estadísticas: Mejor, Peor, Promedio, Mediana, Desviación estándar sobre las mejores soluciones a lo largo de las iteraciones.
