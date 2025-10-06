Spider Swarm Optimizer

Breve:
Implementación en Python de un algoritmo tipo Spider Swarm para resolver un problema de asignación de anuncios (5 variables). El código incluye restricciones del problema, movimiento de individuos, control de factibilidad, visualizaciones y estadísticas básicas.

Requisitos

Python 3.8+

Paquetes: numpy, matplotlib, seaborn, pandas

Instalación rápida:

pip install numpy matplotlib seaborn pandas

Cómo ejecutar

Desde la terminal, pasando los parámetros deseados:

python SSA.py <num_arañas> <num_iteraciones>


Ejemplo:

python SSA.py 12 50


Esto ejecutará el algoritmo con 12 arañas y 50 iteraciones.

Restricciones: máximo 30 arañas y 100 iteraciones.

Si no se proporcionan argumentos válidos, se usan valores por defecto:

nSpiders = 8

maxIter = 40

Opcional: también puedes editar los parámetros dentro del bloque if __name__ == "__main__": en el archivo:

ra — tasa de atenuación de la propagación.

pc — probabilidad de cambiar máscara por dimensión.

pm — probabilidad de que la máscara sea 1 cuando se decide cambiar.

SEED — semilla para reproducibilidad.

Qué hace el código

Modela el problema con 5 variables (x1..x5), límites específicos por variable y estas restricciones:

x1 + x2 ≤ 20

150·x1 + 300·x2 ≤ 1800

x2 == 0 o x3 == 0 (no coexistencia)

Restricción de cobertura: suma(pesos·xi) ≤ 50000

Evalúa una función objetivo (lineal combinada con constantes — igual que en la versión original).

Inicializa un enjambre de arañas con soluciones factibles.

Itera el algoritmo:

Cada araña intenta moverse hacia la mejor global (g) respetando máscaras y parámetros.

Se controla factibilidad y se guarda la mejor solución de cada iteración.

Mantiene la mejor global.

Al final muestra:

Mensajes por consola con estado y mejores soluciones.

Gráfica de convergencia (mejor fitness por iteración).

Boxplot de la población final por variable.

Estadísticas descriptivas de las mejores soluciones por iteración.

Interpretación rápida de resultados

Convergence Plot: muestra la evolución del mejor fitness. Una curva plana y alta indica convergencia exitosa.

Boxplot por variable: visualiza la dispersión final de soluciones en cada canal (útil para ver si todas las arañas convergieron al mismo valor o hay diversidad).

Estadísticas: Mejor, Peor, Promedio, Mediana, Desviación estándar sobre las mejores soluciones a lo largo de las iteraciones.
