import numpy as np

def Cholesky(MatA):
    # Asumiendo que MatA es una matriz cuadrada y 2D
    dimN = MatA.shape[0]

    # Inicializar la matriz L
    MatL = np.zeros((dimN, dimN))

    # Paso 1. Determinar l_11 = sqrt(a_11)
    MatL[0, 0] = np.sqrt(MatA[0, 0])

    # Paso 2. Para j=2,...,n determine l_{j 1} = a_{j 1} / l_{11}
    for j in range(1, dimN):
        MatL[j, 0] = MatA[j, 0] / MatL[0, 0]

    # Paso 3. Para i=2,...,n-1 haga los pasos 4 y 5
    for i in range(1, dimN):
        # Paso 4. Determinar l_{i i} = (a_{i i} - sum_{k=1}^{i-1} l_{i k}^2)^(1/2)
        sum_k = np.sum(np.square(MatL[i, :i]))
        MatL[i, i] = np.sqrt(MatA[i, i] - sum_k)

        # Paso 5. Para j=i+1,...,n determine l_{j i} = (a_{j i} - sum_{k=1}^{i-1} l_{j k} * l_{i k}) / l_{i i}
        for j in range(i+1, dimN):
            sum_k = np.sum(MatL[j, :i] * MatL[i, :i])
            MatL[j, i] = (MatA[j, i] - sum_k) / MatL[i, i]

    # Paso 6 se cubre dentro del bucle de i, ya que i llega hasta n.

    # Paso 7. SALIDA MatL; PARE
    return MatL
