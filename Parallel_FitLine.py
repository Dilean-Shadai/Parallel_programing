from mpi4py import MPI
import numpy as np
from time import time
import math

# Nuestra Función para ajuste de línea utilizando metodo chi2
def fit_line(puntos, media_x):
    x = puntos[:, 0] #coordenadas de x
    y = puntos[:, 1] #coordenadas de y
    n = len(x)   #num de datos

    media_y = np.mean(y)
    varianza_x = np.sum((x - media_x) ** 2)
    # calculamos pendiente e intersección de la línea de ajuste
    pendiente = np.sum((x - media_x) * y) / varianza_x
    interseccion = media_y - media_x * pendiente
    chi2 = np.sum((y - interseccion - pendiente * x) ** 2) # Formula de chi2
    siga = math.sqrt((1.0 / n + media_x ** 2 / varianza_x) * chi2 / n) # desviación estandar de intersección
    sigb = math.sqrt(1.0 / varianza_x * chi2 / n) # desviacipn estandar pendiente

    return interseccion, pendiente, siga, sigb
#__________________________________________________________________________________________________________________

#---ini comunicador MPI
comm = MPI.COMM_WORLD
id_proceso = comm.Get_rank()
Tamaño_comm = comm.Get_size() #

if id_proceso == 0: #para que empice desde 0
    tiempo_inicio = time() # y ya contamos el tiempo de ejec

# Data
puntos_totales = np.array([
    [1, -5.256524853606793], [2, 6.382412887989748], [3, 6.858088240077303], [4, 9.74087800517334],
    [5, 4.890820007833353], [6, 21.49615805328267], [7, 14.254567020482917], [8, 13.244888558180876],
    [9, 17.024654712740035], [10, 19.049140289578286], [11, 25.51436128053622], [12, 27.82306260905608],
    [13, 28.81214727593773], [14, 29.93376186924023], [15, 37.24066620762991], [16, 35.67369856832711],
    [17, 35.11700884399834], [18, 24.57393380139963], [19, 41.965433681432464], [20, 30.175472960670396],
    [21, 38.458245032140604], [22, 36.00945496320289], [23, 53.01711554643736], [24, 44.201203258234],
    [25, 34.25179031223524], [26, 58.64665426941231], [27, 37.01456037466594], [28, 39.33678545074439],
    [29, 45.9342747714443], [30, 38.2926431913939], [31, 45.43083171221709], [32, 45.85757542287413],
    [33, 56.34930842313169], [34, 62.00557219761483], [35, 56.10026806443791], [36, 56.363498113973935],
    [37, 65.9313733720835], [38, 63.932226102760465], [39, 51.377283649727956], [40, 73.76818714547637]
])

media_x = np.mean(puntos_totales[:, 0]) #Calculamos la media de x

#--- hacemos Broadcast de la media de x a todos nuestros procesos del comunicador
media_x = comm.bcast(media_x, root=0) #

# División de los puntos entre los procesos
puntos_por_proceso = len(puntos_totales) // Tamaño_comm
puntos_sobrantes = len(puntos_totales) % Tamaño_comm

# cada proceso recibe una porción de los puntos de datos
'''los puntos se dividan de manera equitativa entre los procesos y
cuando hay un número que no nos es divisible de datos entre el número de procesos para el comm
nos aseguramos de una distribución uniforme o casi uniforme de los datos que le tocan a cada proceso'''

if id_proceso < puntos_sobrantes:
    inicio = id_proceso * (puntos_por_proceso + 1)
    fin = inicio + (puntos_por_proceso + 1)
else:
    inicio = puntos_sobrantes * (puntos_por_proceso + 1) + (id_proceso - puntos_sobrantes) * puntos_por_proceso
    fin = inicio + puntos_por_proceso

puntos_recibidos = puntos_totales[inicio:fin]

# Cada uno de nuestros procesos realiza el ajuste de línea con su parte de los puntos
resultados_locales = fit_line(puntos_recibidos, media_x)

#---El proceso raíz 0 recibe los resultados de todos los demas procesos
resultados_totales = comm.gather(resultados_locales, root=0)

# aqui nuestro proceso raíz imprimira los resultados de los calculos hechos
if id_proceso == 0:
    for i, resultado in enumerate(resultados_totales): #
        print(f"Resultados del proceso {i}:")
        if resultado is not None:
            print("Intersección:", resultado[0])
            print("Pendiente:", resultado[1])
            print("Desviación estándar para intersección (siga):", resultado[2])
            print("Desviación estándar para pendiente (sigb):", resultado[3])

    #
    tiempo_final = time()
    tiempo_total = tiempo_final - tiempo_inicio
    print("tiempo de ejecucion:", tiempo_total)
