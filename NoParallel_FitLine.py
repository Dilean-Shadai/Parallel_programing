import math

def fit_line(puntos): 
    
    x, y = zip(*puntos) 
    n = float(len(x)) 
    media_x = sum(x) / n 
    media_y = sum(y) / n
    varianza_x = sum([(xi - media_x) ** 2 for xi in x]) 
    pendiente = sum([(xi - media_x) * yi for xi, yi in zip(x, y)]) / varianza_x 
    interseccion = media_y - media_x * pendiente 
    chi2 = sum([(yi - interseccion - pendiente * xi) ** 2 for xi, yi in zip(x, y)])
    siga = math.sqrt((1.0 / n + media_x ** 2 / varianza_x) * chi2 / n)
    sigb = math.sqrt(1.0 / varianza_x * chi2 / n)
    
    return interseccion, pendiente, siga, sigb

resultado = fit_line([(0.,1.),(1.,2.1),(2.,2.9),(3.,4.)]) 

print("Intersección :", resultado[0])
print("Pendiente :", resultado[1])
print("Desviación estándar para interseccion (siga):", resultado[2])
print("Desviación estándar para pendiente (sigb):", resultado[3])
