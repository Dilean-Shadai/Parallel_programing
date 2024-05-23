import time
from numba import jit, prange

def prime_numbers(start, end):
      primes = []
    for num in prange(start, end + 1):
        if num > 1:
           for i in range(2, int(num**0.5) + 1):
                if (num % i) == 0:
                    break
            else:
                primes.append(num)
    return primes

start_range = 10
end_range = 50

start_time = time.time()
prime_list = prime_numbers(start_range, end_range)
end_time = time.time()


execution_time = end_time - start_time
print("Números primos encontrados en el rango [", start_range, ",", end_range, "]:", prime_list)
print("Cantidad de números primos encontrados:", len(prime_list))
print("Tiempo de ejecución:", execution_time, "segundos")

