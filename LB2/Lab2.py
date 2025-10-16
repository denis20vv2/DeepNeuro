import numpy as np

n = 30

rnd = np.random.default_rng()

randomMatrix1 = rnd.integers(0, 200, size=(5, 5))

print("Матрица =")
print( randomMatrix1)

summ = 0
for row in randomMatrix1:
    for num in row:
        if num % 2 == 0:
            summ += num

print("Сумма чётных чисел =", summ)




