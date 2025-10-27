import torch.nn as nn
import torch 

X = torch.randn(1)
print(X.dtype)
print (X)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu' 

    
X = X ** 3
rand_val = torch.empty(1).uniform_(1, 10)
print ("Случайное значение", rand_val)
X = X * rand_val
X = torch.exp(X)
print (X)







    
    