import math

p = 1

# probability of heads 10 times in a row
for i in range(10):
    p *= .5

print(p) # 9.332636185032189e-302

# using logarithmic addition
p = 0
for i in range(10):
    p += math.log(.5)

print(math.exp(p)) # 9.332636185154842e-302