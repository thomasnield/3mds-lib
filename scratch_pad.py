from scipy.stats import norm

mean = 14.005
std = 0.955

p = norm.cdf(16, mean, std) - norm.cdf(15, mean, std)

print(x) # prints 0.5