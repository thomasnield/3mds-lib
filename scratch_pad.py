from scipy.stats import norm

# Conventional formula has mean of 9 colic hours
# with 1.6 standard deviations
mean = 9
std = 1.6

# Experimental formula showed 5.5 hours of colic
x = 5.5

# Probability of 5.5
tail_p = norm.cdf(x, mean, std)

# Get p-value of both tails
p_value = 2*tail_p

print(p_value) # 0.028706043217603304