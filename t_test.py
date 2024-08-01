from scipy.stats import ttest_ind_from_stats

# Given values
# mean0 = 61.92
# std0 = 2.88

# mean1 = 62.85
# std1 = 2.17

# new_n0 = new_n1 = sum([7190.8, 4040.2, 9254, 6292.4]) * 5  # Each dataset repeated 5 times


# # Perform t-test with the updated sample sizes
# t_stat_updated, p_value_updated = ttest_ind_from_stats(mean1=mean1, std1=std1, nobs1=new_n1, mean2=mean0, std2=std0, nobs2=new_n0)
# print(t_stat_updated)
# print(p_value_updated)


mean0 = 44.05
std0 = 1.41

mean1 = 44.35 
std1 = 1.99

# new_n0 = new_n1 = sum([12691.2, 3632.2, 3232.8]) * 5  # Each dataset repeated 5 times
new_n0 = new_n1 = 15

# Perform t-test with the updated sample sizes
t_stat_updated, p_value_updated = ttest_ind_from_stats(mean1=mean1, std1=std1, nobs1=new_n1, mean2=mean0, std2=std0, nobs2=new_n0, alternative="less")
# print(std0)
print(t_stat_updated)
print(p_value_updated)