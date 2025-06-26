from classicml.gnb import GaussianNaiveBayes

import numpy as np
import matplotlib.pyplot as plt 

np.random.seed(42) 

n_samples = 500 
p_class_1 = 0.5 


y = np.random.binomial(n=1, p=p_class_1, size=n_samples)


X1 = np.zeros(n_samples)
X2 = np.zeros(n_samples)


mean_X1_0, std_X1_0 = 2, 1
mean_X2_0, std_X2_0 = -1, 0.5


mean_X1_1, std_X1_1 = -2, 1.5
mean_X2_1, std_X2_1 = 3, 1.0

for i in range(n_samples):
    if y[i] == 0:
        X1[i] = np.random.normal(mean_X1_0, std_X1_0)
        X2[i] = np.random.normal(mean_X2_0, std_X2_0)
    else:
        X1[i] = np.random.normal(mean_X1_1, std_X1_1)
        X2[i] = np.random.normal(mean_X2_1, std_X2_1)

X = np.column_stack((X1, X2))

plt.figure(figsize=(7, 6))

plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Class 0', alpha=0.6)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class 1', alpha=0.6)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Class-Conditional Feature Distributions")
plt.grid(True)
plt.legend()
plt.axis('equal')
# plt.show()

gnb = GaussianNaiveBayes()
gnb.fit(X, y)

prediction = gnb.single_predict(np.array([4, 2]))
print(prediction)





