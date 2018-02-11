import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB

## Load Training Data and Test data
df = pd.read_csv('HW_1_training.txt', sep='\t')
X = np.array(df.loc[:, ['x1', 'x2']])
y = np.array(df['y'])

df = pd.read_csv('HW_1_testing.txt', sep='\t')
X_test = np.array(df.loc[:, ['x1', 'x2']])
y_test = np.array(df['y'])


### Question 1:

# calculate mean vector and co-variance matrix
def getMeanVectorsCovMatrices(X,y):
	meanVectors, covMatrices = [],[]
	for cl in np.unique(y):
		meanVectors.append(np.mean(X[y == cl], axis=0))
		covMatrices.append(np.cov(X[y == cl].T))

	return meanVectors, covMatrices

meanVectors, covMatrices = getMeanVectorsCovMatrices(X,y)

for i in range(2):
	print("The mean vector for class %d is: %s" % (i, meanVectors[i]))
	print("The covariance matrix for class %d is: " % i)
	print(covMatrices[i])
	print("")


# Draw Decision Boundary with the Naive Bayesian Model

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	# plot the decision surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
	                       np.arange(x2_min, x2_max, resolution))

	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0], 
		        y=X[y == cl, 1],
		        alpha=0.8, 
		        c=colors[idx],
		        marker=markers[idx], 
		        label=cl, 
		        edgecolor='black')

	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.legend(loc='best')

## Draw Decision Boundary without Prior
plt.figure(1)
plt.title('Naive Bayesian Boundary with no priors')
clf = GaussianNB(priors=[.5,.5])
clf.fit(X,y)

plot_decision_regions(X, y, classifier=clf)
plt.show()

## Calculate Test Error
print('The classification error rate for Bayesian Decision Boundary without priors is: %.2f'
				 %(1-clf.score(X_test,y_test)))


## Draw Decision Boundary with Prior

p0 = list(y).count(0)*1./y.size
priors = [p0, 1-p0]

plt.figure(2)
plt.title('Naive Bayesian Boundary with priors')
clf2 = GaussianNB(priors=priors)
clf2.fit(X,y)

plot_decision_regions(X, y, classifier=clf2)
plt.show()

## Calculate Test Error

print('The classification error rate for Bayesian Decision Boundary with priors is: %.2f' 
				%(1-clf2.score(X_test,y_test)))





