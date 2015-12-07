import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
#from sklearn import datasets
from sklearn.decomposition import PCA

def plotpca(X):
    # iris = datasets.load_iris()
    # X = iris.data
    # X = np.random.rand(150,4)
    
    pca = PCA(n_components=4)
    X_r = pca.fit(X).transform(X)

    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    color=cm.rainbow(np.linspace(0,1,X_r.shape[0]))
    plt.figure()
    for i,c in zip(range(X_r.shape[0]),color):
        plt.scatter(X_r[i,0], X_r[i,1], c=c,label=i)
    plt.legend()
    plt.title('PCA of IRIS dataset')
    plt.show()
