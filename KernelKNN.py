from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np

def K(X,Y=None,metric='poly',coef0=1,gamma=None,degree=3):
    if metric == 'poly':
        k = pairwise_kernels(X,Y=Y,metric=metric,coef0=coef0,gamma=gamma,degree=degree)
    elif metric == 'linear':
        k = pairwise_kernels(X,Y=Y,metric=metric)
    elif metric == 'sigmoid':
        k = pairwise_kernels(X,Y=Y,metric=metric,coef0=coef0,gamma=gamma)
    elif metric == 'rbf':
        k = pairwise_kernels(X,Y=Y,metric=metric,gamma=gamma)
    return k

class KernelKNN():  
    def __init__(self,metric='poly',coef0=1,gamma=None,degree=3,
                 n_neighbors=5,weights='distance'): #uniform
        self.m = metric
        self.coef = coef0
        self.gamma = gamma
        self.degree = degree
        self.n = n_neighbors
        self.weights = weights
    
    def fit(self,X,Y=None):
    # Polynomial kernel: K(a,b) = (coef0+gamma<a,b>)**degree
    # Sigmoid kernel: K(a,b) = tanh(coef+gamma<a,b>)
    # Linear kernel: K(a,b) = <a,b>
        X = np.asarray(X)
        Y = np.asarray(Y)
        self.x = X
        self.y = Y
        classes = np.unique(Y)
        label = np.zeros((len(Y),len(classes)))
        for i in range(len(Y)):
            for ii in range(len(classes)):
                if Y[i] == classes[ii]:
                    label[i][ii] = 1
        self.classes = classes
        self.label = label

        Kaa = []
        for i in range(len(X)):
            Kaa.append(K(X[i,:].reshape(1,-1),metric=self.m,
                         coef0=self.coef,gamma=self.gamma,degree=self.degree))
        self.Kaa = np.asarray(Kaa).ravel().reshape(len(Kaa),1)
        return self
    
    def predict_proba(self,X):
        X = np.asarray(X)
        Kbb = []
        Kab = K(self.x,X,metric=self.m,
                coef0=self.coef,gamma=self.gamma,degree=self.degree)

        for i in range(len(X)):
            Kbb.append(K(X[i,:].reshape(1,-1),metric=self.m,
                         coef0=self.coef,gamma=self.gamma,degree=self.degree))
        self.Kbb = np.asarray(Kbb).ravel()
        d = self.Kaa-2*Kab+self.Kbb #shape: (n_train,n_test)

        n_d = [] #neighbors' distance matrix
        index = []
        for i in range(d.shape[1]):
            index.append(np.argsort(d[:,i])[:self.n])
            n_d.append(d[index[i],i])
        n_d = np.asmatrix(n_d) + 1e-20
        
        w = np.asarray((1/n_d) / np.sum(1/n_d,axis=1)) 
            #weights matrix, shape: (n_test,n_neighbors)
        w_neighbor = w.reshape((w.shape[0],1,w.shape[1]))
            #neighbors' weights matrix, shape: (n_test,1,n_neighbors)
        
        prob = []
        label_neighbor = self.label[index] 
            #neighbors' index, shape: (n_test,n_neighbors,n_classes)
        for i in range(len(w_neighbor)):
            prob.append(np.dot(w_neighbor[i,:,:],label_neighbor[i,:,:]).ravel())
        prob = np.asarray(prob)
        self.prob = prob
        return prob
    
    def predict(self):
        
        #prob = predict_proba(self,X)
        yhat = self.classes[np.argmax(self.prob,axis=1)]
        return yhat