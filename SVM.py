
"""

import pandas as pd

import numpy as  np

data=pd.read_csv("heart.csv")

data= data.sample(frac = 1)

df= data.sample(frac=1).reset_index(drop=True)

df['target'] = df['target'].map({1: 1, 0: -1})

Y = df.loc[:, 'target']  
X = df.iloc[:, 0:-1]

X_normalized=(X-X.mean())/(X.std())

X=pd.DataFrame(X_normalized)

X.insert(0,'one',1)

X_train = X[:212].values
Y_train = Y[:212].values

X_test = X[212:].values
Y_test = Y[212:].values

c=np.matrix(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0]))

R=10000 #regularization parameter

def compute_cost(W, X, Y): 
    N = X.shape[0]
    distances =  Y * (np.dot(X, W.transpose()))
    for i in range(len(distances)):
        distances[i]=1-distances[i]
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    loss= R * (np.sum(distances) / N)
    cost =sum( 1 / 2 * np.dot(W, W.transpose()) )+ loss
    return cost

cost=compute_cost(c,X_train,Y_train)
print("cost function before optimization",cost)

def G_D(w,x,y):
  alpha=0.001
  epochs=1000
  lambdaa=0.01
  for i in range(1,epochs):
    for j in range(x.shape[0]):
      if y[j]*np.dot(x[j],w.transpose())<1:
        w=w-alpha*(2*lambdaa*w-y[j]*x[j])
      else:
        w=w-alpha*2*lambdaa*w
  return(w)

a=np.array([])

a=G_D(c,X_train,Y_train.transpose())

r=compute_cost(a,X_train,Y_train)
print("cost function after optmization of training examples ",r)

def acc(p,x_batch):
  h_x=np.dot(x_batch,p.transpose())
  g=np.matrix(np.zeros((x_batch.shape[0], 1)))
  for i in range(x_batch.shape[0]):
    if h_x[i]>=1:
      g[i]=1
    elif h_x[i]<1:
      g[i]=-1
  return g

def cal_acc(Y_batch,X_batch,p):
  o=acc(p,X_batch)
  Y_batch=np.matrix(Y_batch).transpose()
  n=np.array(Y_batch-o)
  p=np.count_nonzero(n==0)
  accuracy=(p*100)/len(Y_batch)
  return accuracy

accuracy=cal_acc(Y_train,X_train,a)
print("accuracy of training examples ",accuracy)

g=compute_cost(a,X_test,Y_test)
print("cost function of testing examples after optimization ",g)

acc_test=cal_acc(Y_test,X_test,a)
print("accuracy of testing examples ",acc_test)

cols=['trestbps','chol','thalach','oldpeak']
X_4_f=df[cols]

X_normalized_4_f=(X_4_f-X_4_f.mean())/(X_4_f.std())

X_4_f=pd.DataFrame(X_normalized_4_f)

X_4_f.insert(0,'one',1)

X_train_4_f = X_4_f[:212].values
X_test_4_f = X_4_f[212:].values

c_4_f=np.matrix(np.array([0,0,0,0,0]))

p=G_D(c_4_f,X_train_4_f,Y_train.transpose())

c_4=compute_cost(p,X_train_4_f,Y_train)
print("cost function of 4 features after optimizing ",c_4)

acc_4_f=cal_acc(Y_train,X_train_4_f,p)
print("accuray of 4 features of training examples ",acc_4_f)

acc_4_f_test=cal_acc(Y_test,X_test_4_f,p)
print("accuray of 4 features of testing examples ",acc_4_f_test)

colss=['chol','thalach']

X_2_f=df[colss]

X_normalized_2_f=(X_2_f-X_2_f.mean())/(X_2_f.std())

X_2_f=pd.DataFrame(X_normalized_2_f)

X_2_f.insert(0,'one',1)

X_train_2_f = X_2_f[:212].values
X_test_2_f = X_2_f[212:].values

c_2_f=np.matrix(np.array([0,0,0]))

p2=G_D(c_2_f,X_train_2_f,Y_train.transpose())

c_2=compute_cost(p2,X_train_2_f,Y_train)
print("cost function of 2 features after optimizing ",c_2)

acc_2_f=cal_acc(Y_train,X_train_2_f,p2)
print("accuracy of 2 features (chol,thalach)of training examples ",acc_2_f)

acc_2_f_test=cal_acc(Y_test,X_test_2_f,p2)
print("accuracy of 2 features (chol,thalach) of testing examples ",acc_2_f_test)
