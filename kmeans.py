#coding:utf-8
import numpy as np      #科学计算包
from sklearn.datasets import make_blobs
from sklearn.cluster import MiniBatchKMeans

batch_size = 5000

user_ID=[]
user_X=[]
user_Y=[]
f = open('itemAttribute.txt',encoding='UTF-8')
line = f.readline()
while line:
    line=line.replace('None','0')
    user_id,user_x,user_y=line.split('|')
    user_ID.append(int(user_id))
    user_X.append(int(user_x))
    user_Y.append(int(user_y.split('\n')[0]))
    line = f.readline()
f.close()
X=[user_X,user_Y]
X=np.transpose(X)
with open('SSE.txt','w+',encoding='UTF-8') as ff:
    for i in range(50,10000):
        KM=MiniBatchKMeans(init='k-means++', n_clusters=i, batch_size=batch_size, n_init=10, max_no_improvement=10, verbose=0)
        y_pred = KM.fit_predict(X)  
        centers=KM.cluster_centers_
        SSE=0
        for user_n in range(0,len(user_ID)):
            SSE=SSE+pow(pow(user_X[user_n]-centers[y_pred[user_n]][0],2)+pow(user_Y[user_n]-centers[y_pred[user_n]][1],2),0.5)
        ff.write("n_clusters="+str(i)+":    "+str(SSE)+'\n')