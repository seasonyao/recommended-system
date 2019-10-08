#coding:utf-8
import numpy as np
from numpy.linalg import cholesky
import math
import random
from sklearn.datasets import make_blobs
from sklearn.cluster import MiniBatchKMeans


#定下四个参数的量
F=150       #分类数
N=200        #迭代次数
Alpha=0.015  #学习速率
Lambda=0.01 #正则化参数
batch_size = 50000

user_number=19835
item_number=624961


#P矩阵
matrix_P = []
#Q矩阵
matrix_Q = []

#查找一个list里某个值的所有索引,在y中找x
def myfind(x,y):
    return [ a for a in range(len(y)) if y[a] == x]

#根据PQ矩阵计算对应user和item的预测score

def predict_score(user,item):
    temp=0
    for i in range(0,F):
        temp=temp+matrix_P[user][i]*matrix_Q[i][item]
    return temp


#利用itemAtrribute做Kmeans聚类
item_ID=[]
item_X=[]
item_Y=[]
f = open('itemAttribute.txt',encoding='UTF-8')
line = f.readline()
while line:
    line=line.replace('None','0')
    item_id,item_x,item_y=line.split('|')
    item_ID.append(int(item_id))
    item_X.append(int(item_x))
    item_Y.append(int(item_y.split('\n')[0]))
    line = f.readline()
f.close()

X=[item_X,item_Y]
X=np.transpose(X)
KM=MiniBatchKMeans(init='k-means++', n_clusters=F, batch_size=batch_size, n_init=10, max_no_improvement=10, verbose=0)
#y_pred和centers分别是聚类得到的每个点的分类label和每个类的中心点坐标
y_pred = KM.fit_predict(X)  


#读入train.txt制作成[userid，itemid，score]的list

user_item_nums=[]
user_item_score=[]
test_user_item_score=[]
f = open('train.txt',encoding='UTF-8')
line = f.readline()
count=0
while line:
    count=count+1
    if count%100==0:
        print(count)
    user=line.split('|')[0]
    user_item_num=int(line.split('|')[1])
    temp=random.randrange(0,user_item_num-1,1)
    for i in range(0,user_item_num):
        line=f.readline()
        if temp==i:
            test_user_item_score.append([int(user),int(line.split('  ')[0]),int(line.split('  ')[1].split('  ')[0])])
        else:
            user_item_score.append([int(user),int(line.split('  ')[0]),int(line.split('  ')[1].split('  ')[0])])
        if user_item_num<500:
            if int(line.split('  ')[0]) in item_ID:
                temp_list=myfind(y_pred[item_ID.index(int(line.split('  ')[0]))],y_pred)
                temp_index=random.randint(0,len(temp_list)-1)
                temp=temp_list[temp_index]
                user_item_score.append([int(user),item_ID[temp],int(line.split('  ')[1].split('  ')[0])])
                user_item_num=user_item_num+1
    user_item_num=user_item_num-1
    user_item_nums.append(user_item_num)
    line = f.readline()
f.close()

#初始化P和Q，方法：计算user和items所有已有评分的平均值，然后以平均值作为正态分布的中心轴以正态分布来随机这个值
print(1)
#P
#计算平均值
user_average=[]
count_1=0
for user_n in range(0,user_number):
    temp=0
    for ui_score in range(0,user_item_nums[user_n]):
        temp=temp+user_item_score[count_1][2]
        count_1=count_1+1
    user_average.append(temp/user_item_nums[user_n])

#赋值P矩阵
for row_number in range(0,user_number):
    np.random.seed(row_number)
    if 100-user_average[row_number]<=user_average[row_number]:
        s = np.random.normal(user_average[row_number],(100-user_average[row_number])/3, F)
    else:
        s = np.random.normal(user_average[row_number],user_average[row_number]/3, F)
    for i in range(0,F):
        if s[i]>100:
            s[i]=100
        if s[i]<0:
            s[i]=0
    matrix_P.append(s/100)
print(2)

#Q
#计算平均值,粗略地用已有记录的所有user_item的score作为平均值
temp=0
for i in range(0,len(user_item_score)):
    temp=temp+user_item_score[i][2]
average_all_item=temp/len(user_item_score)

#初始化所有item平均值为总评均值
item_average=[]
for i in range(0,item_number):
    item_average.append(average_all_item)

print(2.5)
#对于某个item在它有用户评过价的情况下计算平均值
item_occur=[]
for i in range(0,len(user_item_score)):
    item_occur.append(user_item_score[i][1])
print(2.6)


#赋值Q矩阵
for row_number in range(0,item_number):
    np.random.seed(row_number)
    if 100-item_average[row_number]<=item_average[row_number]:
        s = np.random.normal(item_average[row_number],(100-item_average[row_number])/3, F)
    else:
        s = np.random.normal(item_average[row_number],item_average[row_number]/3, F)
    for i in range(0,F):
        if s[i]>100:
            s[i]=100
        if s[i]<0:
            s[i]=0
    if row_number%10000==0:
        print(row_number)
    matrix_Q.append(s/100)
#转置Q矩阵
matrix_Q=np.transpose(matrix_Q)

print(3)



#正式进入学习阶段,开始迭代,迭代N次
with open('evidence_old.txt','w+',encoding='UTF-8') as ff:
    for step in range(0,N):
        #对于每一个已知的用户对某项目的评分我们都可以对我们的参数进行优化
        for ui_score in user_item_score:
            #计算损失函数
            user=ui_score[0]
            item=ui_score[1]
            cost_funtion=ui_score[2]-predict_score(user,item)
            if ui_score[0]%10000==0:
                print(cost_funtion)
                ff.write(str(cost_funtion))
                ff.write('\n')
            #优化参数
            for f in range(0,F):
                matrix_P[user][f]=matrix_P[user][f]+Alpha*(cost_funtion*matrix_Q[f][item]/100-Lambda*matrix_P[user][f])
                matrix_Q[f][item]=matrix_Q[f][item]+Alpha*(cost_funtion*matrix_P[user][f]/100-Lambda*matrix_Q[f][item])       
        #计算在训练集和测试集合上的准确率
        correct_num=0
        correct_num1=0
        correct_num2=0
        correct_num3=0
        correct_num4=0
        correct_num5=0
        correct_num6=0
        correct_num7=0
        for i in range(0,len(user_item_score)):
            if abs(predict_score(user_item_score[i][0],user_item_score[i][1])-user_item_score[i][2])<5:
                correct_num=correct_num+1
            if abs(predict_score(user_item_score[i][0],user_item_score[i][1])-user_item_score[i][2])<10:
                correct_num1=correct_num1+1
            if abs(predict_score(user_item_score[i][0],user_item_score[i][1])-user_item_score[i][2])<15:
                correct_num2=correct_num2+1
            if abs(predict_score(user_item_score[i][0],user_item_score[i][1])-user_item_score[i][2])<20:
                correct_num3=correct_num3+1
        for i in range(0,len(test_user_item_score)):
            if abs(predict_score(test_user_item_score[i][0],test_user_item_score[i][1])-test_user_item_score[i][2])<5:
                correct_num4=correct_num4+1
            if abs(predict_score(test_user_item_score[i][0],test_user_item_score[i][1])-test_user_item_score[i][2])<10:
                correct_num5=correct_num5+1
            if abs(predict_score(test_user_item_score[i][0],test_user_item_score[i][1])-test_user_item_score[i][2])<15:
                correct_num6=correct_num6+1
            if abs(predict_score(test_user_item_score[i][0],test_user_item_score[i][1])-test_user_item_score[i][2])<20:
                correct_num7=correct_num7+1
        print(correct_num/len(user_item_score))
        print(correct_num1/len(user_item_score))
        print(correct_num2/len(user_item_score))
        print(correct_num3/len(user_item_score))
        print(correct_nu4/len(test_user_item_score))
        print(correct_num5/len(test_user_item_score))
        print(correct_num6/len(test_user_item_score))
        print(correct_num7/len(test_user_item_score))
        ff.write(str(correct_num/len(user_item_score)))
        ff.write('\n')
        ff.write(str(correct_num1/len(user_item_score)))
        ff.write('\n')
        ff.write(str(correct_num2/len(user_item_score)))
        ff.write('\n')
        ff.write(str(correct_num3/len(user_item_score)))
        ff.write('\n')
        ff.write(str(correct_num4/len(test_user_item_score)))
        ff.write('\n')
        ff.write(str(correct_num5/len(test_user_item_score)))
        ff.write('\n')
        ff.write(str(correct_num6/len(test_user_item_score)))
        ff.write('\n')
        ff.write(str(correct_num7/len(test_user_item_score)))
        ff.write('\n')
        ff.write('-----------------------开始新一轮迭代---------------------\n')
    while(1):
        a=input("continue?y/n")
        if a=='y':
            Alpha=float(input("input Alpha"))
            #对于每一个已知的用户对某项目的评分我们都可以对我们的参数进行优化
            slice = random.sample(user_item_score, 10000)
            for ui_score in slice:
                #计算损失函数
                user=ui_score[0]
                item=ui_score[1]
                cost_funtion=ui_score[2]-predict_score(user,item)
                if ui_score[0]%10000==0:
                    print(cost_funtion)
                    ff.write(str(cost_funtion))
                    ff.write('\n')
                #优化参数
                for f in range(0,F):
                    matrix_P[user][f]=matrix_P[user][f]+Alpha*(cost_funtion*matrix_Q[f][item]/100-Lambda*matrix_P[user][f])
                    matrix_Q[f][item]=matrix_Q[f][item]+Alpha*(cost_funtion*matrix_P[user][f]/100-Lambda*matrix_Q[f][item])       
            #计算在测试集合上的准确率
            correct_num=0
            correct_num1=0
            correct_num2=0
            correct_num3=0
            correct_num4=0
            correct_num5=0
            correct_num6=0
            correct_num7=0
            for i in range(0,len(user_item_score)):
                if abs(predict_score(user_item_score[i][0],user_item_score[i][1])-user_item_score[i][2])<5:
                    correct_num=correct_num+1
                if abs(predict_score(user_item_score[i][0],user_item_score[i][1])-user_item_score[i][2])<10:
                    correct_num1=correct_num1+1
                if abs(predict_score(user_item_score[i][0],user_item_score[i][1])-user_item_score[i][2])<15:
                    correct_num2=correct_num2+1
                if abs(predict_score(user_item_score[i][0],user_item_score[i][1])-user_item_score[i][2])<20:
                    correct_num3=correct_num3+1
            for i in range(0,len(test_user_item_score)):
                if abs(predict_score(test_user_item_score[i][0],test_user_item_score[i][1])-test_user_item_score[i][2])<5:
                    correct_num4=correct_num4+1
                if abs(predict_score(test_user_item_score[i][0],test_user_item_score[i][1])-test_user_item_score[i][2])<10:
                    correct_num5=correct_num5+1
                if abs(predict_score(test_user_item_score[i][0],test_user_item_score[i][1])-test_user_item_score[i][2])<15:
                    correct_num6=correct_num6+1
                if abs(predict_score(test_user_item_score[i][0],test_user_item_score[i][1])-test_user_item_score[i][2])<20:
                    correct_num7=correct_num7+1
            print(correct_num/len(user_item_score))
            print(correct_num1/len(user_item_score))
            print(correct_num2/len(user_item_score))
            print(correct_num3/len(user_item_score))
            print(correct_nu4/len(test_user_item_score))
            print(correct_num5/len(test_user_item_score))
            print(correct_num6/len(test_user_item_score))
            print(correct_num7/len(test_user_item_score))
            ff.write(str(correct_num/len(user_item_score)))
            ff.write('\n')
            ff.write(str(correct_num1/len(user_item_score)))
            ff.write('\n')
            ff.write(str(correct_num2/len(user_item_score)))
            ff.write('\n')
            ff.write(str(correct_num3/len(user_item_score)))
            ff.write('\n')
            ff.write(str(correct_num4/len(test_user_item_score)))
            ff.write('\n')
            ff.write(str(correct_num5/len(test_user_item_score)))
            ff.write('\n')
            ff.write(str(correct_num6/len(test_user_item_score)))
            ff.write('\n')
            ff.write(str(correct_num7/len(test_user_item_score)))
            ff.write('\n')
            ff.write('-----------------------开始新一轮迭代---------------------\n')
        elif a=='n':
            break



with open('result_old.txt','w+',encoding='UTF-8') as fk:
    f = open('test.txt',encoding='UTF-8')
    line = f.readline()
    while line:
        user=line.split('|')[0]
        user_item_num=int(line.split('|')[1])
        fk.write(line)
        for i in range(0,user_item_num):
            line=f.readline()
            fk.write(line.split('\n')[0])
            fk.write('  ')
            fk.write(str(predict_score(int(user),int(line.split('  ')[0]))))
            fk.write('\n')
        line = f.readline()