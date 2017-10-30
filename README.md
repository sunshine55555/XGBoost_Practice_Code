
P3:
1.选择样本：
indices =[23,200,401]
2.特征相关性
问题1：
**回答:** 第一家企业对于所有物品的需求量都很高，部分数据都是分别远远高其他两个店，推测为超市，需求量大而且全。   
第三家企业的fresh与frozen和第二家企业的milk，grocery和detergents_paper远多于彼此。  我推测第二家店更像是杂货店买一些grocery和生活日用品。第三家店卖fresh 和 frozen，像一个卖生鲜蔬菜的地方。


2.1 
# TODO：为DataFrame创建一个副本，用'drop'函数丢弃一个特征
new_data = data.drop([u'Detergents_Paper'],axis=1)
labels = data['Detergents_Paper']
# TODO：使用给定的特征作为目标，将数据分割成训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_data, labels, test_size=0.25, random_state=0)

# TODO：创建一个DecisionTreeRegressor（决策树回归器）并在训练集上训练它
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
regressor = DecisionTreeRegressor(random_state = 0)
params = {'max_depth':range(2,10), 'min_samples_leaf':range(1,5),'min_samples_split':range(2,5)}
clf = GridSearchCV(regressor, param_grid=params,cv=10)
clf.fit(X_train, y_train)
# TODO：输出在测试集上的预测得分
from sklearn.metrics import r2_score
score = r2_score(y_test, clf.predict(X_test))
print '被选择的参数是%s,得分是%.4f' %(i,score)
# print clf.best_estimator_
# print clf.best_params_
# print clf.best_score_

2.2
for i in data.columns:
    # TODO：为DataFrame创建一个副本，用'drop'函数丢弃一个特征
    new_data = data.drop([i],axis=1)
    labels = data[i]
    # TODO：使用给定的特征作为目标，将数据分割成训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(new_data, labels, test_size=0.25, random_state=0)

    # TODO：创建一个DecisionTreeRegressor（决策树回归器）并在训练集上训练它
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import GridSearchCV
    regressor = DecisionTreeRegressor(random_state = 0)
    params = {'max_depth':range(2,10), 'min_samples_leaf':range(1,5),'min_samples_split':range(2,5)}
    clf = GridSearchCV(regressor, param_grid=params,cv=10)
    clf.fit(X_train, y_train)
    # TODO：输出在测试集上的预测得分
    from sklearn.metrics import r2_score
    score = r2_score(y_test, clf.predict(X_test))
    print '被选择的数据是%s,得分是%.4f' %(i,score)
#     print clf.best_estimator_
#     print clf.best_params_
#     print clf.best_score_

问题2：
**回答:**我选择的是Detergents_Paper,得分是0.7557。从得分上来看，其他五个商品的选择能够较好的表示Detergents_Paper的消费数量，所以这个特征对于区分用户的消费习惯来说是有必要的。  


问题3:
**回答:** 从图中看，detergents_paper与milk,和Grocery都是明显得分正相关关系，milk与grocey也存在着正相关关系。  
大多数数据点分布在20000之前 ,从对角线的kde图上来看，所有图都不是正态分布而是左偏

特征缩放：
log_data = np.log(data)
log_samples = np.log(samples)
pd.plotting.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

异常值检测：
# 对于每一个特征，找到值异常高或者是异常低的数据点
outliers=[]
outliers2=[]
for feature in log_data.keys():
    
    # TODO：计算给定特征的Q1（数据的25th分位点）
    Q1 = np.percentile(log_data[feature], 25)
    
    # TODO：计算给定特征的Q3（数据的75th分位点）
    Q3 = np.percentile(log_data[feature], 75)
    
    # TODO：使用四分位范围计算异常阶（1.5倍的四分位距）
    step = (Q3-Q1)*1.5
    
    # 显示异常点
    print "Data points considered outliers for the feature '{}':".format(feature)
    display(log_data[((log_data[feature] <= Q1 - step) | (log_data[feature] >= Q3 + step))])
#     display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    outliers.extend((log_data[((log_data[feature] <= Q1 - step) | (log_data[feature] >= Q3 + step))]).index.values)
    outliers2.append((log_data[((log_data[feature] <= Q1 - step) | (log_data[feature] >= Q3 + step))]).index.values)
# 可选：选择你希望移除的数据点的索引

# for feature in log_data.keys():

# 如果选择了的话，移除异常点

good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

问题4：
outliers2
pd.DataFrame(outliers)[0].value_counts()
index_outliers = [154, 66, 75, 128, 65]
data.iloc[index_outliers]
**回答:** index为65,128,75,66,154的数据还有两个及以上的特征为异常值  
我认为应该移除，多于一个特征为异常数据，即使真实存在这样的数据，因为其数据过大或者过小均认为是小概率事情，都应该去掉，防止模型拟合出这些数据，影响模型的性能


PCA:
good_data = log_data.drop(log_data.index[index_outliers]).reset_index(drop = True)
# TODO：通过在good_data上使用PCA，将其转换成和当前特征数一样多的维度
from sklearn.decomposition import PCA
pca = PCA(n_components=6)
pca.fit(good_data)
print pca.explained_variance_ratio_
# TODO：使用上面的PCA拟合将变换施加在log_samples上
pca_samples = pca.transform(log_samples)

# 生成PCA的结果图
pca_results = vs.pca_results(good_data, pca)

问题5：
**回答:** 前两个解释了0.7068的方差，前四个解释了0.9311的方差。

练习：降为
# TODO：通过在good data上进行PCA，将其转换成两个维度
pca = PCA(n_components=2).fit(good_data)

# TODO：使用上面训练的PCA将good data进行转换
reduced_data = pca.transform(good_data)

# TODO：使用上面训练的PCA将log_samples进行转换
pca_samples = pca.transform(log_samples)

# 为降维后的数据创建一个DataFrame
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

聚类：
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
for i in range(2,10):
    clusterer = KMeans(n_clusters=i, random_state=0)
    clusterer.fit(reduced_data)
    # print clusterer.labels_
    print silhouette_score(reduced_data, clusterer.labels_,metric='euclidean')



https://github.com/zhpmatrix/awesome-xgb
http://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters  
http://blog.csdn.net/han_xiaoyang/article/details/52663170  
http://blog.csdn.net/suranxu007/article/details/49910323  
https://www.zhihu.com/search?type=content&q=GBDT  
http://www.jianshu.com/p/005a4e6ac775  
https://en.wikipedia.org/wiki/Gradient_boosting#Gradient_tree_boosting  
[Xgboost参数说明界面](http://xgboost.readthedocs.io/en/latest/parameter.html)

1.general parameters  
booster 一般是default ，有gtree,gblinear,或者 dart   
silent 一般是default
nthread  
num_pbuffer  

2.Parameters for tree    
**eta (学习率）需要调** range[0,1]  
gamma  对树的叶子节点损失的值，越大，越保守。控制过拟合range:>=0  
max_depth 越大模型越复杂  
subsample   
colsampe_bytree  
colsampe_bylevel  
lambda  
scale_pos_weight 调整权重（正负样本）  

3.learning task parameters  
objective：损失函数/目标函数  
     default=reg:linear  
   reg:linear  
   reg:logistic  
   binary:logistic  
   binary:logitraw  
   multi:softmax  
   multi:softprob  
   rank:pairwise  
   reg:gamma  

-----
base_score  
eval_metric:评估标准 rmse, mae, logloss, error, 

## Control Overfitting

1控制模型复杂度： max_depth, min_child_weight % gamma  
2增加随机性去使得模型更健壮 subsample, colsample_bytree 或者减少eta同时要增加num_round

## handle unbalanced Dataset

如果你只关心 positive & negative weights, via scale_pos_weight (**用AUC去评价**）  
关心predict the right probability,you cannot rebalance the dataset, set parameter max_delta_step to a finite number(say 1) will help convergence  

libsvm用来存储稀疏的矩阵特征

import numpy as np
import pandas as pd
import cPickle
import xgboost as xgb

dtrain = xgb.DMatrix(agaricus.txt.train)
dtest = xgb.DMatrix(agaricus.txt.test)


## 1.parameter setting

param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary'}
watch_list = [(dtest,'eval'),(dtrain, 'train')]
num_round =5 
# model = xgb.train(params=param, dtrain = dtrain, num_boost_round = num_round,watch_list)
model  = xgb.train(param, dtrain, num_round, watch_list)
model.predict(dtest) #返回的是个array对应的每个值为正样本的概率
labels = dtest.get_label()#将上面的概率转化为label  

#为啥我觉得是count呢？sum不就是sum（index）？？
error_num = sum([index for index in range(len(pred) if int(pred[index]>0.5)!=labels[index])])

model.dump_model('1.model')

## 2.交叉验证

import pandas as pd
import numpy as np
dtrain = xgb.DMatrix('agaricus.txt.train')
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
num_round = 3 #用三棵树来做，越多越好，但是容易过拟合
xgb.cv(param, dtrain, num_round, nfold=5, metrics = {'error'},seed=1)


## 3 调整样本权重

def preproc(dtrain, dtest, param):
    labels = dtrain.get_label()	
    ratio = float(np.sum(labels == 0))/np.sum(labels==1)
    param['scale_pos_ratio'] = ratio
    return (dtrain, dtest, param)

xgb.cv(params, dtrain, num_round, nfold=5, metrics={'auc'},seed=3,fpreproc=preproc) #预处理的fpreproc=preproc

## 4.自定义损失函数与交叉验证

### 自定义目标函数（log似然损失），交叉验证
# 提供一阶导数 和 二阶导数

def logregobj(pred, dtrain):
    labels = dtrain.get_label()
    pred = 1.0/(1+np.exp(-pred))
    grad = pred - labels #梯度
    hess = pred *(1-pred)
    return grad, hess

def evalerror(pred, dtrain):
    labels = dtrain.get_label()
    return 'error rate:', float(sum(labels!=(pred>0.0)))/len(labels)#pred是linear出来的结果，再放到sigmoid里面

param = {'max_depth':2, 'eta':1, 'silent':1}
num_round = 3 
# model = xgb.train(param, dtrain, num_round, watch_list, logregobj, evalerror)
xgb.cv(param,dtrain, num_round, nfold =5, seed =3 ,obj=logregobj, feval=evalerror)

## 高级用法
用前N颗树做预测

pred2 = model.predict(dtest, ntree_limit = 1)
print evalerror(pred2, dtest)

## 画出特征重要度

%matplotlib inline
from xgboost import plot_importance
import matplotlib.pyplot as plt
help(plot_importance)
plot_importance(model, max_num_features =10)
plt.show()


## 如何和sklearn/pandas结合

import cPickle
# import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.datasets import load_iris,load_digits, load_boston

#用xgboost建模，用sklearn做评估
# 二分类问题，用混淆矩阵

digits = load_digits()
y=digits['target']
X= digits['data']
print X.shape
print y.shape

kf = KFold(n_split=2, shuffle=True, random_state=0)
for train_index, test_index in kf.split(X):
    



    
<pre name="code" class="python"><pre name="code" class="python">#!/usr/bin/env python  
#-*- coding:utf-8 -*-  
2017/09/04
http://www.cnblogs.com/russellluo/p/3299725.html
#####字典基础知识
#text.txt data:i234567890abcdefghijklmno
data=oepn('text.txt','r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print "data has %d character, %d unique." %(data_size, chars)
char_to_ix = {ch:i for i, ch in enumerate(chars)}
ix_to_char = {i:ch for i, ch in enumerate(chars)}

print (char_to_ix)
print (ix_to_char)

a = {'one':1, 'two':2, 'three':3}
b = dict(one=1,tow=2,three=3)
c = dict(zip(['one','two','three'],[1,2,3]))
d = dict({'three': 3, 'one': 1, 'two':2})
d = dict({'three': 3, 'one': 1, 'two': 2})
a==b==c==d

a = {'one':1, 'two':2, 'three':3}
b = dict(one=1, two=2, three=3)
c = dict(zip(['one','two','three'],[1,2,3]))

d=dict.formkeys(['a','b','c'])
d=dict.formkeys(['a','b','c'],6)
len(d)
d.clear()
d = a.copy()
d

d['three']
d['four']
d
del d['one']

#列表转为字典
list1=['key1','key2','key3']
list2=['1','2','3']
dict(zip(list1,list2))
#法2：
new_list = [['key1','value1'],['key2','value2'],['key3','value3']]
dict(new_list)





####英文分词
import pandas as pd
import numpy as np
#先把dict做成字典，方便映射
dict_url=r"C:\Users\samsung\Desktop\py2pronhydra.map"
py22 = pd.read_table(dict_url,encoding='utf-8',header=None)
py22.head()
py_dict=dict(zip(py22[1],py22[0]))


#处理文本
txt_url=r"C:\Users\samsung\Desktop\hehe.txt"
hehe = pd.read_table(txt_url,encoding='utf-8',header=None)

def tran2(x):
    a=""
    length=int(len((x).split())/3)
    try:
        for i in range(length):
            a+=(py_dict[" ".join(x.split()[i*3:(i+1)*3])])
    except KeyError:
        return np.NaN
    return a.lower()


#转化为拼音
hehe['pinyin'] = hehe[1].apply(tran2)
#没有被转化的提取出来，转化为文本
phrase_without = hehe[hehe['pinyin'].isnull()]
del phrase_without['pinyin']
phrase_without.to_csv('phrase_without.txt',encoding='utf-8',header=None,index=None,sep='\t')
#组合成需要的组合，没有被提取出来的将为null
hehe['combine']=hehe[0]+'\t'+(hehe['pinyin'])
#删除不需要的列
del hehe[1]
del hehe['pinyin']
del hehe[0]
hehe.to_csv('hehe.txt',encoding='utf-8',header=None,index=None)

#编码检测包：自认为没啥鸟用，，， 
import chardet
f = open('file','r')
fencoding=chardet.detect(f.read())
print fencoding

#####sklearn的一些基本方法

alpha_can = np.logspace(-3,2,10)
lasso_model =GridSearchCV(model, param_grdi={'alpha':alpha_can}, cv=5)



import multiprocessing

for i in range(5):
	p=multiprocessing.Process(func,args=(i,))
	jobs.append(p)
	p.start()
	

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import wordsegment
from wordsegment import clean
from wordsegment import segment
import pandas as pd
import numpy as np
import re
import copy
import time

text_url = r"/home/asr/wyl/corpus/dictMdl/test.en"
text = pd.read_table(text_url,encoding='utf-8',header=None)
text3 = text.iloc[:100]
pattern=re.compile('[\[\]\'\,]')

def eng_transform(text4):
    i=0
    a=text4.split()
    b= []
    num=[]
    num2=[]
    while i<len(a):
        try:
            tmp=[]
            while bool(re.match(r'.*[a-zA-Z].*',a[i]))==False:
                    i+=1
            num2.append(i)
            while bool(re.match(r'.*[a-zA-Z].*',a[i])) :
                tmp.append(a[i])
                i+=1
            num.append(i)
        except:
            pass    
        m= (segment(str(tmp)))
        b.append(m)
    if (len(num2)>len(num)):
        a[num2[len(num2)-1]:]=b[len(num2)-1]
        for i in range(len(num),0,-1):
            a[num2[i-1]:num[i-1]]=b[i-1]
    elif (len(num2)<len(num)):
        a[num[len(num)-1]:]=b[len(num)-1]
        for i in range(len(num2),0,-1):
            a[num[i-1]:num2[i-1]]=b[i-1]
    elif (len(num2)==len(num)):
        for i in range(len(num2),0,-1):
            if num[i-1]<num2[i-1]:
                a[num[i-1]:num2[i-1]]=b[i-1]
            else:
                a[num2[i-1]:num[i-1]]=b[i-1]

    return re.sub(pattern,"",str(a))

start = time.clock()
text3 = text3[0].apply(lambda x:eng_transform(x))
end = time.clock()
print (str(end-start))
text3.to_csv(r'/home/asr/yhd/test.txt',encoding='utf-8',header=None,index=None)





