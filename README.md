boosting是个串行迭代算法，对每个样本不断的进行调整，对样本依赖性较强，可以很好的拟合数据，但是若训练数据有问题（噪音太多），则模型会很差，bias会高




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
    
    
    
    
    


# TODO：从sklearn中导入三个监督学习模型
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# TODO：初始化三个模型
clf_A = SVC(kernel='rbf')
# clf_A = GradientBoostingClassifier()
clf_B = LogisticRegressionCV()
# clf_C = GradientBoostingClassifier()
# clf_C = GridSearchCV(SVC(kernel='rbf', decision_function_shape='ovr'), param_grid={'C': alpha, 'gamma': alpha})
clf_C = RandomForestClassifier()

# TODO：计算1%， 10%， 100%的训练数据分别对应多少点
samples_1 = int(X_train.shape[0]*0.01)
samples_10 = int(X_train.shape[0]*.1)
samples_100 = X_train.shape[0]

# 收集学习器的结果
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_val, y_val)
vs.evaluate(results, 0.6,0.3)

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]






















    
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





