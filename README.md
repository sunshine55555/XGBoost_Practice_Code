第二步：

#目标：计算价值的最小值
minimum_price = np.min(prices.astype('float'))

#目标：计算价值的最大值
maximum_price = np.max(prices.astype('float'))

#目标：计算价值的平均值
mean_price = np.mean(prices.astype('float'))

#目标：计算价值的中值
median_price = np.median(prices.astype('float'))

#目标：计算价值的标准差
std_price = np.std(prices.astype('float'))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, prices, train_size=0.8, random_state=1)



RM与MEDV成正比，RM越大，MEDV越大  
LSTAT和PTRATIO分别与MEDV成反比，二者越小，MEDV越大


1. test set用来检测模型的好坏， 并通过GridCV来确定超参数来构建更好的模型，便于用test set测试模型好坏
2. 用模型训练过的数据进行模型测试，无法准确的确定模型的预测能力或者说得出的模型的预测能力无法让人信服
3. 没有test set，不能准确知道训练出来的模型的好坏


trainin & testing data  
在一个独立的数据集上来测试模型的好坏
作为一个过拟合的检测方式

第三步：
# TODO 3

# 提示： 导入r2_score
from sklearn.metrics import r2_score
def performance_metric(y_true, y_predict):
    """计算并返回预测值相比于预测值的分数"""
    
    score = r2_score(y_true,y_predict)

    return score
    
    
# TODO 3 可选

# 不允许导入任何计算决定系数的库

def performance_metric2(y_true, y_predict):
    """计算并返回预测值相比于预测值的分数"""
    sum = 0
    y_predict_sum = 0
    for i in y_predict:
        y_predict_sum +=i
    y_predict_avg = y_predict_sum/len(y_predict)    
    print y_predict_sum, np.sum(y_predict)
    print y_predict_avg, np.mean(y_predict)
    for i in range(len(y_true)):
        sum += ((y_true[i]-y_predict[i])**2)/((y_true[i] - y_predict_avg)**2)
    print sum
    score = 1-sum

    return score
    
 R2的值0.923,我通过函数标准化两列数据以后测得的R2为0.948.认为已经很好的描述了目标变量的变化
 
 
 第四步：
 选择第二个图，max_depth=3,随着训练数据增加，测试集R2缓慢的降低，训练集大于50后R2缓慢的增加  
更多的训练数据，测试集的表会提高会显著提高，缓慢增加逼近测试集的R2。
    
 
Q6：
为3时能够最好的未见过的数据进行预测  
在max_depth=4时，测试集上的r2对应的值最大，训练集与测试集之间的误差最小，模型的泛化能力最强。

第五步：
Q7：
通过参数网格的形式提供给模型进行全面的参数选择，可以通过GridSearchCV对训练集进行训练选择出模型效果最好的超参数
Q8:
1. 将训练数据分成K份，然后将每一份拿出来做测试集，其他作为训练集，总共进行K次训练，然后去平均值，这样的得到的结果更精确
2. 
3. cv_results_能够告诉我们所有的参数验证的各个组合对应的得分  
4. 得到的值不够

# TODO 4

#提示: 导入 'KFold' 'DecisionTreeRegressor' 'make_scorer' 'GridSearchCV' 
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

def fit_model(X, y):
    """ 基于输入数据 [X,y]，利于网格搜索找到最优的决策树模型"""
    
    cross_validator = KFold(n_splits=10)
    
    regressor = DecisionTreeRegressor(random_state=0)

    params = {'max_depth':range(1,11)}

    scoring_fnc = make_scorer(performance_metric)

    grid = GridSearchCV(regressor,param_grid=params,scoring=scoring_fnc,cv=cross_validator)

    # 基于输入数据 [X,y]，进行网格搜索
    grid = grid.fit(X, y)

    # 返回网格搜索后的最优模型
    return grid.best_estimator_
    
Q9:
max_depth=4。  
相同，当depth=4

Q10：
我会建议他们的售价和上面的预测结果相一致  
从特征来说，第三个客户的房间最多，社区贫困和师生比最低，价格最高，相对客服2来说，正好相反，所以价格最低。所以价格合理  

Q11:
R^2为0.78,结果不好，可能需要其他的模型进行尝试


Q12
可以考虑  
不足够，特征太少  
不能，大都市和乡镇地区的差别还是很大的，所需要的特征也有很大差别，需要根据具体情况调整  
不合理，需要更多特征，比如人口数，人口种类，犯罪情况，学校分布，交通便利性等等  

from sklearn.model_selection import train_test_split
from sklearn.metrics import Gr
df = pd.read_csv('bj_housing.csv')
print df.shape
y=df['Value']
x=df.drop('Value',axis=1)
X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=0)
print X_train.shape
print X_test.shape

reg = fit_model(X_train,y_train)
print reg
print performance_metric(y_test,reg.predict(X_test))
vs.ModelLearning(X_train, y_train)
vs.ModelComplexity(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
randomf = RandomForestClassifier(random_state=0,min_samples_split=5)
parameters = {'max_depth':range(1,10),'n_estimators':range(5,30,5)}
clf = GridSearchCV(randomf,param_grid=parameters,cv=10)

optimal_reg2=clf.fit(X_train,y_train)
print performance_metric(y_test,optimal_reg2.predict(X_test))
 

# XGBoost_Practice_Code
XGBoost课程的代码
关于文件提取的代码：


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





