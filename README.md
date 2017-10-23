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





