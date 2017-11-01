
1 feature rescaling
algorithm affteced by feature rescaling:
SVM:缩放以后，你的超平面会改变
kmeans clustering： 缩放以后，center 与点的距离也在改变

penultimate topic 倒数第二个topic

2 feature selection
curse of dimensionality



forward search
backward search

Markov Decision Process
States: S 
Model: T(s, a, s') Pr(s' | s, a)
Actions: A(s), A 
Reward: R(s) R(s, a), R(s, a, s')'
Policy







4.1.1
ls -a
.test
所有者 》 所属组 》 其他人
ls -lh
-rw-r--r--. 1 root root 7690 3月 3 09:06 test.sh 
 u  g  o                                  文件的最后一次修改时间
所有者  所属组 其他人
r读 w写 x执行(excute)
x:为最大权限；权限够用就行
10个字符：第一个表示文件类型
-:文件； d:目录; l:软链接
后面九个

ls -d /etc
ls -ld /etc
每个文件都有ID，查看文件ID：ls -i
ls -a -l -h -d -i 

4.1.2
mkdir test
mkdir -p 递归创建
mkdir -p /tmp/Janpa/buduo
mkdir /tmp/Janpan/longze /tmp/Janpa/cangjing

cd 
cd /tmp/Janpan/cangjing 
pwd: print working directory
rmdir: remove empty directories 
rm /tmp/Janpa/cangjing
cp -r /tmp/Janpa/cangjing /etc
cp 可以复制多个文件最后一个是目标目录
cp -p /root/install.log /tmp  (保存复制的文件的属性)
ls -l 

mv
mv 
clear 
rm 
rm -r 
rm -f强制删除 不用回复y是否删除
rm -rf强制删除一个目录

4.1.3
touch cat more less head tail
touch
touch test.txt 
touch shenchao girl 创建了两个文件
touch "program file" 创建了一个文件叫program file。不推荐这样有空格的文件

cat 
cat test.txt
cat -n test.txt 加行号
cat 浏览不长的文件
tac test.txt倒着显示文件

more :分页显示命令
more /etc/services 小命令：空格; f；enter; q:退出
小命令：空格; f；enter; q:退出

less:与more不同是可以向上往回翻页
less /etc/services ;
page up往上翻或者上箭头一行行翻
能够搜索：/
在搜索后，按n: 表示next也是往下翻搜索的东西

head 看文件的前几行后者后几行
head -n 7 /services
head /services 看前10行（默认）
tail -n 3 /services
tail /services
tail -f /var/log/messages 能够实时的看到文件动态的变化


4.1.4
软链接：类似于windows的快捷方式，方便找到文件
ln -s /etc/issue /tmp/issue.soft
将issue生成一个软链接issue.soft
ln /etc/issue /tmp/issue.hard 
ls -l /tmp/issue.soft：显示文件的指示路径。软链接的权限可以忽视，并不代表指示文件的权限
ls -l /tmp/issue.hard:想到与 cp -p ,但是又可以同步更新
硬链接的 id与源文件相同

4.2.1
u所有者  g所有组 o其他人
chmod更改权限
change the permissions mode of a file
chmod {+-=}{rwx}
chmod u+x test.txt (给u权限x(执行))
chmod g+w, o-r testtxt
chmod g=rwx text.txt
用的更多的是数字的方式：
r---4 /  w---2/ x ---1
rwxrw-r--
7 6 4
chmod 777 test.txt 
532: rx--wx-w-
chmod -R(递归修改) 改变一个文件面所有的文件

能够删除一个文件，是对这个文件的目录有写（x）权限
r:ls
w:touch/mkdir/rmdir/rm 
x:cd
rx 一般是同时出现的 


4.2.2

4.3.1
find:少用搜索
find [搜索范围] [匹配条件]
find /etc -name init(精准搜索，只搜索init)
find /etc -name *init*
find /etc -name init???(init后面匹配三个字符)
find /etc -iname(不区分大小写)
find /etc -size +n -n n （大于，小于，等于）
n为数字块为512字节 0.5k，100M=102400kb = 204800
find /etc -size +204800(大于100M的的文件会出来)
find /home -user shenchao
find /etc -cmin -5 改变文件属性
          -mmin -5 改变文件内容
          -


-a:表示and
find /etc -size -204800
-a
-o










强化学习reinforcement learning
0.
Q learning
Policy Gradient

不理解环境（model-Free RL）
Q learning
Sarsa
Policy Gradients

理解环境（model-based RL）
多出来一个建模
方法和model-free一样
可以想象来预测下一步的结果


基于概率（policy-based RL)
基于概率行动不一定选择最高的概率
Policy Gradients
基于价值（Value-Based RL）
选择价值最高的
只能选择离散的，连续的无能为力
Q learning; Sarsa

基于二者结合用的方法是 actor-critic 基于概率进行动作，然后基于动作，估计价值
加速了Policy gradients的学习过程

回合更新（monte-carlo update）
基础版的Policy Gradients
monte-carlo learning
单步更新（temporal-difference update）（有效率，常用）
Q learning
Sarsa
升级版的Policy Gradients

在线学习（on-policy）
Sarsa
Sarsa（lambda）
离线学习（off-policy）
Q learning
Deep Q Network
1
2
3 讲的很好，这节需要重复看
4 例子1
5，6 例子2
7， 8Sarsa
9Sarsa(lambda)










11/1
数据>=模型>=融合
lightGBM:不需要one-hot只需要告诉它这行需要处理（Microsoft）
XGboost需要处理one-hot
model ensemble
ensemble learning 是一组individual learner的组合
base learner 如果individual learner 是同质
component learner 如果individual learner 是异质
learner: h1 h2 h3 h4
统计上，可能四个的平均更接近f
计算上，如果你SGD时的函数是非凸的，多个模型融合能够防止局部极值的出现





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





