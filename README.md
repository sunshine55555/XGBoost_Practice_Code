# XGBoost_Practice_Code
XGBoost课程的代码
关于文件提取的代码：


<pre name="code" class="python"><pre name="code" class="python">#!/usr/bin/env python  
#-*- coding:utf-8 -*-  
  
''''' 
用逆向最大匹配法分词，不去除停用词 
'''  
import codecs  
import xlrd  
  
#读取待分词文本,readlines（）返回句子list  
def readfile(raw_file_path):  
    with codecs.open(raw_file_path,"r",encoding="ANSI") as f:  
        raw_file=f.readlines()  
        return raw_file  
#读取分词词典,返回分词词典list  
def read_dic(dic_path):  
    excel = xlrd.open_workbook(dic_path)  
    sheet = excel.sheets()[0]  
    # 读取第二列的数据  
    data_list = list(sheet.col_values(1))[1:]  
    return data_list  
#逆向最大匹配法分词  
def cut_words(raw_sentences,word_dic):  
    word_cut=[]  
    #最大词长，分词词典中的最大词长,为初始分词的最大词长  
    max_length=max(len(word) for word in word_dic)  
    for sentence in raw_sentences:  
        #strip()函数返回一个没有首尾空白字符(‘\n’、‘\r’、‘\t’、‘’)的sentence，避免分词错误  
        sentence=sentence.strip()  
        #单句中的字数  
        words_length = len(sentence)  
        #存储切分出的词语  
        cut_word_list=[]  
        #判断句子是否切分完毕  
        while words_length > 0:  
            max_cut_length = min(words_length, max_length)  
            for i in range(max_cut_length, 0, -1):  
                #根据切片性质，截取words_length-i到words_length-1索引的字，不包括words_length,所以不会溢出  
                new_word = sentence[words_length - i: words_length]  
                if new_word in word_dic:  
                    cut_word_list.append(new_word)  
                    words_length = words_length - i  
                    break  
                elif i == 1:  
                    cut_word_list.append(new_word)  
                    words_length = words_length - 1  
        #因为是逆向最大匹配，所以最终需要把结果逆向输出，转换为原始顺序  
        cut_word_list.reverse()  
        words="/".join(cut_word_list)  
        #最终把句子首端的分词符号删除，是避免以后将分词结果转化为列表时会出现空字符串元素  
        word_cut.append(words.lstrip("/"))  
    return word_cut  
#输出分词文本  
def outfile(out_path,sentences):  
    #输出模式是“a”即在原始文本上继续追加文本  
    with codecs.open(out_path,"a","utf8") as f:  
        for sentence in sentences:  
            f.write(sentence)  
    print("well done!")  
def main():  
    #读取待分词文本  
    rawfile_path = r"逆向分词文本.txt"  
    raw_file=readfile(rawfile_path)  
    #读取分词词典  
    wordfile_path = r"words.xlsx"  
    words_dic = read_dic(wordfile_path)  
    #逆向最大匹配法分词  
    content_cut = cut_words(raw_file,words_dic)  
    #输出文本  
    outfile_path = r"分词结果.txt"  
    outfile(outfile_path,content_cut)  
         
  
if __name__=="__main__":  
    main()</pre><br>  
<br>  
<pre></pre>  
分词结果：  
<pre></pre>  
<pre name="code" class="python"><pre></pre><img src="http://img.blog.csdn.net/20170720141313207?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGFsYWxhd3h0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" alt="">  
<pre></pre>  
<pre name="code" class="python">   分析分词结果可以知道，机械分词的效果优劣，一方面与分词匹配算法有关，另外一方面极其依赖分词词典。所以若想得到好的分词效果，处理相关领域的文本时，需要在分词词典中加入特定领域的词汇。  
<p></p><p>     </p><p>        </p><p>      </p></pre>  
<pre></pre>  
<pre></pre>  
<pre></pre>  
     
</pre></pre>  
