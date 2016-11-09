# -*- coding:utf8 -*-
from __future__ import division
import os
import math
from sklearn.cross_validation import train_test_split

class document:
    pass

wordHash = {}
categoryDocNum = {}
P = {}
defaultP = {}
categoryFeatureP = {}

docs = []
docs_y = []

totalDocNum = 0

def doc2vec():
    global totalDocNum,wordHash,categoryDocNum,totalDocNum
    #计算出词典,把文档词转化为词对应的key
    dir_path = "data_nb"
    files = os.listdir(dir_path)
    for f in files:
        if(os.path.isfile(dir_path+'/'+f)):
            f1 = f[:f.find(".seg.cln.txt")].lstrip("0123456789")
            id = "".join(c for c in f if c.isdigit())
            d = document()
            d.type = f1
            d.id = id
            fileHandle = open(dir_path+'/'+f)
            fileList = fileHandle.readlines()
            filemap = {}
            num=0
            for fileline in fileList:
                lists = fileline.split(' ')
                for str1 in lists:
                    if str1.__len__()<=0:
                        continue
                    if str1 not in wordHash:
                        num += 1
                        wordHash[str1] = num
                    if filemap.has_key(str1):
                        filemap[str1] += 1
                    else:
                        filemap[str1] = 0

            if filemap.__len__() > 0:
                docs.append(filemap)
                docs_y.append(d)
                #计算出词典,每个分类的文档数
                if f1 in categoryDocNum:
                    pass
                else:
                    categoryDocNum[f1] = 0

    return

def trainmodel(x_train_1docs, y_train_1docs):
    global totalDocNum,wordHash,categoryDocNum,defaultP,categoryFeatureP

    for item in y_train_1docs:
        type = item.type
        categoryDocNum[type] += 1
        totalDocNum += 1
    #扫描训练文档集,计算出二维字典每个特征的对应分类的文档数
    for i,map in enumerate(x_train_1docs):
        flagmap = {}
        type = y_train_1docs[i].type
        for item1 in map:
            if type in categoryFeatureP:
                if item1 in categoryFeatureP[type] and item1 not in flagmap:
                    categoryFeatureP[type][item1] += 1
                    flagmap[item1] = True
                elif item1 not in categoryFeatureP[type] and item1 not in flagmap:
                    categoryFeatureP[type][item1] = 1
                    flagmap[item1] = True
            else:
                categoryFeatureP.update({type:{item1:1}})

    #计算出每个分类的平滑默认值
    for key, value in categoryDocNum.items():
        defaultP[key] = 0.1 / (value+0.2)
        P[key] = math.log(value / totalDocNum)

    #扫描二维字典,计算出每个分类特征对应的概率,同时计算每个分类的大概率
    for type1, value1 in categoryFeatureP.items():
        for word in wordHash:
            probability = defaultP[type1]
            typeCount = categoryDocNum[type1]
            if word in value1:
                probability = (value1[word]+0.1)/(typeCount+0.2)
                value1[word] = math.log(probability) - math.log(1.0 - probability)
            P[type1] += math.log(1.0 - probability)
    return

def predict ( x_test_1docs, y_test_1docs ):

    global totalDocNum,wordHash,categoryDocNum,defaultP,categoryFeatureP
    y_test_pre = []
    notcorrect = 0

    for i,map in enumerate(x_test_1docs):
        type = y_test_1docs[i].type
        id  = y_test_1docs[i].id
        maxP = -100000000
        pri_type = ""
        for pType,score in P.items():
            scoreP = score
            for key in map:
                if key in categoryFeatureP[pType]:
                    scoreP += categoryFeatureP[pType][key]
                elif key not in wordHash:
                    continue
                else:
                    scoreP += (math.log(defaultP[pType])-math.log(1-defaultP[pType]))
            if scoreP > maxP:
                pri_type = pType
                maxP = scoreP
        if pri_type != type:
            notcorrect+=1
        y_test_pre.append(pri_type)

    print "Accuracy:",(len(x_test_1docs) -notcorrect) / len(x_test_1docs)

    return

doc2vec()
X_train_docs, X_test_docs, y_train_docs, y_test_docs = train_test_split(docs, docs_y, test_size=0.2)
trainmodel(X_train_docs, y_train_docs)
predict(X_test_docs, y_test_docs)
