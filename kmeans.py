# -*- coding:utf8 -*-
from __future__ import division
import random
import os
import math
import time

doc_x = []  # vector,文档向量,每一维为文档词频,暂时不用weight,利用余弦
doc_x_norm = []
doc_y = []  #
doc_predict = []
wordmap = {}
typeSet = set()
centerPoint = []
centerPointNorm = []
K = 0
typeList = [[] for i in range(K)]
typeNumList={}

#质点的选择问题
class doctype:
    pass

def calvecnorm(x):
    norm = 0
    for key,value in x.items():
        norm+=value*value
    norm  = math.sqrt(norm)
    return norm

def data2vec():
    global doc_x,doc_y,doc_predict,wordmap,typeSet,centerPoint,K,typeNumList
    dir_path = "kmeans_data"
    files = os.listdir(dir_path)
    for f in files:
        if (os.path.isfile(dir_path + '/' + f)):
            docType = doctype()
            docType.type = f[:f.find(".seg.cln.txt")].lstrip("0123456789")
            if docType.type not in typeSet:
                typeSet.add(docType.type)
            if docType.type not in typeNumList:
                typeNumList[docType.type] = 1
            else:
                typeNumList[docType.type] += 1
            docType.id = "".join(c for c in f if c.isdigit())
            fileHandle = open(dir_path + '/' + f)
            fileList = fileHandle.readlines()

            docmap = {}
            for fileline in fileList:
                lists = fileline.split(' ')
                for wordinfile in lists:
                    if wordinfile.__len__() <= 0:
                        continue

                    if wordinfile not in wordmap:
                        wordmap[wordinfile] = 0

                    if wordinfile in docmap:
                        docmap[wordinfile] += 1
                    else:
                        docmap[wordinfile] = 1

            if docmap.__len__() > 0:
                doc_x.append(docmap)
                doc_y.append(docType)
                doc_x_norm.append(calvecnorm(docmap))
    K = len(typeSet)
    doc_predict = [0]*doc_x.__len__()
    print "doc num:", doc_x.__len__()
    print "word num:", wordmap.__len__()
    return

def selectCenterPoint(typeList):

    global doc_x,doc_y,doc_predict,wordmap,typeSet,centerPoint,K

    wcss = 0
    for i in range(K):
        centerPoint[i] = {}
        centerPointNorm[i] = 0
        for j in range(len(typeList[i])):
            index = typeList[i][j]
            for key in doc_x[index]:
                if key not in centerPoint[i]:
                    centerPoint[i][key] = doc_x[index][key]
                else:
                    centerPoint[i][key] += doc_x[index][key]

        for key in centerPoint[i]:
            centerPoint[i][key]/=len(typeList[i])
            centerPointNorm[i]+= centerPoint[i][key]*centerPoint[i][key]

        centerPointNorm[i] = math.sqrt(centerPointNorm[i])

        print "new centerPoint:",len(centerPoint[i])

    return wcss

def initCenterPoint(list):
    slice = random.sample(list, K)
    for i in range(len(slice)):
        centerPointNorm.append(calvecnorm(slice[i]))
    return slice

# def caldist(x,xnorm,y,ynorm):
#     newValue = 0
#     for xk in x:
#         if xk in y:
#             newValue += x[xk]*y[xk]
#
#     value = xnorm*ynorm
#     if value == 0:
#         return 0
#     cosines = newValue/value
#     return cosines

def caldist(x,xnorm,y,ynorm):
    newValue = 0
    for xk in x:
        if xk in y:
            newValue += abs(x[xk]*x[xk]-y[xk]*y[xk])
        else:
            newValue += abs(x[xk]*x[xk])

    for yk in y:
        if yk not in x:
            newValue ++ abs(y[yk]*y[yk])

    return math.sqrt(newValue)

def fit():
    global doc_x, doc_y, doc_predict, wordmap, typeSet, centerPoint,typeList

    # 随机选择质点
    centerPoint = initCenterPoint(doc_x)
    oldwcss = 2000
    wcss = 1
    stepnum = 0

    while 1:
        stepnum += 1
        if abs(oldwcss - wcss) < 5 or stepnum>10:
            break
        typeList = [[] for i in range(K)]
        oldwcss = wcss
        wcss=0
        print doc_x.__len__()
        for i in range(doc_x.__len__()):
            minValue = -1
            pos = 0
            for j in range(K):
                #计算质点到各个点的距离
                dist = caldist(centerPoint[j], centerPointNorm[j], doc_x[i], doc_x_norm[i])
                #if dist > 1:
                    #print dist
                if minValue < dist:
                    minValue = dist
                    pos = j
            if minValue <=0:
                print "------------dist:",minValue
            wcss += minValue
            typeList[pos].append(i)

        for i in  range(K):
            print len(typeList[i])
        #重新计算质点
        selectCenterPoint(typeList)
        print "oldwcss:",oldwcss,"wcss:",wcss,"stepnum:",stepnum
    return

def effectvalidation():
    global doc_x, doc_y, doc_predict, wordmap, typeSet, centerPoint,typeList,typeNumList
    #print "|||||||||:",typeNumList
    typePredict = []
    for i in  range(K):
        maxType = ""
        maxValue = -1
        typemap={}
        doc = doctype()
        for index in typeList[i]:
            if doc_y[index].type not in typemap:
                typemap[doc_y[index].type] = 1
            else:
                typemap[doc_y[index].type] += 1
        for key,value in typemap.items():
            if value > maxValue:
                maxValue = value
                maxType = key
        doc.maxValue = maxValue
        doc.maxType = maxType
        typePredict.append(doc)

    for i in range(K):
        Precision = 0
        Recall = 0
        type = typePredict[i].maxType
        Precision = typePredict[i].maxValue/len(typeList[i])
        Recall = typePredict[i].maxValue/typeNumList[type]
        print typePredict[i].maxValue,len(typeList[i]),typeNumList[type]
        print type,"Precision:",Precision,"Recall",Recall

    return

print time.time()
data2vec()
fit()
effectvalidation()
print time.time()