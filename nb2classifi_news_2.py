# -*- coding:utf8 -*-
from __future__ import division
import os
import math
from sklearn.cross_validation import train_test_split

class document:
    pass

wordHash = {}
newscategory = []
categoryDocNum = {}
P = {}
defaultP = {}
categoryFeatureP = {}

docs = []
docs_y = []

totalDocNum = 0

def doc2vec():
    global totalDocNum,wordHash,categoryDocNum,totalDocNum,newscategory
    #计算出词典,把文档词转化为词对应的key
    newscategory1  = set()
    dir_path = "data_nb"
    files = os.listdir(dir_path)
    for f in files:
        if(os.path.isfile(dir_path+'/'+f)):
            f1 = f[:f.find(".seg.cln.txt")].lstrip("0123456789")
            id = "".join(c for c in f if c.isdigit())
            d = document()
            d.type = f1
            d.id = id
            newscategory1.add(d)
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
                # totalDocNum += 1

    newscategory = list(newscategory1)
    print wordHash.__len__()

    # num=0
    # for item in wordHash1:
    #     num+=1
    #     wordHash[item] = num
    return

def trainmodel(x_train_1docs, y_train_1docs):
    global totalDocNum,wordHash,categoryDocNum,newscategory,defaultP,categoryFeatureP

    for item in y_train_1docs:
        type = item.type
        categoryDocNum[type] += 1
        totalDocNum += 1

    print categoryDocNum
    print totalDocNum
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
    print "defaultP:",defaultP
    print "P:",P
    fwrite = open("categoryFeatureP","w")
    for key, value in categoryFeatureP.items():
        for key1,value1 in value.items():
            if value1>20:
                fwrite.writelines("type:%s  word:%s  value:%s |" % (key,key1,value1))
                fwrite.writelines("\n")

    pppp = {}

    #扫描二维字典,计算出每个分类特征对应的概率,同时计算每个分类的大概率
    for type1, value1 in categoryFeatureP.items():
        for word in wordHash:
            probability = defaultP[type1]
            typeCount = categoryDocNum[type1]
            if word in value1:
                probability = (value1[word]+0.1)/(typeCount+0.2)
                #if math.log(1.0 - probability)<-0.1:
                    #print math.log(1.0 - probability)," type:",type1," word:",word,value1[word]," ",typeCount
                if type1 not in pppp:
                    aaa = document()
                    aaa.num=1
                    aaa.value = math.log(1.0 - probability)
                    pppp[type1] = aaa
                else:
                    pppp[type1].num+=1
                    pppp[type1].value += math.log(1.0 - probability)
                #print probability,value1[word],typeCount,word,type
                value1[word] = math.log(probability) - math.log(1.0 - probability)
                #if value1[word]>0:
                    #print "111111:",value1[word],word,type1
            P[type1] += math.log(1.0 - probability)

    for type,doc in pppp.items():
        print doc.num,type,doc.value
    for type in P:
        print "type:",type," P:",str(P[type])

    fwrite = open("categoryFeatureP1","w")
    for key, value in categoryFeatureP.items():
        for key1,value1 in value.items():
            #if value1>0.3:
            fwrite.writelines("type:%s  word:%s  value:%f |" % (key,key1,value1))
            fwrite.writelines("\n")
    return

def predict ( x_test_1docs, y_test_1docs ):

    global totalDocNum,wordHash,categoryDocNum,newscategory,defaultP,categoryFeatureP
    y_test_pre = []
    notcorrect = 0
    fwrite = open("predict","w")
    for i,map in enumerate(x_test_1docs):

        type = y_test_1docs[i].type
        id  = y_test_1docs[i].id
        #print "|||||||",id
        maxP = -100000000
        pri_type = ""
        for pType,score in P.items():
            scoreP = score
            for key in map:
                if key in categoryFeatureP[pType]:
                    scoreP += categoryFeatureP[pType][key]
                    #print "    ",key,categoryFeatureP[pType][key]
                elif key not in wordHash:
                    continue
                else:
                    scoreP += (math.log(defaultP[pType])-math.log(1-defaultP[pType]))
                    if (math.log(defaultP[pType])-math.log(1-defaultP[pType])) > 0:
                        print id,key,(math.log(defaultP[pType])-math.log(1-defaultP[pType])),pType
            if scoreP > maxP:
                pri_type = pType
                maxP = scoreP
                #print "-------:",pType,id,maxP
        if pri_type != type:
            #print pri_type,type
            notcorrect+=1
            fwrite.writelines("id:%s type:%s predict:%s,%f\n" % (id,type,pri_type,maxP))
        y_test_pre.append(pri_type)

    print "Accuracy:",(len(x_test_1docs) -notcorrect) / len(x_test_1docs)

    return

doc2vec()
X_train_docs, X_test_docs, y_train_docs, y_test_docs = train_test_split(docs, docs_y, test_size=0.01)
#print X_train_docs[100], "\n" , y_train_docs[100]
trainmodel(X_train_docs, y_train_docs)
predict(X_test_docs, y_test_docs)
