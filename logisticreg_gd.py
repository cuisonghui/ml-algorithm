# -*- coding:utf8 -*-
from __future__ import division
import math

doc_x = []  # vector,文档向量,每一维为文档词频,暂时不用weight,利用余弦
w_new = []
w_old = []
alpha = 0.01
beta = 0.5
mininterval = 1.0e-10

def loadData():
    file = open("LRTrainNew.txt")
    fileList = file.readlines()
    for fileline in fileList:
        lists = fileline.split(' ')
        x = []
        for word in lists:
            x.append(float(word))
        x.append(1)
        x.append(0)
        doc_x.append(x)
    print len(doc_x)
    return

def calexpwx(w,expwx):

    pos = 2
    for x in doc_x:
        value=0.0
        for i in range(len(w)):
            value += w[i]*x[pos+i]

        value = math.exp(value)
        tmpvalue = value/(1+value)
        #if tmpvalue == 1.0:
            #print "hahahaha",value,x,w
        expwx.append(tmpvalue)
    return

def calexpwx_docx(w):
    pos = 2
    for x in doc_x:
        x[-1] = 0
        for i in range(len(w)):
            x[-1] += w[i]*x[pos+i]
        #print x[-1]
        f = math.exp(x[-1])
        x[-1] = f/(1+f)
    return

def calGradient():
    global doc_x
    gradient = [0] * (len(doc_x[0]) - 3)
    for x in doc_x:
        value = x[0] - (x[0] + x[1]) * x[-1]
        for i in range(len(x) - 3):
            gradient[i] -= x[2 + i] * value
    return gradient

def calgranorm2(g):
    value = 0
    for v in g:
        value += v*v
    return value

def getneww(w_old,t,d):
    wnew = []
    v=0.0
    for i in range(len(d)):
        v += d[i]*d[i]
    v = math.sqrt(v)

    for i in range(len(w_old)):
        value = w_old[i]-t*d[i]/v
        wnew.append(value)
    return wnew

def lfun(w, expwx):
    global doc_x
    value = 0
    for i, x in enumerate(doc_x):
        try:
            value -= (x[0] * math.log(expwx[i]) + x[1] * math.log(1-expwx[i]))
        except:
            continue
            #print "hahahaha:",expwx[i]
    return value

def lfun_one(w):
    global doc_x
    value = 0
    for x in doc_x:
        value -= (x[0] * math.log(x[-1]) + x[1] * math.log(1-x[-1]))
    return value

def calinterval(gradient,w_old,w_new):
    value = 0.0
    for i in range(len(gradient)):
        value += abs(gradient[i]*gradient[i])
    return math.sqrt(value)
    # for i in range(len(w_old)):
    #     value += abs(gradient[i]*(w_old[i] - w_new[i]))
    # return value

def fit():
    global doc_x,w_new,w_old,alpha,beta,mininterval
    w_old = [0.1] * (len(doc_x[0]) - 3)
    calexpwx_docx(w_old)
    lwold = lfun_one(w_old)
    lwnew = 0
    inter_maxk=50
    max_k = 500
    stepnum = 0

    while stepnum<max_k:
        stepnum += 1
        gradient = calGradient()

        t = 0.1
        lwold = lfun_one(w_old)
        lwnew = 0

        print "<<<<<<<gradient:",gradient
        interstep=0
        while 1:#and abs(lwold - lwnew) > mininterval:#
            t /= 2
            interstep+=1
            w_new = getneww(w_old, t, gradient)

            expwx = []
            calexpwx(w_new, expwx)
            lwnew = lfun(w_new, expwx)

            interval = alpha *calinterval(gradient,w_old,w_new)
            print "         ------weis new:",w_new
            print "         ------",lwold,lwnew,lwold-lwnew,interval

            if lwold - lwnew > interval or interstep > inter_maxk:
                break

        print "<<<<<<<"
        print lwold,lwnew
        if abs(lwold - lwnew) <= mininterval:
            w_old = w_new
            break

        w_old = w_new
        lwold = lwnew

        calexpwx_docx(w_old)

        print stepnum
        print w_old
    return

def predict(doc):
    rate = 0.0
    for i in range(len(w_old)):
        rate += w_old[i]*doc[i]

    value = math.exp(rate)
    rate = value/(1+value)
    return rate

loadData()
fit()
print "weis:",w_old
print "rate:",predict(doc_x[10][2:])