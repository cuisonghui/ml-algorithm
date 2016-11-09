#! c:\python27\

import sys
import os
import random
import math

inpath = "../train-data"
outFile = "_plsa_out_"
DocList = []
DocNameList = []
WordDic = {}
WordList = []

nIteration = 30
Epsilon = 0.01
K = 3
N = 10000
W = 10000
TotalWordCount = 0
pwz = []
pzd = []
newpwz = []

def LoadData():
  global pwz
  global pzd
  global N
  global W
  global TotalWordCount
  i = 0
  for filename in os.listdir(inpath):
    if filename.find(".txt") == -1:
      continue
    i += 1
    infile = file(inpath+'/'+filename, 'r')
    DocNameList.append(filename)
    content = infile.read().strip()
    content = content.decode("utf-8")
    words = content.replace('\n', ' ').split(' ')
    newdic = {}
    widlist = []
    freqlist = []
    wordNum = 0
    for word in words:
      if len(word.strip()) < 1:
        continue
      if word not in WordDic:
        WordDic[word] = len(WordList)
        WordList.append(word)
      wid = WordDic[word]
      wordNum += 1
      if wid not in newdic:
        newdic[wid] = 1.0
      else:
        newdic[wid] += 1.0
    for (wid, freq) in newdic.items():
      widlist.append(wid)
      freqlist.append(freq)
    DocList.append((widlist, freqlist))
    TotalWordCount += wordNum
  N = len(DocList)
  W = len(WordList)
  pwz = [[]]*K
  pzd = [[]]*N
  for i in range(K):
    pwz[i] = [0.0]*W
  for i in range(N):
    pzd[i] = [0.0]*K
  print len(DocList), "files loaded!"
  print len(WordList), "unique words in total!"
  print TotalWordCount, "occurrences in total!"

def Init():
  global pwz
  global pzd
  for i in range(K):
    tempsum = 0.0
    for j in range(W):
      pwz[i][j] = random.random()
      tempsum += pwz[i][j]
    for j in range(W):
      pwz[i][j] /= tempsum
  for i in range(N):
    tempsum = 0.0
    for j in range(K):
      pzd[i][j] = random.random()
      tempsum += pzd[i][j]
    for j in range(K):
      pzd[i][j] /= tempsum
  print "Init over!"

def EMIterate():
  global pwz
  global newpwz
  global pzd
  global TotalWordCount
  newpwz = [[]]*K
  for i in range(K):
    newpwz[i] = [0.0]*W
  did = 0
  pp = 0.0
  while did < len(DocList):
    #if did % 100 == 0:
    #	print did,"docs processed!"
    newpzd = [0.0]*K
    doc = DocList[did]
    ndoc = 0
    for index in range(len(doc[0])):
      #compute posterior probability
      word = doc[0][index]
      ndw = doc[1][index]
      ndoc += ndw
      pzdw = [0.0]*K
      pzdwsum = 0.0
      for z in range(K):
        pzdw[z] = pzd[did][z]*pwz[z][word]
        pzdwsum += pzdw[z]
      pp += ndw*math.log(pzdwsum)
      for z in range(K):
        pzdw[z] = pzdw[z]/pzdwsum
        temppzdw = ndw*pzdw[z]
        newpwz[z][word] += temppzdw
        newpzd[z] += temppzdw
    #normalize and update pzd
    for z in range(K):
      pzd[did][z] = newpzd[z]/ndoc
    did += 1
  #normalize and update pwz
  for z in range(K):
    pwzsum = 0.0
    for wid in range(W):
      pwzsum += newpwz[z][wid]
    for wid in range(W):
      newpwz[z][wid] /= pwzsum
  pwz = newpwz
  pp /= TotalWordCount
  #pp = math.exp(-pp)
  return pp

def Learn():
  prepp = -1
  n = 0
  pp = EMIterate()
  print n,"iteration:",pp
  while n < nIteration and math.fabs(prepp-pp) > Epsilon:
    prepp = pp
    pp = EMIterate()
    n += 1
    print n,"iteration:",pp
  print "learning finished!"

def SaveModel():
  #write the pwz pdz into outFile
  TopN = 20
  pwzfile = file(outFile+"pwz.txt",'w')
  pdzfile = file(outFile+"pdz.txt",'w')
  for i in range(K):
    templist = []
    for j in range(W):
      templist.append((j,pwz[i][j]))
    templist.sort(key=lambda x:x[1])
    pwzfile.write("Topic " + str(i))
    pwzfile.write("\n")
    j = 0
    while j < W and j < TopN:
      pwzfile.write("\t"+WordList[templist[j][0]].encode("utf-8"))
      pwzfile.write("\n")
      j += 1
  for i in range(K):
    templist = []
    for j in range(N):
      templist.append((j,pzd[j][i]))
    templist.sort(key=lambda x:x[1])
    pdzfile.write("Topic " + str(i))
    pdzfile.write("\n")
    j = 0
    while j < N and j < TopN:
      pdzfile.write("\t"+DocNameList[templist[j][0]])
      pdzfile.write("\n")
      j += 1
  pwzfile.close()
  pdzfile.close()

#main framework

LoadData()
Init()
Learn()
SaveModel()

