import os
import re
import pickle as pkl 
import numpy as np
dic ={}
dic_keys ={}
dic_docs = {}
len_docs = []
with open("/home/shrey/Desktop/hash/copynet/pickle_embed.txt",'r') as file:
	dic2 = pkl.load(file)
word2int=dic2['word2int']
int2word=dic2['int2word']
n = len(word2int.keys())
delta = dic2['delta']
words = dic2['words']
word2int1 = dict(word2int)
int2word1 = dict(int2word)
for word in delta:
	if word not in words:
		word2int1[word]=n
		int2word1[n]=word
		n+=1
print len(word2int)
print len(word2int1)
files = os.listdir("/home/shrey/Desktop/hash/copynet/data")
for file in files:
	if re.search('txt',file) !=None:
		fil = re.sub('txt','key',file)
		with open("/home/shrey/Desktop/hash/copynet/data/"+fil,'r') as w:
			alpha = w.read().splitlines()
			for i,beta in enumerate(alpha):
				dic_keys[file+'_'+str(i)]=beta.lower()

dic['dic_keys']=dic_keys
max_len=0
temp ={}
tdic_docs = {}
for i,file in enumerate(files):
	if re.search('txt',file) !=None:
		lis = []
		tlis = []
		with open("/home/shrey/Desktop/hash/copynet/data/"+file,'r') as w:
			length = 0
			alpha = w.read().splitlines()
			for beta in alpha:
				gamma = re.sub('[^a-z\ ]+','',beta.lower()).split()
				for j,word in enumerate(gamma):
					length+=1
					if word in word2int:
						lis.append(word2int[word])
						tlis.append(word2int[word])
					elif word in word2int1 and word not in word2int:
						lis.append(word2int['UNK'])
						tlis.append(word2int1[word])
					else:
						lis.append(word2int['UNK'])
						tlis.append(word2int['UNK'])
			lis.append(word2int['EOS'])
			tlis.append(word2int['EOS'])
		max_len=max(max_len,length)
		temp[file]=length
		dic_docs[file]=lis
		tdic_docs[file]=tlis
for doc in dic_docs.keys():
	lis = dic_docs[doc]
	tlis = tdic_docs[doc]
	for i in range(len(lis),max_len):
		lis.append(word2int['EOS'])
		tlis.append(word2int['EOS'])
	dic_docs[doc]=lis
	tdic_docs[doc]=tlis

dic['dic_docs']=dic_docs
dic['tdic_docs']=tdic_docs
dic['max_len']=max_len
t = []
x = []
y = []
z=[]
len_keys=[]
max_len2=0
for doc in dic_keys.keys():
	key = dic_keys[doc]
	i=len(key.split())
	i+=1
	max_len2=max(max_len2,i)


for doc in dic_keys.keys():
	key = dic_keys[doc]
	lis = []
	liss=[]
	document = re.sub('_[0-9]+$','',doc)
	x.append(dic_docs[document])
	t.append(tdic_docs[document])
	len_docs.append(temp[document])
	liss.append(word2int['SOS'])
	for i,word in enumerate(key.lower().split()):
		lis.append(word2int1[word])
		if word in word2int.keys():
			liss.append(word2int[word])
		else:
			liss.append(word2int['UNK'])
	lis.append(word2int['EOS'])
	i+=2
	len_keys.append(i)
	for _ in range(i,max_len2):
		lis.append(word2int['EOS'])
		liss.append(word2int['EOS'])
	y.append(lis)
	z.append(liss)



dic['max_len2']=max_len2
dic['x']=x
dic['t']=t
dic['y']=y
dic['z']=z
dic['len_docs']=len_docs
dic['len_keys']=len_keys
with open("/home/shrey/Desktop/hash/copynet/pickle_data.txt",'w') as file:
	pkl.dump(dic,file)