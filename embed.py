import os
import re
import pickle as pkl 
import numpy as np
import tensorflow as tf
import math
dic ={}
words = []
delta = []
dim_word_emb = 300
num_sampled = 20
batch_size = 128
nm_epochs = 3
# nm_epochs = 5
files = os.listdir("/home/shrey/Desktop/hash/copynet/data")
for file in files:
	if re.search('txt',file) !=None:
		fil = re.sub('txt','key',file)
		with open("/home/shrey/Desktop/hash/copynet/data/"+fil,'r') as w:
			alpha = w.read().splitlines()
			for i,beta in enumerate(alpha):
				gamma = beta.split()
				for word in gamma:
					words.append(word.lower())
					delta.append(word.lower())
		with open("/home/shrey/Desktop/hash/copynet/data/"+file,'r') as w:
			alpha = w.read().splitlines()
			for beta in alpha:
				beta = re.sub('[^a-z\ ]+','',beta.lower())
				gamma = beta.split()
				for word in gamma:
					words.append(word)

qw = {}
for word in words:
	if word in qw:
		qw[word]+=1
	else:
		qw[word]=1
popular_words = sorted(qw, key = qw.get, reverse = True)
popular_words = popular_words[:min(35783,int(len(set(words))*0.6))]
if 'UNK' not in popular_words:
	popular_words.append('UNK')
if 'SOS' not in popular_words:
	popular_words.append('SOS')
if 'EOS' not in popular_words:
	popular_words.append('EOS')
for word in words:
	if word not in popular_words:
		delta.append(word)
words=popular_words
delta = set(delta)
dic['words']=words



word2int = {}
int2word = {}
vocab_size = len(words)
print vocab_size
for i in range(vocab_size):
	word = words[i]
	word2int[word] =i
	int2word[i] = word

x = []
y = []
dic['word2int']=word2int
dic['int2word']=int2word
dic['delta']=delta
for file in files:
	if re.search('txt',file) !=None:
		fil = re.sub('txt','key',file)
		with open("/home/shrey/Desktop/hash/copynet/data/"+fil,'r') as w:
			alpha = w.read().splitlines()
			for i,beta in enumerate(alpha):
				gamma = beta.lower().split()
				n = len(gamma)
				if n > 1:
					for i in range(n):
						if i ==0:
							word = gamma[i]
							if word not in words:
								word = 'UNK'
							target1 = gamma[i+1]
							if target1 not in words:
								target1 = 'UNK'
							x.append(word2int[word])
							y.append([word2int[target1]])
						elif i ==n-1:
							word = gamma[n-1]
							if word not in words:
								word = 'UNK'
							target1 = gamma[n-2]
							if target1 not in words:
								target1 = 'UNK'
							x.append(word2int[word])
							y.append([word2int[target1]])
						else:
							word = gamma[i]
							if word not in words:
								word = 'UNK'
							target1 = gamma[i+1]
							target2 = gamma[i-1]
							if target1 not in words:
								target1 = 'UNK'
							if target2 not in words:
								target2 = 'UNK'
							x.append(word2int[word])
							y.append([word2int[target1]])
							x.append(word2int[word])
							y.append([word2int[target2]])
				if n!=0:
					w1 = gamma[0]
					if w1 not in words:
						w1 = 'UNK'
					w2 = gamma[n-1]
					if w2 not in words:
						w2 = 'UNK'
					x.append(word2int[w1])
					y.append([word2int['SOS']])
					x.append(word2int[w2])
					y.append([word2int['EOS']])
					x.append(word2int['SOS'])
					y.append([word2int[w1]])
					x.append(word2int['EOS'])
					y.append([word2int[w2]])
					
		with open("/home/shrey/Desktop/hash/copynet/data/"+file,'r') as w:
			alpha = w.read().splitlines()
			for beta in alpha:
				beta = re.sub('[^a-z\ ]+','',beta.lower())
				gamma = beta.split()
				n = len(gamma)
				if n > 1:
					for i in range(n):
						if i ==0:
							word = gamma[i]
							if word not in words:
								word = 'UNK'
							target1 = gamma[i+1]
							if target1 not in words:
								target1 = 'UNK'
							x.append(word2int[word])
							y.append([word2int[target1]])
						elif i ==n-1:
							word = gamma[n-1]
							if word not in words:
								word = 'UNK'
							target1 = gamma[n-2]
							if target1 not in words:
								target1 = 'UNK'
							x.append(word2int[word])
							y.append([word2int[target1]])
						else:
							word = gamma[i]
							if word not in words:
								word = 'UNK'
							target1 = gamma[i+1]
							target2 = gamma[i-1]
							if target1 not in words:
								target1 = 'UNK'
							if target2 not in words:
								target2 = 'UNK'
							x.append(word2int[word])
							y.append([word2int[target1]])
							x.append(word2int[word])
							y.append([word2int[target2]])
				
				if n!=0:
					w1 = gamma[0]
					if w1 not in words:
						w1 = 'UNK'
					w2 = gamma[n-1]
					if w2 not in words:
						w2 = 'UNK'
					x.append(word2int[w1])
					y.append([word2int['SOS']])
					x.append(word2int[w2])
					y.append([word2int['EOS']])
					x.append(word2int['SOS'])
					y.append([word2int[w1]])
					x.append(word2int['EOS'])
					y.append([word2int[w2]])
print len(x)
print len(y)
n = len(x)
x = np.array(x)
y = np.array(y)

X = tf.placeholder(tf.int32, shape=[None])
Y = tf.placeholder(tf.int32, shape=[None, 1])
a = tf.placeholder(tf.int32, shape=[None])
b = tf.placeholder(tf.int32, shape=[None, 1])

dataset = tf.data.Dataset.from_tensor_slices((X,Y))
dataset.shuffle(n+1)
batched_dataset = dataset.batch(batch_size)
iterator = batched_dataset.make_initializable_iterator()
next_element = iterator.get_next()

embeddings = tf.Variable(tf.random_uniform([vocab_size,dim_word_emb], -1.0, 1.0))
nce_weights = tf.Variable(tf.truncated_normal([vocab_size, dim_word_emb],stddev=1.0/math.sqrt(dim_word_emb)))
nce_biases = tf.Variable(tf.zeros([vocab_size]))


embed = tf.nn.embedding_lookup(embeddings,a)
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,biases=nce_biases,labels=b,inputs=embed,num_sampled=num_sampled,num_classes=vocab_size))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(1,nm_epochs+1):
		epoch_loss = 0
		sess.run(iterator.initializer,feed_dict={X:x,Y:y})
		for _ in range(int(n/batch_size)):
			a1,a2=sess.run(next_element)
			_,c = sess.run([optimizer,loss],feed_dict={a:a1,b:a2})
			epoch_loss += c
		print("Epoch",epoch,'completed out of',nm_epochs,'loss:',epoch_loss)
	dic['embeddings']=sess.run(embeddings)

with open("/home/shrey/Desktop/hash/copynet/pickle_embed.txt",'w') as file:
	pkl.dump(dic,file)