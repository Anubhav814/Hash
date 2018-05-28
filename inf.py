import pickle as pkl 
import numpy as np
import tensorflow as tf 
import re
from tensorflow.contrib import rnn
from copynet import CopyNetWrapper
with open("/home/shrey/Desktop/hash/copynet/pickle_data.txt",'r') as file:
	dic = pkl.load(file)
with open("/home/shrey/Desktop/hash/copynet/pickle_embed.txt",'r') as file:
	dic2 = pkl.load(file)


embeddings=dic2['embeddings']
word2int=dic2['word2int']
int2word=dic2['int2word']
word2int1 = dict(word2int)
words = word2int.keys()
int2word1 = dict(int2word)
delta = []
x = []
t = []
with open('/home/shrey/Desktop/hash/copynet/text.txt','r') as file:
	alpha = file.read().splitlines()
	for beta in alpha:
		gamma = re.sub('[^a-z\ ]+','',beta.lower())
		for word in gamma.split():
			if word in word2int:
				x.append(word2int[word])
			if word not in word2int:
				delta.append(word)
				x.append(word2int['UNK'])
n1 = len(word2int.keys())
for word in delta:
	if word not in word2int.keys():
		word2int1[word]=n1
		int2word1[n1]=word
		n1+=1
with open('/home/shrey/Desktop/hash/copynet/text.txt','r') as file:
	alpha = file.read().splitlines()
	for beta in alpha:
		gamma = re.sub('[^a-z\ ]+','',beta.lower())
		for word in gamma.split():
			t.append(word2int1[word])
			

length = len(x)
len_docs=[min(length,dic['max_len'])]
x = x[:min(dic['max_len']-1,length-1)]
t = t[:min(dic['max_len']-1,length-1)]
length = len(x)
for i in range(length,dic['max_len']):
	x.append(word2int['EOS'])
	t.append(word2int['EOS'])
x = [x]
x = np.array(x)
t = [t]
t = np.array(t)


len_docs=np.array(len_docs)
vocab_size=len(word2int)
beam_width =5
rnn_size = 8
batch_size = np.shape(len_docs)[0]
L1=tf.placeholder('int32',[batch_size])
X = tf.placeholder('int32',[batch_size,dic['max_len']])
T = tf.placeholder('int32',[batch_size,dic['max_len']])




def nn(x,len_docs,t):
	encoder_emb_inp = tf.nn.embedding_lookup(embeddings, x)
	encoder_cell = rnn.GRUCell(rnn_size,name='encoder')
	encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,encoder_emb_inp,sequence_length=len_docs,dtype=tf.float32)
	print encoder_outputs.shape[-1].value
	tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
	tiled_sequence_length = tf.contrib.seq2seq.tile_batch(len_docs, multiplier=beam_width)
	tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)
	tiled_t = tf.contrib.seq2seq.tile_batch(t,multiplier=beam_width)
	print np.shape(tiled_encoder_outputs),np.shape(tiled_encoder_final_state)
	start_tokens = tf.constant(word2int['SOS'], shape=[batch_size])
	print np.shape(start_tokens)
	decoder_cell = rnn.GRUCell(rnn_size,name='decoder')
	attention_mechanism = tf.contrib.seq2seq.LuongAttention(rnn_size,tiled_encoder_outputs,memory_sequence_length=tiled_sequence_length)
	decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,attention_layer_size=rnn_size)
	initial_state = decoder_cell.zero_state(batch_size*beam_width, dtype=tf.float32).clone(cell_state=tiled_encoder_final_state)
	print np.shape(initial_state)
	decoder_cell = CopyNetWrapper(decoder_cell, tiled_encoder_outputs, tiled_t,len(set(delta).union(words)),vocab_size)
	initial_state = decoder_cell.zero_state(batch_size*beam_width, dtype=tf.float32).clone(cell_state=initial_state)
	print np.shape(initial_state)
	# helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,start_tokens,word2int['EOS'])
	# decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state,output_layer=None)
	decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,embedding=embeddings,start_tokens=start_tokens,end_token=word2int['EOS'],initial_state=initial_state,beam_width=beam_width,output_layer=None,length_penalty_weight=0.0)
	outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder,maximum_iterations=7)
	logits = outputs.predicted_ids
	return logits


def answer():
	logits = nn(X,L1,T)
	print tf.trainable_variables()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, '/home/shrey/Desktop/hash/copynet/model' + '/data-all')
		array = (sess.run(logits,feed_dict={X:x,L1:len_docs,T:t}))
		print ([int2word1[i] for i in array]) 


answer()