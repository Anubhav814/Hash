import pickle as pkl 
import numpy as np
import tensorflow as tf 
from tensorflow.contrib import rnn
from copynet import CopyNetWrapper
with open("/home/shrey/Desktop/hash/copynet/pickle_data.txt",'r') as file:
	dic = pkl.load(file)
with open("/home/shrey/Desktop/hash/copynet/pickle_embed.txt",'r') as file:
	dic2 = pkl.load(file)

embeddings=dic2['embeddings']
word2int=dic2['word2int']
delta = dic2['delta']
words = dic2['words']
print len(words)
print len((delta).union(words))
len_docs=dic['len_docs']
len_keys=dic['len_keys']
x=dic['x']
t=dic['t']
y=dic['y']
z = dic['z']

x=np.array(x)
t=np.array(t)
y=np.array(y)
z=np.array(z)
len_keys=np.array(len_keys)
len_docs=np.array(len_docs)


vocab_size=len(word2int)
print dic['max_len']
rnn_size = 4
batch_size = 5
nm_epochs =2

L1=tf.placeholder('int32',[None])
L2=tf.placeholder('int32',[None])
X = tf.placeholder('int32',[None,dic['max_len']])
T = tf.placeholder('int32',[None,dic['max_len']])
Y = tf.placeholder('int32',[None,dic['max_len2']])
Z = tf.placeholder('int32',[None,dic['max_len2']])
b=tf.placeholder('int32',[None])
d=tf.placeholder('int32',[None])
a = tf.placeholder('int32',[None,dic['max_len']])
g = tf.placeholder('int32',[None,dic['max_len']])
c = tf.placeholder('int32',[None,dic['max_len2']])
f = tf.placeholder('int32',[None,dic['max_len2']])





n = np.shape(len_keys)[0]
encoder_emb_inp = tf.nn.embedding_lookup(embeddings, a)
decoder_emb_inp = tf.nn.embedding_lookup(embeddings, f)
encoder_cell = rnn.GRUCell(rnn_size,name='encoder')
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,encoder_emb_inp,sequence_length=b,dtype=tf.float32)
print np.shape(encoder_outputs),np.shape(encoder_state)
decoder_cell = rnn.GRUCell(rnn_size,name='decoder')
attention_mechanism = tf.contrib.seq2seq.LuongAttention(rnn_size,encoder_outputs,memory_sequence_length=b)
decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,attention_layer_size=rnn_size)
initial_state = decoder_cell.zero_state(tf.shape(a)[0], dtype=tf.float32).clone(cell_state=encoder_state)
decoder_cell = CopyNetWrapper(decoder_cell, encoder_outputs, g,len((delta).union(words)),vocab_size,sequence_length=b)
helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp,d)
initial_state = decoder_cell.zero_state(tf.shape(a)[0], dtype=tf.float32).clone(cell_state=initial_state)
print np.shape(initial_state)
decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,helper,initial_state,output_layer=None)
outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder)
logits = outputs.rnn_output
print np.shape(logits)
def train():
	dataset = tf.data.Dataset.from_tensor_slices((X,Y,L1,L2,Z,T))
	dataset.shuffle(666666)
	batched_dataset = dataset.batch(batch_size)
	iterator = batched_dataset.make_initializable_iterator()
	next_element = iterator.get_next()
	all_saver = tf.train.Saver()
	labels =tf.one_hot(c,len((delta).union(words)),on_value=1.0, off_value=0.0,axis=-1)
	crossent = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
	print np.shape(crossent)
	weights=tf.sequence_mask(b,dtype=tf.float32)
	loss = (tf.reduce_sum(crossent * tf.transpose(weights))/batch_size)
	params = tf.trainable_variables()
	print params
	gradients = tf.gradients(loss, params)
	clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)
	optimizer = tf.train.AdamOptimizer(0.1)
	update_step = optimizer.apply_gradients(zip(clipped_gradients, params))
	
	with tf.Session() as sess:
		print 'yay'
		sess.run(tf.global_variables_initializer())
		for epoch in range(1,nm_epochs+1):
			epoch_loss = 0
			sess.run(iterator.initializer,feed_dict={X:x,Y:y,L1:len_docs,L2:len_keys,Z:z,T:t})
			for _ in range(int(n/batch_size)):
				a1,a2,a3,a4,a6,a5=sess.run(next_element)
				_,co = sess.run([update_step,loss],feed_dict={a:a1,b:a3,c:a2,d:a4,f:a6,g:a5})
				epoch_loss += co
			print("Epoch",epoch,'completed out of',nm_epochs,'loss:',epoch_loss)
		all_saver.save(sess, '/home/shrey/Desktop/hash/copynet/model' + '/data-all')

train()