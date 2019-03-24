import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

def gen_data(num):
    x_col1 = np.random.uniform(-1,0,num)[:,np.newaxis]
    x_col2 = np.random.uniform(0,1,num)[:,np.newaxis]
    label_left = np.tile(np.array([1,0]),(num,1))
    #print (label_left)

    x_row1 = np.random.uniform(0,1,num)[:,np.newaxis]
    x_row2 = np.random.uniform(0,1,num)[:,np.newaxis]
    label_right = np.tile(np.array([0,1]),(num,1))
    #print (label_right)

    x_col = np.hstack((x_col1,x_col2))
    x_row = np.hstack((x_row1,x_row2))
    data = np.vstack((x_col,x_row))
    label= np.vstack((label_left,label_right))
    return data,label

def add_layer(input,insize,outsize,active_fun = None):
    W = tf.Variable(tf.random_normal([insize,outsize]))
    b = tf.Variable(tf.zeros([1,outsize]) + 0.1)
    W_summary = tf.summary.histogram("W",W)
    b_summary = tf.summary.histogram("b",b)
    Wx_plus_b = tf.matmul(input,W)+b 
    if active_fun == None:
        return Wx_plus_b
    else :
        return active_fun(Wx_plus_b)

def get_train_test(input,step):
    test = input[::step]
    train = []
    i = 1
    while i < len(input):
        train.extend(input[i:(i+step-1)])
        i+=step
    return train,test

data_ph = tf.placeholder(tf.float32,[None,2])
label_ph = tf.placeholder(tf.float32,[None,2])

hider_layer = add_layer(data_ph,2,10,tf.nn.relu)
out_layer = add_layer(hider_layer,10,2,None)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label_ph,logits = out_layer))
#acc,acc_op = tf.metrics.accuracy(label_ph,out_layer)
acc_op = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(label_ph,1),tf.arg_max(out_layer,1)),tf.float32))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

loss_summary = tf.summary.scalar("loss",loss)
acc_summary = tf.summary.scalar("acc",acc_op)

init_op = tf.global_variables_initializer()
init_op_for_acc = tf.local_variables_initializer()
sess = tf.Session()
summary = tf.summary.merge_all()
if os.path.exists("logs"):
    #os.removedirs("logs")
    shutil.rmtree("logs")
writer = tf.summary.FileWriter("logs",sess.graph)
saver = tf.train.Saver()
sess.run(init_op)
sess.run(init_op_for_acc)
data,label = gen_data(5000)
length = len(data)
train_data,test_data = get_train_test(data,5)
train_label,test_label = get_train_test(label,5)

for i in range(500):
    _,summary_val = sess.run([train,summary],feed_dict={data_ph:train_data,label_ph:train_label})
    writer.add_summary(summary_val,i)
    if i % 50 == 0:
        loss_val,acc_val = sess.run([loss,acc_op],feed_dict={data_ph:train_data,label_ph:train_label})
        saver.save(sess,"model/model.ckpt",i)
        print(loss_val,acc_val)

model_file = tf.train.latest_checkpoint("model/")
saver.restore(sess,model_file)

loss_val,acc_val = sess.run([loss,acc_op],feed_dict={data_ph:test_data,label_ph:test_label})
#print(loss_val,acc_val)
predict = sess.run(out_layer,feed_dict={data_ph:test_data})
prelabel = np.argmax(predict,axis=1)[:,np.newaxis]

#result = np.hstack((test_data,np.argmax(predict,axis=1)[:,np.newaxis]))
#print(result)
#print(test_data)
#print(predict)
#print(np.argmax(predict,axis = 1))
pred1 = []
pred0 = []
pred_diff = []
test_label = np.argmax(test_label,axis=1)[:,np.newaxis]
#print(test_data)
#print(prelabel)
for _test_data,_prelabel,_truelabel in zip(test_data,prelabel,test_label):
    if _prelabel == 1 and _prelabel == _truelabel:
        pred1.append(_test_data.tolist())
    elif _prelabel == 0 and _prelabel == _truelabel:
        #print(_test_data)
        pred0.append(_test_data.tolist())
    if  _prelabel != _truelabel:
        pred_diff.append(np.hstack((_test_data,_truelabel,_prelabel)).tolist())
pred0 = np.array(pred0)
pred1 = np.array(pred1)
pred_diff = np.array(pred_diff)
print(pred_diff)
#plt.scatter(data[:len(data)//2,0],data[:len(data)//2,1])
#plt.scatter(data[len(data)//2:length,0],data[len(data)//2:length,1])
plt.scatter(pred1[:,0],pred1[:,1])
plt.scatter(pred0[:,0],pred0[:,1])
plt.scatter(pred_diff[:,0],pred_diff[:,1],c = 'r')
plt.show()


