import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#a = tf.constant(1)
#b = tf.Variable(0,dtype=tf.int32)
#b = tf.assign(b,a)
##b = a
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    print(sess.run(a))
#    print(sess.run(b)

#a = tf.constant([[1,2],[3,4]])
#with tf.Session() as sess:
#    b = tf.reduce_max(a)
#    c = tf.reduce_max(a,0)
#    d = tf.reduce_sum(a)
#    e = tf.reduce_mean(a)
#    print(sess.run(c))
#    print(sess.run(c))
#    print(sess.run(b))
#    print(sess.run(a))
#    print(sess.run(d))
#    print(sess.run(e))

#a = tf.Variable(tf.random_normal([2,2]))
##w = tf.Variable([[1,1],[2,2],[3,3]])# 3 row 2 col
#x = tf.Variable([[1],[1]])#2 row 1 col
#b = tf.constant(1)
#y = tf.add(tf.matmul(w,x),b)
#z = tf.zeros(shape=(2,3))
#zz = tf.ones(shape=(2,3))
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    print(sess.run(y))
#    print(sess.run(z))
#    print(sess.run(zz))


#x =tf.placeholder(tf.float32,[2,2])
#def run(_x):
#    with tf.name_scope("run"):
#        sum_summary = tf.summary.scalar("sum_sum",tf.reduce_sum(_x))
#        #return _x,tf.reduce_sum(_x), tf.nn.softmax(_x)
#        sm_summary = tf.summary.histogram("sm_sum",tf.nn.softmax(x))
#        return sum_summary,sm_summary
#def main():
#    with tf.Session() as sess:
#        x_op = run(x)
#        #summary = tf.summary.merge_all()
#        file_writer =  tf.summary.FileWriter("summary_dir",sess.graph)
#        sess.run(tf.global_variables_initializer())
#        for global_step in range(20):
#            #_x,_sum_x,_sm_x=sess.run(x_op,feed_dict={x:[[1,2],[3,4]]})
#            sum_summary ,sm_summary= sess.run(x_op,feed_dict={x:[[1,2],[3,4]]})
#            ##train_summary = sess.run(summary,feed_dict={x:[[1,2],[3,4]]})
#            ##file_writer.add_summary(train_summary,global_step)
#            file_writer.add_summary(sum_summary,global_step)
#            file_writer.add_summary(sm_summary,global_step)
#            #print(_x)
#            #print(_sum_x)
#            #print(_sm_x)
#        file_writer.close()
#    print("main")
#if __name__ == "__main__":
#    main()

# 
# def variable_scope(name_scope,name):
#     with tf.name_scope(name_scope) as scope:
#         return tf.Variable(name,tf.float32)
# 
# 
# def input(_x):
#     name_scope = "input"
#     W = variable_scope(name_scope,"W")
#     W = tf.random_normal([2,2],stddev = 0.01)
#     b = variable_scope(name_scope,"b")
#     b = tf.random_normal([1,2],stddev = 0.01)
#     return tf.add(tf.matmul(_x,W),b)
# 
# def main():
#     x = tf.placeholder(tf.float32,[None,2])
# 
#     input_op = input(x)
#     
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         _xWb = sess.run(input_op,feed_dict={x:[[1,1],[1,1],[1,1]]})
#         print(_xWb)
#  
# if __name__ == "__main__":
#     main()
#

def add_layer(input,insize,outsize,active_fun = None):
    W = tf.Variable(tf.random_normal([insize,outsize]))
    b = tf.Variable(tf.zeros([1,outsize])) 
    Wx_b = tf.matmul(input,W) + b
    if active_fun == None:
        return Wx_b
    else: 
        return active_fun(Wx_b)


def gen_data(start,stop,count,step):
    x = np.linspace(start,stop,count)[:,np.newaxis]
    y = np.square(x) + np.random.normal(0,0.01,x.shape)
    x_test = x[::step]
    y_test = y[::step]

    x_train = []
    y_train = []
    count = 0
    for _x,_y in zip(x,y):
        if count % step != 0:
            x_train.append(_x)
            y_train.append(_y)
        count +=1
    return x,x_train,x_test,y,y_train,y_test



sess = tf.Session()
x_ph = tf.placeholder(tf.float32,[None,1])
y_ph = tf.placeholder(tf.float32,[None,1])
l1 = add_layer(x_ph,1,30,tf.tanh)
l2 = add_layer(l1,30,1,tf.tanh)
loss = tf.reduce_mean(tf.square(y_ph-l2))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#x,x_train,x_test,y,y_train,y_test = gen_data(-1,1,10,5)
x,x_train,x_test,y,y_train,y_test = gen_data(-0.5,0.5,500,5)

#plt.plot(x,y)
#plt.show()

#print(x,x_train,x_test,y,y_train,y_test)
sess.run(tf.global_variables_initializer())
for i in range(2000):
    sess.run(train,feed_dict={x_ph:x,y_ph:y})
    if i % 50==0:
        print(sess.run(loss,feed_dict={x_ph:x,y_ph:y}))

_l2 = sess.run(l2,feed_dict={x_ph:x})
plt.scatter(x,y)
plt.plot(x,_l2,'r')
plt.show()
    


