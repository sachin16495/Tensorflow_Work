#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


plt.ion()


# In[3]:


n_observations = 100
fig, ax = plt.subplots(1, 1)
xs=np.linspace(-3,3,n_observations)
ys=np.sin(xs)+np.random.uniform(-0.5,0.5,n_observations)
ax.scatter(xs, ys)
fig.show()
plt.draw()


# In[4]:


X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)


# In[5]:


W=tf.Variable(tf.random_normal([1]), name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')


# Y=WX+b

# In[6]:


Y_pred = tf.add(tf.multiply(X, W), b)


# In[7]:


cost=tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_observations - 1)


# In[8]:


learning_rate = 0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# In[9]:


n_epochs = 1000


# In[11]:





# In[12]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prev_training_cost = 0.0
    for ep in range(n_epochs):
        for (x, y) in zip(xs, ys):
                sess.run(optimizer,feed_dict={X:xs,Y:ys})
        training_cost=sess.run(cost,feed_dict={X:xs,Y:ys})
        print(training_cost)
        if ep % 20 == 0:
            #print()
            print("Pridicted")
            #print(Y_pred.eval(feed_dict={X: xs}, session=sess))
            #print(Y_pred.eval(feed_dict={X: x}, session=sess))
            ax.plot(xs, Y_pred.eval(feed_dict={X: xs}, session=sess),'k', alpha=(ep / n_epochs))
            fig.show()
            plt.draw()
            
fig.show()
plt.show()
                


# In[ ]:




