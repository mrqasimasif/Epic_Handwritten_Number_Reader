#!/usr/bin/env python
# coding: utf-8

# In[28]:


import tensorflow as tf

mnist = tf.keras.datasets.mnist #28x28 images of hand written images 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
             )
model.fit(x_train, y_train, epochs = 5)


# In[29]:


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)


# In[23]:


import matplotlib.pyplot as plt

plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()

#print(x_train[0])


# In[30]:


model.save('epicNumReader.model')


# In[39]:


new_model = tf.keras.models.load_model('epicNumReader.model')


# In[41]:


predictions = new_model.predict(x_test)


# In[42]:


print(predictions)


# In[45]:


plt.imshow(x_test[1])
plt.show()


# In[44]:


import numpy as np
print(np.argmax(predictions[1]))

