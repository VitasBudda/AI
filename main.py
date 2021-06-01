import tensorflow as tf
import numpy as np
import cv2


#get the mnist dataset from keras
mnist = tf.keras.datasets.mnist


#divide the dataset into training and testing
(x_train,y_train), (x_test, y_test)=mnist.load_data()


#normalize
x_train=tf.keras.utils.normalize(x_train, axis=1)
x_test=tf.keras.utils.normalize(x_test, axis=1)


#build the neural network
#input layers
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

#hidden layers
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


#define parameters for training
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#train the model
model.fit(x_train, y_train,epochs=3)


#calculate validation loss and validation accuracy(out of sample)
# val_loss, val_acc=model.evaluate(x_test, y_test)


img = cv2.imread('test.jpg', 0)
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
th3 = cv2.subtract(255, th3)
pred_img = th3
pred_img = cv2.resize(pred_img, (28, 28))
pred_img = pred_img / 255.0
pred_img = pred_img.reshape(1,784)

#Testing
predictions=model.predict(np.array(x_test))
print('Symbol is: ', np.argmax(predictions[1]))