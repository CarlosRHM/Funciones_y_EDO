
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np


# el dominio de la funcion es [-1,1]
x = tf.linspace(-1,1,500)
y = 3*np.sin(np.pi*x) #La funcion a reproducir

model = Sequential()
model.add(Dense(20, activation='tanh',input_shape=(1,)))
model.add(Dense(20, activation ='tanh'))
model.add(Dense(1, activation = 'linear'))

model.summary()

model.compile(optimizer=RMSprop(),loss='mse')

history = model.fit(x,y,epochs=500,batch_size=10)

f = model.predict(x)
plt.plot(x,f,label='$f(x)=3\sin(\pi x)$ red entrenada',color='tab:cyan')
plt.plot(x,y,label='$f(x)=3\sin(\pi x)$ funcion de prueba',color = 'r',ls='dotted')
plt.legend()
plt.show()

model.save('3sin(pix).h5')
