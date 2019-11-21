import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(X_train,y_train),(X_test,y_test) = keras.datasets.imdb.load_data(num_words=10000) # 从 IMDB 上下载 影评
x_train = keras.preprocessing.sequence.pad_sequences(X_train,maxlen=80) # 筛选长度不超过80的影评
print(x_train.shape)

emb = embedding(x_train)
print(emb.shape)

out = rnn(emb[:4])
print(out.shape)
