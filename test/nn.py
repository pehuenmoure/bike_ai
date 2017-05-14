import tensorflow as tf
import numpy as np
from variables import *

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1


# Create model
def multilayer_perceptron(x, weight, bias):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weight['h1']), bias['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weight['h2']), bias['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weight['out']) + bias['out']
    return out_layer

class NeuralNet(object):
    def __init__(self):# tf Graph input
        self.x = tf.placeholder("float", [None, n_input])
        self.y = tf.placeholder("float", [None, n_classes])

        self.w = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
        }
        self.b = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        self.model = tf.nn.softmax(multilayer_perceptron(self.x, self.w, self.b))


        # Define loss and optimizer
        # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.y))
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def predict(self, array):
        index = self.sess.run(tf.argmax(self.model,1), feed_dict={self.x: array})[0]
        move = [-1,0,1]
        return move[index]

    def setW(self,w):
        assign1 = self.w['h1'].assign(w['h1'])
        assign2 = self.w['h2'].assign(w['h2'])
        assign = self.w['out'].assign(w['out'])
        self.sess.run(assign)
        self.sess.run(assign1)
        self.sess.run(assign2)

    def setB(self,b):
        assign1 = self.b['b1'].assign(b['b1'])
        assign2 = self.b['b2'].assign(b['b2'])
        assign = self.b['out'].assign(b['out'])
        self.sess.run(assign)
        self.sess.run(assign1)
        self.sess.run(assign2)

    def getW(self):
        return self.sess.run(self.w)

    def getB(self):
        return self.sess.run(self.b)


if __name__ == '__main__':
    m = NeuralNet()
    m.predict(np.array([1.,2.,3.,4.,5.,1.,2.]).reshape(1,7))

    w = {
        'h1': np.random.rand(n_input, n_hidden_1),
        'h2': np.random.rand(n_hidden_1, n_hidden_2),
        'out': np.random.rand(n_hidden_2, n_classes),
    }
    m.setW(w)
    p = m.predict(np.array([1.,2.,3.,4.,5.,1.,2.]).reshape(1,7))
