import numpy as np
import tensorflow as tf
import tflearn
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Add, Subtract, Lambda

class DuelingNetwork:

    def __init__(self, session, dim_state, dim_action, learning_rate, tau=0.01):
        self._sess = session
        self._dim_s = dim_state
        self._dim_a = dim_action
        self._lr = learning_rate

        self._inputs = tflearn.input_data(shape=[None, self._dim_s])

        self._out, self._params = self.buildNetwork(self._inputs, 'dqn')
        self._out_target, self._params_target = self.buildNetwork(self._inputs, 'target')

        self._actions = tf.placeholder(tf.float32, [None, self._dim_a])
        self._y_values = tf.placeholder(tf.float32, [None])#yahan change

        action_q_values = tf.reduce_sum(tf.multiply(self._out, self._actions), reduction_indices=1)#yahan bih

        self._update_target = \
            [t_p.assign(tau * g_p - (1 - tau) * t_p) for g_p, t_p in zip(self._params, self._params_target)]

        self.loss = tflearn.mean_square(self._y_values, action_q_values)
        self.optimize = tf.train.AdamOptimizer(self._lr).minimize(self.loss)

    def buildNetwork(self, state, type):
        with tf.variable_scope(type):

            # w_init = tflearn.initializations.truncated_normal(stddev=1.0)
            # weights_init=w_init,
            net = tflearn.fully_connected(state, 64, activation='relu')
            net = tflearn.fully_connected(net, 32, activation='relu')
            value = Dense(8, activation="relu")(net)
            value = Dense(1, activation="relu")(value)
            advantage = Dense(8, activation="relu")(net)
#            advantage = Dense(self.action_size, activation="relu")(advantage)
            advantage_mean = Lambda(lambda x: K.mean(x, axis=1))(advantage)
            advantage = Subtract()([advantage, advantage_mean])
            net = Add()([value, advantage])


            q_values = tflearn.fully_connected(net, self._dim_a)

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=type)

        return q_values, params

    def train(self, inputs, action, y_values):
        return self._sess.run([self.optimize, self.loss], feed_dict={
            self._inputs: inputs,
            self._actions: action,
            self._y_values: y_values
        })

    def predict(self, inputs):
        return self._sess.run(self._out, feed_dict={
            self._inputs: inputs,
        })

    def predict_target(self, inputs):
        return self._sess.run(self._out_target, feed_dict={
            self._inputs: inputs
        })

    def update_target(self):
        self._sess.run(self._update_target)
        
