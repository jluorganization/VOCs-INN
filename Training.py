# -*- coding: utf-8 -*-
"""


"""

import sys

sys.path.insert(0, '../../Utilities/')

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas
import math
import tensorflow as tf
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import datetime
from pyDOE import lhs
from scipy.integrate import odeint
start_time = time.time()


# np.random.seed(1234)
# tf.set_random_seed(1234)
# tf.random.set_seed(1234)

# %%
class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, t_train, I1_new_train, I2_new_train,
                 I1_sum_train, I2_sum_train, U0, t_f, lb, ub, N,
                 layers, layers_beta1, layers_beta2, layers_gamma1, layers_gamma2, sf):

        self.N = N
        self.sf = sf

        # Data for training
        self.t_train = t_train
        self.I1_new_train = I1_new_train
        self.I2_new_train = I2_new_train
        self.I1_sum_train = I1_sum_train
        self.I2_sum_train = I2_sum_train
        self.S0 = U0[0]
        self.I10 = U0[1]
        self.I20 = U0[2]
        self.R0 = U0[3]
        self.t_f = t_f

        # Time division s
        self.M = len(t_f) - 1
        self.tau = t_f[1] - t_f[0]

        # Bounds
        self.lb = lb
        self.ub = ub

        # initialize NN
        self.weights, self.biases = self.initialize_NN(layers)
        self.weights_beta1, self.biases_beta1 = self.initialize_NN(layers_beta1)
        self.weights_beta2, self.biases_beta2 = self.initialize_NN(layers_beta2)
        self.weights_gamma1, self.biases_gamma1 = self.initialize_NN(layers_gamma1)
        self.weights_gamma2, self.biases_gamma2 = self.initialize_NN(layers_gamma2)

        # Fixed parameters
        self.N = N
        self.delta = tf.Variable(0.000023, dtype=tf.float64, trainable=False)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.saver = tf.train.Saver()

        # placeholders for inputs
        self.t_u = tf.placeholder(tf.float64, shape=[None, self.t_train.shape[1]])
        self.I1_new_u = tf.placeholder(tf.float64, shape=[None, self.I1_new_train.shape[1]])
        self.I2_new_u = tf.placeholder(tf.float64, shape=[None, self.I2_new_train.shape[1]])
        self.I1_sum_u = tf.placeholder(tf.float64, shape=[None, self.I1_sum_train.shape[1]])
        self.I2_sum_u = tf.placeholder(tf.float64, shape=[None, self.I2_sum_train.shape[1]])
        self.S0_u = tf.placeholder(tf.float64, shape=[None, self.S0.shape[1]])
        self.I10_u = tf.placeholder(tf.float64, shape=[None, self.I10.shape[1]])
        self.I20_u = tf.placeholder(tf.float64, shape=[None, self.I20.shape[1]])
        self.R0_u = tf.placeholder(tf.float64, shape=[None, self.R0.shape[1]])
        self.t_tf = tf.placeholder(tf.float64, shape=[None, self.t_f.shape[1]])

        # physics informed neural networks
        self.S_pred, self.I1_pred, self.I2_pred, self.R_pred, self.I1_sum_pred, self.I2_sum_pred = self.net_u(
            self.t_u)

        ##
        self.BetaI1_pred = self.net_BetaI1(self.t_u)
        self.BetaI2_pred = self.net_BetaI2(self.t_u)
        self.Gamma1_pred = self.net_Gamma1(self.t_u)
        self.Gamma2_pred = self.net_Gamma2(self.t_u)

        #
        self.I1_new_pred = self.I1_sum_pred[1:, :] - self.I1_sum_pred[0:-1, :]
        self.I2_new_pred = self.I2_sum_pred[1:, :] - self.I2_sum_pred[0:-1, :]

        self.S0_pred = self.S_pred[0]
        self.I10_pred = self.I1_pred[0]
        self.I20_pred = self.I2_pred[0]
        self.R0_pred = self.R_pred[0]

        # self.S_f, self.I1_f, self.I2_f, self.D_f, self.R_f, self.I1_sum_f, self.I2_sum_f = self.net_f(self.t_u)
        self.I1_f, self.I2_f,  self.R_f, self.I1_sum_f, self.I2_sum_f = self.net_f(self.t_u)
        # self.I1_f, self.I2_f, self.D_f, self.R_f, self.I1_sum_f, self.I2_sum_f, self.con_f = self.net_f(self.t_u)

        # loss
        self.lossU0 = tf.reduce_mean(tf.square(self.S0_u - self.S0_pred)) + \
                      tf.reduce_mean(tf.square(self.I10_u - self.I10_pred)) + \
                      tf.reduce_mean(tf.square(self.I20_u - self.I20_pred)) + \
                      tf.reduce_mean(tf.square(self.R0_u - self.R0_pred))

        self.lossU = 100 * tf.reduce_mean(tf.square(self.I1_new_u[1:, :] - self.I1_new_pred)) + \
                     100 * tf.reduce_mean(tf.square(self.I2_new_u[1:, :] - self.I2_new_pred)) + \
                     tf.reduce_mean(tf.square(self.I1_sum_u - self.I1_sum_pred)) + \
                     tf.reduce_mean(tf.square(self.I2_sum_u - self.I2_sum_pred))

        # self.lossF = tf.reduce_mean(tf.square(self.S_f))+tf.reduce_mean(tf.square(self.I1_f)) + \
        #     tf.reduce_mean(tf.square(self.I2_f)) + tf.reduce_mean(tf.square(self.D_f)) + \
        #     tf.reduce_mean(tf.square(self.R_f)) + tf.reduce_mean(tf.square(self.I1_sum_f))+ \
        #     tf.reduce_mean(tf.square(self.I2_sum_f))

        self.lossF = tf.reduce_mean(tf.square(self.I1_f)) + \
                     tf.reduce_mean(tf.square(self.I2_f)) + \
                     tf.reduce_mean(tf.square(self.R_f)) + tf.reduce_mean(tf.square(self.I1_sum_f)) + \
                     tf.reduce_mean(tf.square(self.I2_sum_f))

        # self.lossF = tf.reduce_mean(tf.square(self.con_f))+tf.reduce_mean(tf.square(self.I1_f)) + \
        #     tf.reduce_mean(tf.square(self.I2_f)) + tf.reduce_mean(tf.square(self.D_f)) + \
        #     tf.reduce_mean(tf.square(self.R_f)) + tf.reduce_mean(tf.square(self.I1_sum_f))+ \
        #     tf.reduce_mean(tf.square(self.I2_sum_f))

        self.loss = self.lossU0 + self.lossU + 100*self.lossF

        # Optimizer
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Initialize the nueral network

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])  # weights for the current layer
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64),
                            dtype=tf.float64)  # biases for the current layer
            weights.append(W)  # save the elements in W to weights (a row vector)
            biases.append(b)  # save the elements in b to biases (a 1Xsum(layers) row vector)
        return weights, biases

    # generating weights
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64),
                           dtype=tf.float64)

    # Architecture of the neural network
    def neural_net(self, t, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (t - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, t):
        SI1I2DR = self.neural_net(t, self.weights, self.biases)
        I1 = SI1I2DR[:, 0:1]
        I2 = SI1I2DR[:, 1:2]
        R = SI1I2DR[:, 2:3]
        I1_sum = SI1I2DR[:, 3:4]
        I2_sum = SI1I2DR[:, 4:5]
        S = self.N - I1 - I2 - R
        return S, I1, I2, R, I1_sum, I2_sum

    def net_BetaI1(self, t):
        BetaI1 = self.neural_net(t, self.weights_beta1, self.biases_beta1)
        bound_b = [tf.constant(0.0, dtype=tf.float64), tf.constant(1, dtype=tf.float64)]
        return bound_b[0] + (bound_b[1] - bound_b[0]) * tf.sigmoid(BetaI1)
        # return  tf.sigmoid(BetaI1)

    def net_BetaI2(self, t):
        BetaI2 = self.neural_net(t, self.weights_beta2, self.biases_beta2)
        bound_b = [tf.constant(0.0, dtype=tf.float64), tf.constant(1, dtype=tf.float64)]
        return bound_b[0] + (bound_b[1] - bound_b[0]) * tf.sigmoid(BetaI2)
        # return tf.sigmoid(BetaI2)

    def net_Gamma1(self, t):
        gamma1 = self.neural_net(t, self.weights_gamma1, self.biases_gamma1)
        bound_b = [tf.constant(0.49, dtype=tf.float64), tf.constant(0.51, dtype=tf.float64)]
        return bound_b[0] + (bound_b[1] - bound_b[0]) * tf.sigmoid(gamma1)

    def net_Gamma2(self, t):
        gamma2 = self.neural_net(t, self.weights_gamma2, self.biases_gamma2)
        bound_b = [tf.constant(0.49, dtype=tf.float64), tf.constant(0.51, dtype=tf.float64)]
        return bound_b[0] + (bound_b[1] - bound_b[0]) * tf.sigmoid(gamma2)

    def net_f(self, t):
        # load fixed parameters
        delta = self.delta

        # load time-dependent parameters
        betaI1 = self.net_BetaI1(t)
        betaI2 = self.net_BetaI2(t)
        gamma1 = self.net_Gamma1(t)
        gamma2 = self.net_Gamma2(t)

        # Obtain S,E,I,J,D,H,R from Neural network
        S, I1, I2, R, I1_sum, I2_sum = self.net_u(t)

        # Time derivatives
        # S_t = tf.gradients(S, t, unconnected_gradients='zero')[0]
        I1_t = tf.gradients(I1, t, unconnected_gradients='zero')[0]
        I2_t = tf.gradients(I2, t, unconnected_gradients='zero')[0]
        R_t = tf.gradients(R, t, unconnected_gradients='zero')[0]
        I1_sum_t = tf.gradients(I1_sum, t, unconnected_gradients='zero')[0]
        I2_sum_t = tf.gradients(I2_sum, t, unconnected_gradients='zero')[0]

        # f_S = S_t + ((betaI1 * I1 + betaI2 * I2)/self.N)*S + delta * S
        f_I1 = I1_t - (betaI1 * S * I1) / self.N  + gamma1 * I1
        f_I2 = I2_t - (betaI2 * S * I2) / self.N +  gamma2 * I2
        f_R = R_t - gamma1 * I1 - gamma2 * I2
        f_I1_sum = I1_sum_t - (betaI1 * S * I1) / self.N
        f_I2_sum = I2_sum_t - (betaI2 * S * I2) / self.N
        # f_con = S+I1+I2+D+R-self.N

        return f_I1, f_I2, f_R, f_I1_sum, f_I2_sum
        # return f_I1, f_I2, f_D, f_R, f_I1_sum, f_I2_sum, f_con

        # return f_S, f_I1, f_I2, f_D, f_R, f_I1_sum, f_I2_sum



    def callback(self, loss, lossU0, lossU, lossF):
        total_records_LBFGS.append(np.array([loss, lossU0, lossU, lossF]))
        print('Loss: %.3e, LossU0: %.3e, LossU: %.3e, LossF: %.3e'
              % (loss, lossU0, lossU, lossF))

    def train(self, nIter):

        tf_dict = {self.t_u: self.t_train, self.t_tf: self.t_f,
                   self.I1_new_u: self.I1_new_train, self.I2_new_u: self.I2_new_train,
                   self.I1_sum_u: self.I1_sum_train, self.I2_sum_u: self.I2_sum_train,
                   self.S0_u: self.S0, self.I10_u: self.I10, self.I20_u: self.I20,
                   self.R0_u: self.R0}

        start_time = time.time()
        for it in range(nIter + 1):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lossU0_value = self.sess.run(self.lossU0, tf_dict)
                lossU_value = self.sess.run(self.lossU, tf_dict)
                lossF_value = self.sess.run(self.lossF, tf_dict)
                total_records.append(np.array([it, loss_value, lossU0_value, lossU_value, lossF_value]))
                print('It: %d, Loss: %.3e, LossU0: %.3e, LossU: %.3e, LossF: %.3e, Time: %.2f' %
                      (it, loss_value, lossU0_value, lossU_value, lossF_value, elapsed))
                start_time = time.time()

        if LBFGS:
            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,  # Inputs of the minimize operator
                                    fetches=[self.loss, self.lossU0, self.lossU, self.lossF],
                                    loss_callback=self.callback)  # Show the results of minimize operator

    def predict_data(self, t_star):

        tf_dict = {self.t_u: t_star}

        S = self.sess.run(self.S_pred, tf_dict)
        I1 = self.sess.run(self.I1_pred, tf_dict)
        I2 = self.sess.run(self.I2_pred, tf_dict)
        R = self.sess.run(self.R_pred, tf_dict)
        I1_sum = self.sess.run(self.I1_sum_pred, tf_dict)
        I2_sum = self.sess.run(self.I2_sum_pred, tf_dict)
        I1_new = self.sess.run(self.I1_new_pred, tf_dict)
        I2_new = self.sess.run(self.I2_new_pred, tf_dict)
        return S, I1, I2, R, I1_new, I2_new, I1_sum, I2_sum

    def predict_par(self, t_star):

        tf_dict = {self.t_u: t_star}
        BetaI1 = self.sess.run(self.BetaI1_pred, tf_dict)
        BetaI2 = self.sess.run(self.BetaI2_pred, tf_dict)
        Gamma1 = self.sess.run(self.Gamma1_pred, tf_dict)
        Gamma2 = self.sess.run(self.Gamma2_pred, tf_dict)
        return BetaI1, BetaI2, Gamma1, Gamma2


############################################################
if __name__ == "__main__":

    # Architecture of  the NN
    layers = [1] + 5 * [32] + [5]  # The inout is t while the outputs are S I1 I2 R  I1_sum I2_sum
    layers_beta1 = [1] + 3 * [64] + [1]
    layers_beta2 = [1] + 3 * [64] + [1]
    layers_gamma1 = [1] + 3 * [64] + [1]
    layers_gamma2 = [1] + 3 * [64] + [1]

    # Load data
    data_frame = pandas.read_csv('Data/BC_data.csv')
    I1_new_star = data_frame['I1new']  # T x 1 array
    I2_new_star = data_frame['I2new']
    I1_sum_star = data_frame['I1sum']
    I2_sum_star = data_frame['I2sum']

    I1_new_star = I1_new_star.to_numpy(dtype=np.float64)
    I2_new_star = I2_new_star.to_numpy(dtype=np.float64)
    I1_sum_star = I1_sum_star.to_numpy(dtype=np.float64)
    I2_sum_star = I2_sum_star.to_numpy(dtype=np.float64)
    I1_new_star = I1_new_star.reshape([len(I1_new_star), 1])
    I2_new_star = I2_new_star.reshape([len(I2_new_star), 1])
    I1_sum_star = I1_sum_star.reshape([len(I1_sum_star), 1])
    I2_sum_star = I2_sum_star.reshape([len(I2_sum_star), 1])
    t_star = np.arange(len(I1_new_star))
    t_star = t_star.reshape([len(t_star), 1])
    N = 5.5e6 + 13 + 306
    X0 = [5.5e6, 13, 306, 0, 13, 306]
    X0 = np.array(X0)

    # lower and upper bounds
    lb = t_star.min(0)
    ub = t_star.max(0)

    # Initial conditions
    I10_new = I1_new_star[0:1, :]
    I20_new = I2_new_star[0:1, :]
    I10_sum = I1_sum_star[0:1, :]
    I20_sum = I2_sum_star[0:1, :]

    # Scaling
    sf = 1e-4
    N = N * sf
    I1_new_star = I1_new_star * sf
    I2_new_star = I2_new_star * sf
    I1_sum_star = I1_sum_star * sf
    I2_sum_star = I2_sum_star * sf
    X0 = X0*sf

    # Initial conditions
    S0 = np.array([[5.5e6]]) * sf
    I10 = I1_sum_star[0:1, :]
    I20 = I2_sum_star[0:1, :]
    R0 = np.array([[0.0]]) * sf
    U0 = [S0, I10, I20, R0]
    N_f = 500
    t_f = lb + (ub - lb) * lhs(1, N_f)

    ######################################################################
    ######################## Training and Predicting #####################
    ######################################################################
    t_train = t_star
    I1_new_train = I1_new_star
    I2_new_train = I2_new_star
    I1_sum_train = I1_sum_star
    I2_sum_train = I2_sum_star

    from datetime import datetime

    now = datetime.now()
    dt_string = now.strftime("%m-%d")

    # save results
    current_directory = os.getcwd()
    for j in range(10):
        casenumber = 'set' + str(j + 1)

        relative_path_results = '/Model1/Train-Results-' + dt_string + '-' + casenumber + '/'
        save_results_to = current_directory + relative_path_results
        if not os.path.exists(save_results_to):
            os.makedirs(save_results_to)

        relative_path = '/Model1/Train-model-' + dt_string + '-' + casenumber + '/'
        save_models_to = current_directory + relative_path
        if not os.path.exists(save_models_to):
            os.makedirs(save_models_to)


        ####model
        total_records = []
        total_records_LBFGS = []
        model = PhysicsInformedNN(t_train, I1_new_train, I2_new_train,
                                  I1_sum_train, I2_sum_train, U0, t_f, lb, ub, N,
                                  layers, layers_beta1, layers_beta2, layers_gamma1, layers_gamma2, sf)
        ####Training
        LBFGS=True
        # LBFGS = False
        model.train(10000)  # Training with n iterations

        ####save model
        model.saver.save(model.sess, save_models_to + "model.ckpt")

        ####Predicting
        S, I1, I2, R, I1_new, I2_new, I1_sum, I2_sum = model.predict_data(t_star)
        BetaI1, BetaI2, Gamma1, Gamma2 = model.predict_par(t_star)
        import datetime

        end_time = time.time()
        print(datetime.timedelta(seconds=int(end_time - start_time)))

        ##################save data and plot

        ####save data
        np.savetxt(save_results_to + 'S.txt', S.reshape((-1, 1)))
        np.savetxt(save_results_to + 'I1.txt', I1.reshape((-1, 1)))
        np.savetxt(save_results_to + 'I2.txt', I2.reshape((-1, 1)))
        np.savetxt(save_results_to + 'R.txt', R.reshape((-1, 1)))
        np.savetxt(save_results_to + 'I1_new.txt', I1_new.reshape((-1, 1)))
        np.savetxt(save_results_to + 'I2_new.txt', I2_new.reshape((-1, 1)))
        np.savetxt(save_results_to + 'I1_sum.txt', I1_sum.reshape((-1, 1)))
        np.savetxt(save_results_to + 'I2_sum.txt', I2_sum.reshape((-1, 1)))

        ####save BetaI1, BetaI2  Gamma1 Gamma2
        np.savetxt(save_results_to + 't_star.txt', t_star.reshape((-1, 1)))
        np.savetxt(save_results_to + 'BetaI1.txt', BetaI1.reshape((-1, 1)))
        np.savetxt(save_results_to + 'BetaI2.txt', BetaI2.reshape((-1, 1)))
        np.savetxt(save_results_to + 'Gamma1.txt', Gamma1.reshape((-1, 1)))
        np.savetxt(save_results_to + 'Gamma2.txt', Gamma2.reshape((-1, 1)))

        ####records for Adam
        N_Iter = len(total_records)
        iteration = np.asarray(total_records)[:, 0]
        loss_his = np.asarray(total_records)[:, 1]
        loss_his_u0 = np.asarray(total_records)[:, 2]
        loss_his_u = np.asarray(total_records)[:, 3]
        loss_his_f = np.asarray(total_records)[:, 4]

        ####records for LBFGS
        if LBFGS:
            N_Iter_LBFGS = len(total_records_LBFGS)
            iteration_LBFGS = np.arange(N_Iter_LBFGS) + N_Iter * 100
            loss_his_LBFGS = np.asarray(total_records_LBFGS)[:, 0]
            loss_his_u0_LBFGS = np.asarray(total_records_LBFGS)[:, 1]
            loss_his_u_LBFGS = np.asarray(total_records_LBFGS)[:, 2]
            loss_his_f_LBFGS = np.asarray(total_records_LBFGS)[:, 3]

        ####save records
        np.savetxt(save_results_to + 'iteration.txt', iteration.reshape((-1, 1)))
        np.savetxt(save_results_to + 'loss_his.txt', loss_his.reshape((-1, 1)))
        np.savetxt(save_results_to + 'loss_his_u0.txt', loss_his_u0.reshape((-1, 1)))
        np.savetxt(save_results_to + 'loss_his_u.txt', loss_his_u.reshape((-1, 1)))
        np.savetxt(save_results_to + 'loss_his_f.txt', loss_his_f.reshape((-1, 1)))

        if LBFGS:
            np.savetxt(save_results_to + 'iteration_LBFGS.txt', iteration_LBFGS.reshape((-1, 1)))
            np.savetxt(save_results_to + 'loss_his_LBFGS.txt', loss_his_LBFGS.reshape((-1, 1)))
            np.savetxt(save_results_to + 'loss_his_u0_LBFGS.txt', loss_his_u0_LBFGS.reshape((-1, 1)))
            np.savetxt(save_results_to + 'loss_his_u_LBFGS.txt', loss_his_u_LBFGS.reshape((-1, 1)))
            np.savetxt(save_results_to + 'loss_his_f_LBFGS.txt', loss_his_f_LBFGS.reshape((-1, 1)))


        ############################# Plotting ###############################
        ######################################################################
        SAVE_FIG = True

        # History of loss
        font = 24
        fig, ax = plt.subplots()
        plt.locator_params(axis='x', nbins=6)
        plt.locator_params(axis='y', nbins=6)
        plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
        plt.xlabel('$iteration$', fontsize=font)
        plt.ylabel('$loss values$', fontsize=font)
        plt.yscale('log')
        plt.grid(True)
        plt.plot(iteration, loss_his, label='$loss$')
        plt.plot(iteration, loss_his_u0, label='$loss_{u0}$')
        plt.plot(iteration, loss_his_u, label='$loss_u$')
        plt.plot(iteration, loss_his_f, label='$loss_f$')
        if LBFGS:
            plt.plot(iteration_LBFGS, loss_his_LBFGS, label='$loss-LBFGS$')
            plt.plot(iteration_LBFGS, loss_his_u0_LBFGS, label='$loss_{u0}-LBFGS$')
            plt.plot(iteration_LBFGS, loss_his_u_LBFGS, label='$loss_u-LBFGS$')
            plt.plot(iteration_LBFGS, loss_his_f_LBFGS, label='$loss_f-LBFGS$')
        plt.legend(loc="upper right", fontsize=24, ncol=4)
        plt.legend()
        ax.tick_params(axis='both', labelsize=24)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'History_loss.png', dpi=300)


        # Infections 1
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, I1 / sf, 'r-', lw=2, label='VOCs-INN')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('Current infections ($I_{1}$)', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'Current_infections1_VOCs-INN.png', dpi=300)

        # Infections 2
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, I2 / sf, 'r-', lw=2, label='VOCs-INN')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('Current infections ($I_{2}$)', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'Current_infections2_VOCs-INN.png', dpi=300)

        # New cases
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, I1_new_star / sf, 'k--', marker='o', lw=2, markersize=5, label='Data-Weekly')
        ax.plot(t_star[1:], I1_new / sf, 'r-', lw=2, label='VOCs-INN')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('New cases ($I_{1}^{new}$)', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'new1_cases_VOCs-INN.png', dpi=300)


        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, I2_new_star / sf, 'k--', marker='o', lw=2, markersize=5, label='Data-Weekly')
        ax.plot(t_star[1:], I2_new / sf, 'r-', lw=2, label='VOCs-INN')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('New cases ($I_{2}^{new}$)', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'new2_cases_VOCs-INN.png', dpi=300)

        # Cumulative  cases
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, I1_sum_star / sf, 'k--', marker='o', lw=2, markersize=5, label='Data-Weekly')
        ax.plot(t_star, I1_sum / sf, 'r-', lw=2, label='VOCs-INN')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('Cumulative cases ($I_{1}^{c}$)', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'Cumulative_cases1_VOCs-INN.png', dpi=300)

        # cumulative confirmed Cases2
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, I2_sum_star / sf, 'k--', marker='o', lw=2, markersize=5, label='Data-Weekly')
        ax.plot(t_star, I2_sum / sf, 'r-', lw=2, label='VOCs-INN')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('Cumulative cases ($I_{2}^{c}$)', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'Cumulative_cases2_VOCs-INN.png', dpi=300)

        # BetaI1 curve
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, BetaI1, 'r-', lw=2, label='VOCs-INN')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('$Beta_{1}$', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'Beta1.png', dpi=300)

        # BetaI2 curve
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, BetaI2, 'r-', lw=2, label='VOCs-INN')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('$Beta_{2}$', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'Beta2.png', dpi=300)

        # Gamma1 curve
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, Gamma1, 'r-', lw=2, label='VOCs-INN')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('$Gamma_{1}$', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'Gamma1.png', dpi=300)

        # Gamma2 curve
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, Gamma2, 'r-', lw=2, label='VOCs-INN')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('$Gamma_{2}$', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'Gamma2.png', dpi=300)

        # Current SI1I2DR
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, S / sf, lw=2, label='VOCs-INN')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('$S$', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'Current_S_VOCs-INN.png', dpi=300)

        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, R / sf, lw=2, label='PINN')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('$R$', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'Current_R_VOCs-INN.png', dpi=300)



        ######## ode
        def Par_fun(t):
            t = np.array(t)
            t = t.reshape([1, 1])
            BetaI1, BetaI2, Gamma1, Gamma2 = model.predict_par(t)
            return BetaI1,BetaI2, Gamma1, Gamma2


        def ODEs_mean(X, t):
            BetaI1_NN,BetaI2_NN,Gamma1_NN,Gamma2_NN=Par_fun(t)
            S, I1, I2, R, sumI1, sumI2 = X
            dSdt = -( BetaI1_NN) * S * I1 / N - (
                        BetaI2_NN ) * S * I2 / N
            dI1dt = (BetaI1_NN) * S * I1 / N  - (Gamma1_NN) * I1
            dI2dt = (BetaI2_NN) * S * I2 / N - (Gamma2_NN) * I2
            dRdt = (Gamma1_NN) * I1 + (Gamma2_NN) * I2
            dsumI1dt = (BetaI1_NN) * S * I1 / N
            dsumI2dt = (BetaI2_NN) * S * I2 / N
            return [float(dSdt), float(dI1dt), float(dI2dt), float(dRdt), float(dsumI1dt), float(dsumI2dt)]

        Sol = odeint(ODEs_mean, X0, t_star.flatten())
        S1 = Sol[:, 0]
        I11 = Sol[:, 1]
        I21 = Sol[:, 2]
        R1 = Sol[:, 3]
        ISUM1 = Sol[:,4]
        ISUM2 = Sol[:, 5]
        INEW1 = np.diff(ISUM1)
        INEW2 = np.diff(ISUM2)

        S1 = S1.reshape([len(S1), 1])/sf
        I11 = I11.reshape([len(I11), 1])/sf
        I21 = I21.reshape([len(I21), 1])/sf
        R1 = R1.reshape([len(R1), 1])/sf
        ISUM1 = ISUM1.reshape([len(ISUM1), 1])/sf
        ISUM2 = ISUM2.reshape([len(ISUM2), 1])/sf
        INEW1 = INEW1.reshape([len(INEW1), 1])/sf
        INEW2 = INEW2.reshape([len(INEW2), 1])/sf

        np.savetxt(save_results_to + 'S_ode.txt', S1.reshape((-1, 1)))
        np.savetxt(save_results_to + 'I1_ode.txt', I11.reshape((-1, 1)))
        np.savetxt(save_results_to + 'I2_ode.txt', I21.reshape((-1, 1)))
        np.savetxt(save_results_to + 'R_ode.txt', R1.reshape((-1, 1)))
        np.savetxt(save_results_to + 'I1_new_ode.txt', INEW1.reshape((-1, 1)))
        np.savetxt(save_results_to + 'I2_new_ode.txt', INEW2.reshape((-1, 1)))
        np.savetxt(save_results_to + 'I1_sum_ode.txt', ISUM1.reshape((-1, 1)))
        np.savetxt(save_results_to + 'I2_sum_ode.txt', ISUM2.reshape((-1, 1)))

        # New cases
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, I1_new_star/sf, 'k--', marker='o', lw=2, markersize=5, label='Data-Weekly')
        ax.plot(t_star[1:], INEW1, 'r-', lw=2, label='Odeslover')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('New cases ($I_{1}^{new}$)', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'new1_cases_ode.png', dpi=300)


        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, I2_new_star / sf, 'k--', marker='o', lw=2, markersize=5, label='Data-Weekly')
        ax.plot(t_star[1:], INEW2, 'r-', lw=2, label='Odeslover')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('New cases ($I_{2}^{new}$)', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'new2_cases_ode.png', dpi=300)


        # Cumulative cases
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, I1_sum_star/ sf, 'k--', marker='o', lw=2, markersize=5, label='Data-Weekly')
        ax.plot(t_star, ISUM1, 'r-', lw=2, label='Odeslover')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('Cumulative cases ($I_{1}^{c}$)', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'Cumulative_cases1_ode.png', dpi=300)

        # Cumulative cases
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, I2_sum_star / sf, 'k--', marker='o', lw=2, markersize=5, label='Data-Weekly')
        ax.plot(t_star, ISUM2, 'r-', lw=2, label='Odeslover')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('Cumulative cases ($I_{2}^{c}$)', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'Cumulative_cases2_ode.png', dpi=300)

        # Current SI1I2R
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, S1, lw=2, label='Odeslover')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('$S$', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'Current_S_ode.png', dpi=300)

        # Infections 1
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, I11, 'r-', lw=2, label='Odeslover')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('Current infections ($I_{1}$)', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'Current_infections1_ode.png', dpi=300)

        # Infections 2
        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, I21, 'r-', lw=2, label='Odeslover')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('Current infections ($I_{2}$)', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'Current_infections2_ode.png', dpi=300)

        font = 24
        fig, ax = plt.subplots()
        ax.plot(t_star, R1, lw=2, label='Odeslover')
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', labelsize=24)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        ax.grid(True)
        ax.set_xlabel('Weeks', fontsize=font)
        ax.set_ylabel('$R$', fontsize=font)
        fig.set_size_inches(w=13, h=6.5)
        if SAVE_FIG:
            plt.savefig(save_results_to + 'Current_R_ode.png', dpi=300)



