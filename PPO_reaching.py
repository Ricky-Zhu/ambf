import tensorflow as tf
import os
import numpy as np
from collections import deque, namedtuple
import itertools
import sys
sys.path.append("library/")
import random
import pandas as pd
import matplotlib.pyplot as pl
from PPO_env import GameState
import matplotlib.pyplot as plt

EP_MAX = 1000
EP_LEN = 100
GAMMA = 0.9
A_LR = 0.001
C_LR = 0.002
BATCH = 32
A_UPDATE_STEPS = 5
C_UPDATE_STEPS = 5
A_DIM = 2
epsilon=0.2

class StateProcessor():
    def __init__(self):
        self.pretrain = True
        with tf.variable_scope("state_processor"):
            self.input_img1 = tf.placeholder(shape=[540, 960, 3], dtype=tf.uint8)
            self.input_img1a = tf.placeholder(shape=[540, 960, 3], dtype=tf.uint8)
            self.image_decoded1 = tf.image.crop_to_bounding_box(self.input_img1, 150, 250, 390, 460)
            self.output_img1 = tf.image.resize_images(self.image_decoded1, [150, 150]) / 255
            self.output_img1 = tf.image.rgb_to_grayscale(self.output_img1)
            self.output_img1a = tf.image.resize_images(self.input_img1a, [150, 150]) / 255
            self.output_img1a = tf.image.rgb_to_grayscale(self.output_img1a)
            # self.output_img1 = tf.squeeze(self.output_img1)
            # self.output_img1a = tf.squeeze(self.output_img1a)

    def process(self, sess, img1, img1a):
        return sess.run([self.output_img1, self.output_img1a], {self.input_img1: img1, self.input_img1a: img1a})

class PPO(object):

    def __init__(self):
        self.tfs1 = tf.placeholder(tf.float32, [None, 150,150,1], 'state1')
        self.tfs1a = tf.placeholder(tf.float32, [None,150,150,1], 'state1a')

        def encoder(X, name):
            l1 = tf.layers.conv2d(X, 32, 2, 1, padding='same')
            pool1 = tf.layers.max_pooling2d(l1, 2, 1, padding='same')
            l2 = tf.layers.conv2d(pool1, 64, 2, 1, padding='same')
            p2 = tf.layers.max_pooling2d(l2, 2, 1, padding='same')
            flatten = tf.layers.flatten(p2)
            encoded = tf.layers.dense(flatten, 64, tf.nn.relu)
            return encoded
        def concat(a, b, name):
            conct = tf.concat([a, b], 1, name=name)
            tf.summary.histogram(name + '/outputs', conct)
            return conct

        # critic
        with tf.variable_scope('critic'):
            encoder1=encoder(self.tfs1,'encoder1')
            encoder1a=encoder(self.tfs1a,'encoder1a')
            encoder_all = concat(encoder1,encoder1a,'encoder_all')
            self.v = tf.layers.dense(encoder_all, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
                surr = ratio * self.tfadv

            self.aloss = -tf.reduce_mean(tf.minimum(
                surr,
                tf.clip_by_value(ratio, 1.-epsilon, 1.+epsilon)*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

    def load_model(self,sess,saver,latest_checkpoint):
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
    def save_model(self,sess,saver,model_path,total_step):
        saver.save(sess, model_path, global_step=total_step, write_meta_graph=False)
        if not os.path.exists('/home/dandan/ruiqi_ppo_model/checkpoint_dir/Mymodel.meta'):
            saver.export_meta_graph('/home/dandan/ruiqi_ppo_model/checkpoint_dir/Mymodel.meta')



    def update(self,sess, s_1,s_1a, a, r):
        sess.run(self.update_oldpi_op)
        adv = sess.run(self.advantage, {self.tfs1: s_1,self.tfs1a:s_1a, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful
        [sess.run(self.atrain_op, {self.tfs1: s_1,self.tfs1a:s_1a, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]
        # update critic
        [sess.run(self.ctrain_op, {self.tfs1: s_1,self.tfs1a:s_1a, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.conv2d(self.tfs1, 32, 2, 1, padding='same', trainable=trainable)
            pool1 = tf.layers.max_pooling2d(l1, 2, 1, padding='same')
            flatten1 = tf.layers.flatten(pool1)
            encoded1 = tf.layers.dense(flatten1, 64, tf.nn.relu,trainable=trainable)

            l1a = tf.layers.conv2d(self.tfs1a, 32, 2, 1, padding='same', trainable=trainable)
            pool1a = tf.layers.max_pooling2d(l1a, 2, 1, padding='same')
            flatten1a = tf.layers.flatten(pool1a)
            encoded1a = tf.layers.dense(flatten1a, 64, tf.nn.relu,trainable=trainable)

            encoder_1a_all = tf.concat([encoded1,encoded1a], 1)

            conct = tf.layers.dense(encoder_1a_all, 100, tf.nn.relu, trainable=trainable)
            mu = 0.1 * tf.layers.dense(conct, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(conct, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, sess,s1,s1a):
        s1 = s1[np.newaxis, :]
        s1a = s1a[np.newaxis, :]
        a = sess.run(self.sample_op, {self.tfs1: s1, self.tfs1a: s1a})[0]
        return np.clip(a, -0.08, 0.08)

    def get_v(self,sess, s1,s1a):
        if s1.ndim < 4: s1 = s1[np.newaxis, :]
        if s1a.ndim < 4: s1a = s1a[np.newaxis, :]
        return np.squeeze(sess.run(self.v, {self.tfs1: s1, self.tfs1a: s1a}))

env = GameState()
ppo = PPO()
all_ep_r = []
rough_reward=[]
tt = 1
sp = StateProcessor()
saver=tf.train.Saver()
global_step = tf.Variable(0, name='global_step', trainable=False)
path = '/home/dandan/ruiqi_ppo_model/checkpoint_dir/Mymodel'
latest_checkpoint = tf.train.latest_checkpoint('/home/dandan/ruiqi_ppo_model/checkpoint_dir')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ppo.load_model(sess,saver,latest_checkpoint)
    total_t = sess.run(tf.train.get_global_step())
    env.first_reset()

    for ep in range(EP_MAX):
        s1,s1a = env.reset((tt%3))
        s1, s1a = sp.process(sess,s1,s1a)
        tt += 1
        buffer_s1, buffer_s1a, buffer_a, buffer_r = [], [], [],[]
        ep_r = 0
        for t in range(EP_LEN):    # in one episode
            a = ppo.choose_action(sess,s1,s1a)
            s1_,s1a_, r, done = env.step(a)
            s1_, s1a_ = sp.process(sess,s1_,s1a_)
            buffer_s1.append(s1)
            buffer_s1a.append(s1a)
            buffer_a.append(a)
            buffer_r.append(r)    # normalize reward, find to be useful
            s1 = s1_
            s1a = s1a_
            ep_r += r
            if (total_t+1) % (500) == 0:
                ppo.save_model(sess,saver,path,total_t)
                df1 = pd.DataFrame({'reward':np.array(all_ep_r)})
                df1.to_csv('/home/dandan/ruiqi_ppo_model/record/reward2.csv')
                df2 = pd.DataFrame({'rough_reward': np.array(rough_reward)})
                df2.to_csv('/home/dandan/ruiqi_ppo_model/record/rough_reward2.csv')
            total_t+=1

            # update ppo
            if (t+1) % BATCH == 0 or t == EP_LEN-1:
		print('updating')
                v_s_ = ppo.get_v(sess,s1_,s1a_)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()
#                bs1, bs1a, ba, br = np.vstack(buffer_s1),np.vstack(buffer_s1a), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                bs1, bs1a, ba, br = np.reshape(buffer_s1,[-1,150,150,1]),np.reshape(buffer_s1a,[-1,150,150,1]),np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s1,buffer_s1a, buffer_a, buffer_r = [], [], [],[]
                ppo.update(sess,bs1,bs1a, ba, br)
	    if done:
		s1,s1a = env.gri_reset()
		s1, s1a = sp.process(sess,s1,s1a)

        print('episode:{}/{}, reward:{},totol_step:{}'.format(ep+1,EP_MAX,ep_r,total_t))

        if ep == 0: all_ep_r.append(ep_r)
        else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
        rough_reward.append(ep_r)
        # print(
        #     'Ep: %i' % ep,
        #     "|Ep_r: %i" % ep_r
        # )
# 
# plt.plot(np.arange(len(all_ep_r)), all_ep_r)
# plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()



