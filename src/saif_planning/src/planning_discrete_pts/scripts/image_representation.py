#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
#from keras.applications.inception_v3 import InceptionV3


#import tensorflow as tf
import torch
import torch.nn as nn 
import torch.optim as optim 
import torchvision
#import torchvision.transforms.CenterCrop as CenterCrop

import os
import datetime
from os import listdir
from os.path import isfile, join
import time
import numpy as np
import glob
import random
import matplotlib.pyplot as plt
import PIL
import cv2
#import imageio

#from IPython import display


# def toFeatureRepresentation(img, img_shape):

    

#     model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=img_shape, pooling='avg', classes=1000)


#     return model.predict(img)

# keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)


# class VAE(nn.Module):
#     def __init__(self, in_shape, n_latent):
#         super().__init__()
#         self.in_shape = in_shape
#         self.n_latent = n_latent
#         c,h,w = in_shape
#         self.z_dim = h//2**2 # receptive field downsampled 2 times
#         self.encoder = nn.Sequential(
#             nn.BatchNorm2d(c),
#             nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),  # 32, 16, 16
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32, 8, 8
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(),
#         )
#         self.z_mean = nn.Linear(64 * self.z_dim**2, n_latent)
#         self.z_var = nn.Linear(64 * self.z_dim**2, n_latent)
#         self.z_develop = nn.Linear(n_latent, 64 * self.z_dim**2)
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1),
#             CenterCrop(h,w),
#             nn.Sigmoid()
#         )

#     def sample_z(self, mean, logvar):
#         stddev = torch.exp(0.5 * logvar)
#         noise = Variable(torch.randn(stddev.size()))
#         return (noise * stddev) + mean

#     def encode(self, x):
#         x = self.encoder(x)
#         x = x.view(x.size(0), -1)
#         mean = self.z_mean(x)
#         var = self.z_var(x)
#         return mean, var

#     def decode(self, z):
#         out = self.z_develop(z)
#         out = out.view(z.size(0), 64, self.z_dim, self.z_dim)
#         out = self.decoder(out)
#         return out

#     def forward(self, x):
#         mean, logvar = self.encode(x)
#         z = self.sample_z(mean, logvar)
#         out = self.decode(z)
#         return out, mean, logvar

# def train(model, loader, loss_func, optimizer):
#     model.train()
#     for inputs, _ in loader:
#         inputs = Variable(inputs)

#         output, mean, logvar = model(inputs)
#         loss = vae_loss(output, inputs, mean, logvar, loss_func)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
# def vae_loss(output, input, mean, logvar, loss_func):
#     recon_loss = loss_func(output, input)
#     kl_loss = torch.mean(0.5 * torch.sum(
#         torch.exp(logvar) + mean**2 - 1. - logvar, 1))
#     return recon_loss + kl_loss


class Embedder(tf.keras.Model):
    def __init__(self, embedding_size, c=0.2):
        super(Embedder, self).__init__()
        self.embedding_size = embedding_size
        self.w = w
        self.h = h
        self.c = c
        input_shape = (3, self.w, self.h, 3)

        all_inputs = Input(shape=input_shape, name='input')

        single_input = Input(shape=(self.w, self.h, 3), name='single_input')

        base_model = InceptionV3(include_top=False, weights='imagenet')
        inception_layers = base_model.get_layer("Mixed_5d").output

        x1 = inception_layers(single_input)

        for i in range(2):
            filters *= 2
            x1 = Conv2D(filters=self.filters,
               kernel_size=self.kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x1)

        x1 = tf.contrib.layers.spatial_softmax(
            x1,
            temperature=None,
            name=None,
            variables_collections=None,
            trainable=True,
            data_format='NHWC'
        )

        x1 = Dense(5000, activation='relu')(x1)
        output = Dense(self.latent_dim, name='output')


        self.embedder = Model(single_input, output)

        output_0 = self.embedder(all_inputs[0, :, :, :])
        output_1 = self.embedder(all_inputs[1, :, :, :])
        output_2 = self.embedder(all_inputs[2, :, :, :])

        tcn_loss = np.matmul(output_1, output_2) \
            - np.matmul(output_1, output_2) + self.c
        loss = K.mean(tcn_loss)
        self.model = Model(inputs, [output_0, output_1, output_2], name='TCN_model')
        self.model.add_loss(loss)

        self.model.compile(optimizer='adam')
        self.vae.summary()

    def train(self,fol_path='../data/*'):
        
        data_list = self.prepare_data()
        X_train = np.array(data_list[:-1]).reshape(-1, 3, self.w,self.h,3)/255.0
        X_test = np.array(data_list[-1]).reshape(-1, 3, self.w,self.h,3)/255.0

        print(np.shape(X_train))
        self.model.fit(X_train,epochs=500,batch_size=self.batch_size,shuffle=True,validation_data=(X_test,None),callbacks=[TensorBoard(log_dir='./logs/tcn/')])

        self.vae.save_weights('./logs/tcn/embedder.h5')
        self.encoder.save_weights('./logs/tcn/encoder.h5')
        self.decoder.save_weights('./logs/tcn/decoder.h5')
        print('Saved trained weights.')


    def prepare_data(self, fol_path='../data/target*'):

        fol_list = glob.glob(fol_path)
        data_list = []
        
        #print(f_list)
        for fol in fol_list:
            seq_list = glob.glob(fol_path + "/*")
            im_list = []
            for seq in seq_list:
                rewards = np.load(seq)
                fs = sorted(glob.glob(fol_path + "/*/*.jpg"))
                for i in range(len(fs)):
                    im = cv2.imread(fs[i])
                    im_list.append((im, rewards[i], seq))
                     
            data_list.append(self.generate_triplets(im_list))

        return data_list

    def generate_triplets(self, data, attract_thresh=0.01, repel_thresh=0.05, samples_per_target=10):
        
        triplet_list = []
        for (image, reward, seq_index) in data:
            attract_list = [(im, rw, seq) for (im, rw, seq) in data if abs(reward - rw) < attract_thresh and not seq_index = seq]
            repel_list = [(im, rw, s) for (im, rw, s) in data if abs(reward - rw) > repel_thresh]

            for i in range(samples_per_target):
                a_index = random.randint(0, len(attract_list) - 1)
                a_img = attract_list[a_index][0]
                attract_list.pop(a_index)
                r_index = random.randint(0, len(repel_list) - 1)
                r_img = repel_list[r_index][0]
                repel_list.pop(r_index)
                triplet_list.append([image, a_img, r_img])

        return np.array(triplet_list)


class CVAE(tf.keras.Model):
  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.inference_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(480, 640, 3)), #(480,640,3)
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.generative_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=120*160*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(120, 160, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=3, kernel_size=3, strides=(1, 1), padding="SAME"),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.generative_net(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs

    return logits



def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

@tf.function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# def generate_and_save_images(model, epoch, test_input):
#     sess = tf.Session()
#     predictions = model.sample(test_input)  #.astype('float32')
    
#     tf.image.convert_image_dtype(
#         predictions,
#         'float',
#         saturate=False,
#         name=None
#     )

#     print("preds: " + str(predictions[0, :, :, 0]))
#     fig = plt.figure(figsize=(4,4))

#     for i in range(predictions.shape[0]):
#         plt.subplot(4, 4, i+1)
#         print(np.array(predictions[i, :, :, 0]))
#         with sess.as_default():
#             print("type: " + str(type(predictions[i, :, :, 0].eval())))
#         #plt.imshow(predictions[i, :, :, 0], cmap='gray')
        
#         plt.imshow(np.array(predictions[i, :, :, 0]))
#         plt.axis('off')

#     # tight_layout minimizes the overlap between 2 sub-plots
#     plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
#     plt.show()

def load_data(dir_path):
  #uploaded = files.upload()
    #print([str(f) for f in listdir(dir_path)])
    img_files = [join(str(dir_path), f) for f in listdir(dir_path) if isfile(join(dir_path, f))]
    #print(img_files)
    random.shuffle(img_files)
    size_test = int(len(img_files)/4)
    test_files = img_files[0:size_test] 
    train_files = img_files[size_test:]
    test_set = []
    train_set = []

    for img in test_files:
        im = cv2.imread(img)
        test_set.append(im)

    for img in train_files:
        im = cv2.imread(img)
        train_set.append(im)

    return np.array(train_set), np.array(test_set)


if __name__ == "__main__":
    
    #model = VAE((28, 28, 1), 1024)
    
    train_images, test_images = load_data("all_imgs/")
    
    print(np.shape(train_images))

    #(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    # x = torch.randn(1, 28, 28, 1)

    # print(model(x))

    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    # train(model, loader, nn.MSELoss, optimizer)
#     (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 480, 640, 3).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 480, 640, 3).astype('float32')

    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.

    # Binarization
#     train_images[train_images >= .5] = 1.
#     train_images[train_images < .5] = 0.
#     test_images[test_images >= .5] = 1.
#     test_images[test_images < .5] = 0.
    
    train_dataset = train_images.astype('float32')
    test_dataset = test_images.astype('float32')

    TRAIN_BUF = 600
    BATCH_SIZE = 10

    TEST_BUF = 100

    optimizer = tf.keras.optimizers.Adam(1e-4)

    epochs = 50
    latent_dim = 1000
    num_examples_to_generate = 16

    # keeping the random vector constant for generation (prediction) so
    # it will be easier to see the improvement.
    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])
    model = CVAE(latent_dim)



    #generate_and_save_images(model, 0, random_vector_for_generation)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        #for train_x in train_images:
        compute_apply_gradients(model, train_images, optimizer)
        end_time = time.time()

        if epoch % 1 == 0:
            loss = tf.keras.metrics.Mean()
            #for test_x in test_dataset:
            loss(compute_loss(model, test_dataset))
            elbo = -loss.result()
            #display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, '
                'time elapse for current epoch {}'.format(epoch,
                                                            elbo,
                                                            end_time - start_time))
#             generate_and_save_images(
#                 model, epoch, random_vector_for_generation)

    model.save("test_cvae.h5")
    clear_session()

