#!/usr/bin/env python

import sys 
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import glob
import numpy as np
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Lambda, Reshape
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.losses import mse, binary_crossentropy
import os 

import generate_data as G 

class vae:

    def __init__(self,w=480,h=640,batch_size=60,kernel_size=3,filters=16,latent_dim=1000):
        
        self.w = w
        self.h = h
        input_shape = (self.w, self.h, 3)
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.filters = filters
        self.latent_dim = latent_dim

        inputs = Input(shape=input_shape, name='encoder_input')
        x = inputs
        for i in range(4):
            filters *= 2
            x = Conv2D(filters=self.filters,
               kernel_size=self.kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

        shape = K.int_shape(x)

        x = Flatten()(x)
        x = Dense(5000, activation='relu')(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        #z_joints = Dense(6, name='z_joints')(x)

        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var]) # concat w joint  keras.layers.Concatenate()([z, z_joint])

        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder') # add joints to output list
        self.encoder.summary()

        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling') # + 6 to shape
        x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        for i in range(4):
            x = Conv2DTranspose(filters=self.filters,
                        kernel_size=self.kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
        filters //= 2

        outputs = Conv2DTranspose(filters=3,
                          kernel_size=self.kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

        self.decoder = Model(latent_inputs, outputs, name='decoder')
        self.decoder.summary()

        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae')

        reconstruction_loss = binary_crossentropy(K.flatten(inputs),K.flatten(outputs))
        reconstruction_loss *= w*h
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)

        self.vae.compile(optimizer='rmsprop')
        self.vae.summary()


    def train(self,fol_path='../data/*'):
        
        f_list = glob.glob(fol_path+'*.jpg')
        im_list = []
        #print(f_list)
        for f in sorted(f_list):
	        #Crop to ultrasound active area
            im = cv2.imread(f)
            im_list.append(im)

        X_train = np.array(im_list[:-1]).reshape(-1,self.w,self.h,3)/255.0
        X_test = np.array(im_list[-1]).reshape(-1,self.w,self.h,3)/255.0

        print(np.shape(X_train))
        self.vae.fit(X_train,epochs=500,batch_size=self.batch_size,shuffle=True,validation_data=(X_test,None),callbacks=[TensorBoard(log_dir='./logs/3/')])

        self.vae.save_weights('./logs/3/1000_vae.h5')
        self.encoder.save_weights('./logs/3/1000_encoder.h5')
        self.decoder.save_weights('./logs/3/1000_decoder.h5')
        print('Saved trained weights.')

    def sampling(self,args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def load(self,path='./logs/3/'):
        self.encoder.load_weights(path+'encoder.h5')
        self.decoder.load_weights(path+'decoder.h5')
        self.vae.load_weights(path+'vae.h5')
        print('Loaded saved weights.')

    def predict(self, img_path):
        #im = cv2.imread(img_path).reshape(-1,self.w,self.h,3)/255.0
        im = img_path/255.0
        out = self.vae(K.constant(im))
        return K.eval(out)

class vae_2:

    def __init__(self,w=480,h=640,batch_size=60,kernel_size=3,filters=16,latent_dim=1006):
        
        self.w = w
        self.h = h
        input_shape = (self.w, self.h, 3)
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.filters = filters
        self.latent_dim = latent_dim

        inputs = Input(shape=input_shape, name='encoder_input')
        joint_input = Input(shape=6, name='joint_input') 
        x = inputs
        for i in range(4):
            filters *= 2
            x = Conv2D(filters=self.filters,
               kernel_size=self.kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

        shape = K.int_shape(x)

        x = Flatten()(x)
        x = Dense(5000, activation='relu')(x)
        z_mean = Dense(self.latent_dim - 6, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim - 6, name='z_log_var')(x)
        z_joints = Dense(6, name='z_joints')(x)

        z = Lambda(self.sampling, output_shape=(self.latent_dim - 6,), name='z')([z_mean, z_log_var]) # concat w joint  
        
        z = keras.layers.Concatenate()([z, z_joints])

        self.encoder = Model(inputs, [z_mean, z_log_var, z, z_joints], name='encoder') # add joints to output list
        self.encoder.summary()

        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling') 
        x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        for i in range(4):
            x = Conv2DTranspose(filters=self.filters,
                        kernel_size=self.kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
        filters //= 2

        outputs = Conv2DTranspose(filters=3,
                          kernel_size=self.kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

        self.decoder = Model(latent_inputs, outputs, name='decoder')
        self.decoder.summary()

        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model((inputs, joint_input), outputs, name='vae')

        reconstruction_loss = binary_crossentropy(K.flatten(inputs),K.flatten(outputs))
        reconstruction_loss *= w*h
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        joint_loss = mse(z_joints, joint_input)
        vae_loss = K.mean(reconstruction_loss + kl_loss + joint_loss) 
 
        self.vae.add_loss(vae_loss)

        self.vae.compile(optimizer='rmsprop')
        self.vae.summary()


    def train(self,fol_path='../data/*'):
        # TODO add input joint data
        f_list = glob.glob(fol_path+'*.jpg')
        im_list = []
        #print(f_list)
        for f in sorted(f_list):
	        #Crop to ultrasound active area
            im = cv2.imread(f)
            im_list.append(im)

        X_train = np.array(im_list[:-1]).reshape(-1,self.w,self.h,3)/255.0
        X_test = np.array(im_list[-1]).reshape(-1,self.w,self.h,3)/255.0

        print(np.shape(X_train))
        self.vae.fit(X_train,epochs=1000,batch_size=self.batch_size,shuffle=True,validation_data=(X_test,None),callbacks=[TensorBoard(log_dir='./logs/vae2/')])

        self.vae.save_weights('./logs/vae2/vae.h5')
        self.encoder.save_weights('./logs/vae2/encoder.h5')
        self.decoder.save_weights('./logs/vae2/decoder.h5')
        print('Saved trained weights.')

    def sampling(self,args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def load(self,path='./logs/vae2/'):
        self.encoder.load_weights(path+'encoder.h5')
        self.decoder.load_weights(path+'decoder.h5')
        self.vae.load_weights(path+'vae.h5')
        print('Loaded saved weights.')

    def predict(self, img_path):
        im = cv2.imread(img_path).reshape(-1,self.w,self.h,3)/255.0
        out = self.vae(K.constant(im))
        return K.eval(out)

class vae_occ:

    def __init__(self,w=480,h=640,batch_size=60,kernel_size=3,filters=16,latent_dim=1000):
        
        self.w = w
        self.h = h
        input_shape = (self.w, self.h, 3)
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.filters = filters
        self.latent_dim = latent_dim

        inputs = Input(shape=input_shape, name='encoder_input')
        x = inputs
        for i in range(4):
            filters *= 2
            x = Conv2D(filters=self.filters,
               kernel_size=self.kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

        shape = K.int_shape(x)

        x = Flatten()(x)
        x = Dense(5000, activation='relu')(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)
        #z_joints = Dense(6, name='z_joints')(x)

        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var]) # concat w joint  keras.layers.Concatenate()([z, z_joint])

        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder') # add joints to output list
        self.encoder.summary()

        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling') # + 6 to shape
        x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        for i in range(4):
            x = Conv2DTranspose(filters=self.filters,
                        kernel_size=self.kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
        filters //= 2

        outputs = Conv2DTranspose(filters=3,
                          kernel_size=self.kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

        self.decoder = Model(latent_inputs, outputs, name='decoder')
        self.decoder.summary()

        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae')

        reconstruction_loss = binary_crossentropy(K.flatten(inputs),K.flatten(outputs))
        reconstruction_loss *= w*h
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)

        self.vae.compile(optimizer='rmsprop')
        self.vae.summary()


    def train(self,fol_path='./data/*'):
        
        f_list = glob.glob(fol_path+'*.jpg')
        im_list = []
        occ_im_list = []
        #print(f_list)
        for f in sorted(f_list):
	        #Crop to ultrasound active area
            im = cv2.imread(f)
            occ_im = G.generate_occluded_image(im)
            im_list.append(im)
            occ_im_list.append(occ_im)
            cv2.imwrite("./data/occ_train/occ_{}".format(f), occ_im)

        X_train = np.array(occ_im_list[:-1]).reshape(-1,self.w,self.h,3)/255.0
        X_test = np.array(im_list[-1]).reshape(-1,self.w,self.h,3)/255.0

        print(np.shape(X_train))
        self.vae.fit(X_train,epochs=100,batch_size=self.batch_size,shuffle=True,validation_data=(X_test,None),callbacks=[TensorBoard(log_dir='./logs/3/')])

        if not os.path.isdir("./logs/occ_vae/".format(target_index)):
            os.mkdir("./logs/occ_vae/".format(target_index))
        self.vae.save_weights('./logs/occ_vae/vae.h5')
        self.encoder.save_weights('./logs/occ_vae/encoder.h5')
        self.decoder.save_weights('./logs/occ_vae/decoder.h5')
        print('Saved trained weights.')

    def sampling(self,args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def load(self,path='./logs/occ_vae/'):
        self.encoder.load_weights(path+'encoder.h5')
        self.decoder.load_weights(path+'decoder.h5')
        self.vae.load_weights(path+'vae.h5')
        print('Loaded saved weights.')

    def predict(self, img_path):
        im = cv2.imread(img_path).reshape(-1,self.w,self.h,3)/255.0
        out = self.vae(K.constant(im))
        return K.eval(out)


if __name__ == '__main__':
    vae_model = vae_occ()
    #vae_model.load()
    vae_model.train()