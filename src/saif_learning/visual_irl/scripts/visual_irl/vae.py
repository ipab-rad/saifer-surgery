#!/usr/bin/env python

import cv2
import glob
import numpy as np
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Lambda, Reshape
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.losses import mse, binary_crossentropy

class vae:

	def __init__(self,w=112,h=112,batch_size=128,kernel_size=3,filters=8,latent_dim=16):
		
		self.w = w
		self.h = h
		input_shape = (self.w, self.h, 1)
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
		x = Dense(32, activation='relu')(x)
		z_mean = Dense(self.latent_dim, name='z_mean')(x)
		z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

		z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

		self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
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

		outputs = Conv2DTranspose(filters=1,
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
		fol_list = glob.glob(fol_path)
		print(fol_list)
		seq_list = []
		for fol in fol_list:
			f_list = glob.glob(fol+'/*.jpg')
			im_list = []
			for f in sorted(f_list):
			#Crop to ultrasound active area
				im = np.mean(cv2.resize(cv2.imread(f)[180:700,500:1020,:],(self.w,self.h)),axis=-1)
				im_list.append(im)
			seq_list.append(np.array(im_list))

		X_train = np.vstack(seq_list[:-1]).reshape(-1,self.w,self.h,1)/255.0
		X_test = np.vstack(seq_list[-1]).reshape(-1,self.w,self.h,1)/255.0
		self.vae.fit(X_train,epochs=100,batch_size=self.batch_size,shuffle=True,validation_data=(X_test,None),callbacks=[TensorBoard(log_dir='./logs/')])

		self.vae.save_weights('./logs/vae.h5')
		self.encoder.save_weights('./logs/encoder.h5')
		self.decoder.save_weights('./logs/decoder.h5')
		print('Saved trained weights.')

	def sampling(self,args):
		z_mean, z_log_var = args
		batch = K.shape(z_mean)[0]
		dim = K.int_shape(z_mean)[1]
		epsilon = K.random_normal(shape=(batch, dim))
		return z_mean + K.exp(0.5 * z_log_var) * epsilon

	def load(self,path='./logs/'):
		self.encoder.load_weights(path+'encoder.h5')
		self.decoder.load_weights(path+'decoder.h5')
		self.vae.load_weights(path+'vae.h5')
		print('Loaded saved weights.')

if __name__ == '__main__':
	vae_model = vae()
	vae_model.train()
