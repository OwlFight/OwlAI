#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
# Author: Emilien Taisne											#
# Date_start: 16/01/2019											#
# Date_finish: __/__/____											#
# Use: An artifitial intelligence build from keras(tensorflow-gpu)  #
#	   type GAN, that reconises and generate images of owl for my   #
#	   profile picture.												#
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#							  Imports								#
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#


from keras import backend as K

from keras.layers import Reshape, Dense, Dropout, Flatten, Conv2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Input
from keras import initializers

import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from tqdm import tqdm
import numpy as np


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
# 			  Importing, cleaning and preparing the data 			#
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#


image_list = glob("train_image/*")

arr_image = []
for image_name in tqdm(image_list):

	image = Image.open(image_name)
	image = image.convert('RGB')
	image = image.resize((100,100))

	image_arr = np.asarray(image)
	image_arr = image_arr.astype('float32')
	image_arr = image_arr / 255
	image_arr = image_arr.reshape(100, 100, 3)

	arr_image.append(image_arr)

arr_image = np.array(arr_image)	#numpy arrays are cool

print(arr_image.shape)	#showing the dimention of the array

#Setting the optimizer

adam = Adam(lr=0.0002, beta_1=0.5)

#idk, just random stuff for the gen
randomDim = 100

#
epoch_num = 1

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
# 					     Building the models 						#
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#


# = = = =The Generator= = = = #

generator = Sequential()

generator.add(Dense(64*100*100, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))

generator.add(LeakyReLU(0.2))

generator.add(Reshape((100, 100, 64)))

generator.add(UpSampling2D(size=(2, 2)))

generator.add(Conv2D(32, kernel_size=(3, 3), padding='same'))

generator.add(LeakyReLU(0.2))

generator.add(Conv2D(16, kernel_size=(5, 5), padding='same'))

generator.add(LeakyReLU(0.2))

generator.add(UpSampling2D(size=(2, 2)))

generator.add(Conv2D(3, kernel_size=(5, 5), padding='same', activation='tanh'))

generator.compile(loss='binary_crossentropy', optimizer=adam)

# generator = load_model('models/dcgan_GPU_generator_epoch_%d.h5' % epoch_num)

#= = = = =The Discriminator= = = = =#

discriminator = Sequential()

discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(400, 400, 3), kernel_initializer=initializers.RandomNormal(stddev=0.02)))

discriminator.add(LeakyReLU(0.2))

discriminator.add(Dropout(0.3))

discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))

discriminator.add(LeakyReLU(0.2))

discriminator.add(Dropout(0.3))

discriminator.add(Flatten())

discriminator.add(Dense(1, activation='sigmoid'))

discriminator.compile(loss='binary_crossentropy', optimizer=adam)

# discriminator = load_model('models/dcgan_GPU_discriminator_epoch_%d.h5' % epoch_num)


#= = = = =Combining the models in one= = = = =#

discriminator.trainable = False

ganInput = Input(shape=(randomDim,))

x = generator(ganInput)

ganOutput = discriminator(x)

gan = Model(inputs=ganInput, outputs=ganOutput)

gan.compile(loss='binary_crossentropy', optimizer=adam)

gan.summary()


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
# 	Training + cleaning results + showing results   #
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#


def plotGeneratedImages(epoch, examples=25, dim=(5, 5), figsize=(10, 10)):

    noise = np.random.normal(0, 1, size=[examples, randomDim])

    generatedImages = generator.predict(noise)

    print(generatedImages.shape)

    plt.figure(figsize=figsize)

    for i in range(generatedImages.shape[0]):

        plt.subplot(dim[0], dim[1], i+1)

        plt.imshow(generatedImages[i], interpolation='nearest')

        plt.axis('off')

    plt.tight_layout()

    plt.savefig('generated_image/dcgan_generated_image_epoch_%d.png' % epoch)




def saveModels(epoch):

    generator.save('models/dcgan_GPU_generator_epoch_%d.h5' % epoch)

    discriminator.save('models/dcgan_GPU_discriminator_epoch_%d.h5' % epoch)




def train(epochs=1, batchSize=128):

    # batchCount = int(arr_image.shape[0] / batchSize)

    # if arr_image.shape[0] < batchSize:
    batchCount = arr_image.shape[0]

    # batchCount = 5

    print('Epochs:', epochs)

    print('Batch size:', batchSize)

    print('Batches per epoch:', batchCount)



    for e in range(1,epochs+1):
           
        print('-'*15, 'Epoch %d' % e, '-'*15)

        for _ in tqdm(range(batchCount)):

            # Get a random set of input noise and images

            noise = np.random.normal(0, 1, size=[batchSize, randomDim])

            imageBatch = arr_image[np.random.randint(0, arr_image.shape[0], size=batchSize)]



            # Generate fake MNIST images

            generatedImages = generator.predict(noise)

            X = np.concatenate([imageBatch, generatedImages])



            # Labels for generated and real data

            yDis = np.zeros(2*batchSize)

            # One-sided label smoothing

            yDis[:batchSize] = 0.95



            # Train discriminator

            discriminator.trainable = True

            dloss = discriminator.train_on_batch(X, yDis)



            # Train generator

            noise = np.random.normal(0, 1, size=[batchSize, randomDim])

            yGen = np.ones(batchSize)

            discriminator.trainable = False

            gloss = gan.train_on_batch(noise, yGen)

        plotGeneratedImages(e)
        
        if e % 50 == 0:
            saveModels(e)






if __name__ == '__main__':

    train(5000, 128)