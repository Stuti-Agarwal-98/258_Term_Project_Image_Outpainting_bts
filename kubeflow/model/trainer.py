from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Text

import absl
import tensorflow as tf
import keras
import keras.backend as K
from keras.utils import generic_utils
from PIL import Image
import PIL
from keras.models import Sequential, Model, load_model, model_from_json
from keras import layers
from keras.layers import Dense, Flatten, Dropout, Activation, LeakyReLU, Reshape, Concatenate, Input
from keras.layers import Conv2D, UpSampling2D, Conv2DTranspose, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers, losses
from copy import deepcopy
from tfx.components.trainer.fn_args_utils import FnArgs
from IPython.display import display, Image, Markdown, SVG
from keras.utils.vis_utils import model_to_dot
import itertools
import numpy as np
IMAGE_SZ = 128
IMAGE_KEY = 'image_raw'
LABEL_KEY = 'label'
# logdir = "/content/tfx/logs/train_data/"

def transformed_name(key):
    return key + '_xf'

def generator_conv_block(layer_input, filters, kernel_size, name,strides, padding='same', activation='relu', norm=True, dilation_rate=1):
    conv = Conv2D(filters, name=name,kernel_size=kernel_size, strides=strides, dilation_rate=(dilation_rate, dilation_rate), padding=padding)(layer_input)
    if activation=='relu':
        conv = Activation('relu')(conv)
    if norm:
        conv = BatchNormalization()(conv)
    return conv

def generator_Deconv_block(layer_input, filters, kernel_size, strides,name, padding='same', activation='relu'):
    deconv = Conv2DTranspose(filters, name=name,kernel_size = kernel_size, strides = strides, padding = 'same')(layer_input)
    if activation == 'relu':
        deconv = Activation('relu')(deconv)
    return deconv


def build_generator():

    generator_input = Input(name='image_raw', shape=(128, 128, 3))      ## These are masked or Padded Images of shape 128x128 but the 0 to 32 and 96 to 128 will be masked

    ##### Encoder #####
    g1 = generator_conv_block(generator_input, 64, 5, strides=1,name="Gen_Layer_1")
    g2 = generator_conv_block(g1, 128, 3, strides=2,name="Gen_Layer_2")
    g3 = generator_conv_block(g2, 256, 3, strides=1, name="Gen_Layer_3")
    # Dilated Convolutions
    g4 = generator_conv_block(g3, 256, 3, strides=1, dilation_rate=2,name="Gen_Layer_4")
    g5 = generator_conv_block(g4, 256, 3, strides=1, dilation_rate=4,name="Gen_Layer_5")
    g6 = generator_conv_block(g5, 256, 3, strides=1, dilation_rate=8,name="Gen_Layer_6")
    g7 = generator_conv_block(g6, 256, 3, strides=1,name="Gen_Layer_7")

    #### Decoder ####
    g8 = generator_Deconv_block(g7, 128, 4, strides=2,name="Gen_Layer_8")
    g9 = generator_conv_block(g8, 64, 3, strides=1,name="Gen_Layer_9")
    
    generator_output = Conv2D(3, kernel_size=3, name="Gen_Layer_10",strides=(1,1), activation='sigmoid', padding='same', dilation_rate=(1,1))(g9) ### Some people used 'tanh' instead of sigmoid check later
    
    return Model(generator_input, generator_output)

#discriminator
def create_discriminator_layer(layer_input, filters, kernel_size = 5, strides = 2, padding = 'same', activation='leakyrelu', dropout_rate=0.25, norm=True,name='layer'):
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name+'conv2d')(layer_input)
    if activation == 'leakyrelu':
        conv = LeakyReLU(alpha=0.2,name=name+'leakyRelu')(conv)
    if dropout_rate:
        conv = Dropout(rate=dropout_rate,name=name+'dropRate')(conv)
    if norm:
        conv = BatchNormalization(name=name+'batch')(conv)
    return conv

def build_discriminator():

    discriminator_input = Input(name='image_raw', shape=(128, 128, 3))      ## These are masked or Padded Images of shape 128x128 but the 0 to 32 and 96 to 128 will be masked

    model = create_discriminator_layer(discriminator_input, 32, 5, norm=False,name='layer1')
    model = create_discriminator_layer(model, 64, 5, 2,name='layer2')
    model = create_discriminator_layer(model, 64, 5, 2,name='layer3')
    model = create_discriminator_layer(model, 64, 5, 2,name='layer4')
    model = create_discriminator_layer(model, 64, 5, 2,name='layer5')

    model = Flatten()(model)
    model = Dense(512, activation='relu')(model)

    discriminator_output = Dense(1, activation='sigmoid')(model)

    return Model(discriminator_input, discriminator_output)
#gan starts here    
def build_gan(generator,discriminator):
    discriminator.trainable = False

    gan_input = Input(name='image_raw', shape=(128, 128, 3))
    print("GAN INPUT INFO")
    print(type(gan_input))
    print(gan_input.shape)
    print(type(generator))
    generated_image = generator(gan_input)
    print("Generator output shape:", generated_image.shape)
    gan_output = discriminator(generated_image)
    
    gan = Model(gan_input,[generated_image, gan_output])
    return gan


###############################
##Feature engineering functions
def feature_engg_features(features):
    #resize and decode
    image = tf.io.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [-1, 128, 128, 3])
    image = image/255
    features['image_raw'] = image

    return features

def label_engg_features(label):
  #resize and decode
  label = tf.io.decode_raw(label, tf.uint8)
  label = tf.reshape(label, [-1,128, 128, 3])
  label = label/255
  return label
 
#To be called from TF
def feature_engg(features, label):
  #Add new features
  features = feature_engg_features(features)
  label = label_engg_features(label)

  return(features, label)

def make_input_fn(data_root, mode, vnum_epochs = None, batch_size = 512):
    def decode_tfr(serialized_example):
      # 1. define a parser
      features = tf.io.parse_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        })

      return features, features['label']

    def _input_fn(v_test=False):
      # Get the list of files in this directory (all compressed TFRecord files)
      tfrecord_filenames = tf.io.gfile.glob(data_root)

      # Create a `TFRecordDataset` to read these files
      dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")

      if mode == tf.estimator.ModeKeys.TRAIN:
        num_epochs = vnum_epochs # indefinitely
      else:
        num_epochs = 1 

      dataset = dataset.batch(batch_size)
      dataset = dataset.prefetch(buffer_size = batch_size)

      #Convert TFRecord data to dict
      dataset = dataset.map(decode_tfr)

      #Feature engineering
      dataset = dataset.map(feature_engg)

      if mode == tf.estimator.ModeKeys.TRAIN:
          num_epochs = vnum_epochs # indefinitely
          dataset = dataset.shuffle(buffer_size = batch_size)
      else:
          num_epochs = 1 # end-of-input after this

      dataset = dataset.repeat(num_epochs)       
      
      if v_test == True:
        print(next(dataset.__iter__()))
        
      return dataset
    return _input_fn

def save_model(model, model_save_path):
  @tf.function
  def serving(image_raw):
      ##Feature engineering( calculate distance ) 

      payload = {
          'image_raw': image_raw
      }
      
      predictions = model(payload)
      return predictions

  serving = serving.get_concrete_function(image_raw=tf.TensorSpec([None, 128,128,3], dtype=tf.uint8, name='image_raw')
                                          )
  # version = "1"  #{'serving_default': call_output}
  tf.saved_model.save(
      model,
      model_save_path + "/",
      signatures=serving
  )


def show_images(dataset):
      for data in itertools.islice(dataset, None, 5):
        images, label = data
        masked_images = images['image_raw']
        print("masked image",masked_images.shape)
        img_norm = (np.asarray(masked_images[0])*255.0).astype(np.uint8)
        print("image_norm shape",img_norm.shape)
        display(PIL.Image.fromarray(img_norm,'RGB'))
        print("real image")
        
        img_norm = (np.asarray(label[0])*255.0).astype(np.uint8)
        display(PIL.Image.fromarray(img_norm,'RGB'))

# Main function called by TFX
def run_fn(fn_args: FnArgs):
      """
      Train the model based on given args.
      Args:
        fn_args: Holds args used to train the model as name/value pairs.
      """
      print("Starting Training!!!!!!!!!!!!!!")
      # Getting custom arguments
      batch_size = fn_args.custom_config['batch_size']
      gen_epoch = fn_args.custom_config['gen_epoch']
      dis_epoch = fn_args.custom_config['dis_epoch']
      gan_epoch = fn_args.custom_config['gan_epoch']
      data_size = fn_args.custom_config['data_size']

      steps_per_epoch = 1

      train_dataset = make_input_fn(data_root = fn_args.train_files,
                          mode = tf.estimator.ModeKeys.TRAIN,
                          batch_size = batch_size)()

      validation_dataset = make_input_fn(data_root = fn_args.eval_files,
                          mode = tf.estimator.ModeKeys.EVAL,
                          batch_size = batch_size)()    

      mirrored_strategy = tf.distribute.MirroredStrategy()
      with mirrored_strategy.scope():
        discriminator = build_discriminator()
        discriminator.trainable = False
        discriminator.compile(loss = losses.MSE, optimizer = optimizers.Adam(lr=0.0001, beta_1=0.5))

        generator = build_generator()
        generator.compile(loss = 'mse', optimizer = optimizers.Adam(lr=0.0001, beta_1=0.5))

        gan = build_gan(generator,discriminator)
        alpha=0.0004
        gan.compile(loss = [losses.MSE, losses.MSE], optimizer = optimizers.Adam(lr=0.0001, beta_1=0.5), loss_weights = [1, alpha])

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir, update_freq='batch')

      print("Generator Model Summary")
      print(generator.summary())
      print("Discriminator Model Summary")
      print(discriminator.summary())
      print("Gan Model Summary")
      print(gan.summary())

      show_images(train_dataset)
      # Training Generator
      print("Training Generator")
      generator.fit(
          train_dataset,
          epochs = gen_epoch,
          steps_per_epoch=steps_per_epoch,
          validation_data=validation_dataset,
          validation_steps=2,
          callbacks=[tensorboard_callback])

      # Training Discriminator
      print("Training Discriminator")
      for current_epoch in range(dis_epoch):
              print('Epoch {}/{}'.format(current_epoch, dis_epoch))
              progressbar = generic_utils.Progbar(steps_per_epoch)
              for data in itertools.islice(train_dataset, None, steps_per_epoch):
                  images, label = data
                  masked_images = images['image_raw']

                  fake_images = generator.predict(masked_images)
                  disc_loss_real = discriminator.train_on_batch(label, np.ones(len(label), dtype='float32'))
                  disc_loss_fake = discriminator.train_on_batch(fake_images, np.zeros(len(fake_images), dtype='float32'))

                  disc_loss = (disc_loss_real + disc_loss_fake)/2
                  progressbar.add(1, values = [('Discriminator Loss', disc_loss)])

      # Training Gan
      print("Training Gan")
      for current_epoch in range(gan_epoch):
            print('Epoch {}/{}'.format(current_epoch, gan_epoch))
            progressbar = generic_utils.Progbar(steps_per_epoch)
            for data in itertools.islice(train_dataset, None, steps_per_epoch):
                images, label = data
                masked_images = images['image_raw']
                fake_images = generator.predict(masked_images)

                disc_loss_real = discriminator.train_on_batch(label, np.ones(len(label), dtype='float32'))
                disc_loss_fake = discriminator.train_on_batch(fake_images, np.zeros(len(fake_images), dtype='float32'))

                disc_loss = (disc_loss_real + disc_loss_fake)/2

                gan_loss = gan.train_on_batch(masked_images, [label, np.ones((len(label), 1), dtype='float32')])

                progressbar.add(1, values = [('Discriminator Loss', disc_loss), ('GAN Loss', gan_loss[0]), ('Generator Loss', gan_loss[1])])

      save_model(generator,fn_args.serving_model_dir)
      print('serving model dir',fn_args.serving_model_dir)
