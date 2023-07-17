#IMPORTING NECESSARY LIBRARIES

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import math
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, Reshape, Concatenate, MaxPooling2D
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, ReLU, Dropout, Flatten, LeakyReLU, BatchNormalization
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.initializers import RandomNormal

TRAIN_PATH_CSV = 'dataset\emnist-balanced-train.csv'
TEST_PATH_CSV = 'dataset\emnist-balanced-test.csv'
MAPPING_PATH = 'dataset\mapping.txt'
NUM_CLASSES = 47

GENERATOR_PATH = 'models\Generator_Model.h5'
DISCRIMINATOR_PATH = 'models\Discriminator_Model.h5'


def viewTrainImages(height,width):
    train_data = pd.read_csv(TRAIN_PATH_CSV, header = None)
    mapping = pd.read_csv(MAPPING_PATH,sep = ' ', header = None)

    train_data.rename(columns= {0: 'label'}, inplace = True)

    Y_train = train_data['label']
    Y_train = Y_train.values.reshape(train_data.shape[0],1).astype('float32')

    X_train = train_data.drop(columns = 'label')
    X_train = X_train.values.reshape(train_data.shape[0],28,28,1).astype('float32')

    mapping = mapping.values
    for i in range(height):
        for j in range(width):
            n = 20*i + j + 1
            plt.subplot(height, width, width * i + j + 1)
            plt.imshow(X_train[n])
            character = chr(mapping[int(Y_train[n])][1])
            plt.gca().set_title(character)
            plt.axis('off')


#Generator generates the fake images
def Generator(Noise_Dim=(100,) , ):
    #Generator takes two inputs, one is the label ,type of character
    #the other is a random noise array which becomes the image
    LabelInput = Input((1,), name='Label_Input')
    NoiseInput = Input(Noise_Dim, name='Noise_Input')
    
    side = 7
    f1 = 32
    f2 = 32
    
    Label = Dense(side*side*f1, name='Dense_Label')(LabelInput)
    Label = Reshape((side,side,f1), name='Reshape_Label')(Label)
    
    Noise = Dense(side*side*f2, name='Dense_Noise')(NoiseInput)
    Noise = Reshape((side,side,f2), name='Reshape_Noise')(Noise)
    
    Concat1 = Concatenate(name='Concat')([Label,Noise])
    
    x = Conv2DTranspose(128, 4, 2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='Conv_Transpose_1')(Concat1)
    x = BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='BN_1')(x)
    x = ReLU(name='ReLU_1')(x)
     
    x = Conv2DTranspose(64, 4, 2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='Conv_Transpose_2')(x)
    x = BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='BN_2')(x)
    x = ReLU(name='ReLU_2')(x)
     
    x = Conv2DTranspose(64, 4, 2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='Conv_Transpose_3')(x)
    x = BatchNormalization(momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='BN_3')(x)
    x = ReLU(name='ReLU_3')(x)
   
 
    x = Conv2DTranspose(64, 4, 2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='Conv_Transpose_4')(x)
    x = BatchNormalization(momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='BN_4')(x)
    x = ReLU(name='ReLU_4')(x) 
    
    x = Conv2D(128, 4, 2, padding='same', name='Conv2D_1')(x)
    x = BatchNormalization(momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='BN_5')(x)
    x = ReLU(name='reLU_5')(x)
    
    x = Conv2D(64 , 4, 1, padding='same', name='Conv2D_2')(x)
    x = BatchNormalization(momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='BN_6')(x)
    x = ReLU(name='ReLU_6')(x)
    
    x = Conv2D(16, 2, 1, padding='same', name='Conv2D_3')(x)
    x = BatchNormalization(momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='BN_7')(x)
    x = ReLU(name='ReLU_7')(x)
    
    x = Conv2D(1, 1, 2, padding='same', name='Conv2D_4')(x)
    x = BatchNormalization(momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='BN_8')(x)
    x = ReLU(name='ReLU_8')(x)
    
   # define model
    model = Model([LabelInput, NoiseInput], x)
    return model

#Discriminator finds the real and fake images. A real image is a 1
def Discriminator(Img_Dim=(28,28,1)):
    #Discriminator also takes 2 inputs
    #One is the label and other is the Image that it needs to identify as real or fake
    LabelInput = Input((1,), name='Label_Input')
    ImgInput = Input(Img_Dim, name='Img_Input')
    
    Label = Dense(32, name='Dense_Label_1')(LabelInput)
    Label = Dense(3136, name='Dense_Label_2')(Label)
    Label = Reshape((28,28,4))(Label)

    Img = Conv2D(7, 3 , padding='same', name='Conv2D_Img_1')(ImgInput)
    Img = Conv2D(14, 3 , padding='same', name='Conv2D_Img_2')(Img)
    Img = Conv2D(28, 3 , padding='same', name='Conv2D_Img_3')(Img)
    
    x = Concatenate()([Label,Img])
    
    x = Conv2D(64, 4, 2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='Conv_1')(x)
    x = LeakyReLU(0.2, name='Leaky_ReLU_1')(x)
    
    x = Conv2D(128, 4, 3, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='Conv_2')(x)
    x = BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='BN_1')(x)
    x = LeakyReLU(0.2, name='Leaky_ReLU_2')(x)
    
    x = Conv2D(256, 4, 3, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='Conv_3')(x)
    x = BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='BN_2')(x)
    x = LeakyReLU(0.2, name='Leaky_ReLU_3')(x)
    
    x = Conv2D(256, 4, 3, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), use_bias=False, name='Conv_4')(x)
    x = BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='BN_3')(x)
    x = LeakyReLU(0.2, name='Leaky_ReLU_4')(x)
    
    x = Flatten(name='Flatten')(x)
    x = Dropout(0.4, name='Dropout')(x)
    x = Dense(1, activation='sigmoid', name='Dense')(x)
    
    Discriminator_Model = Model([LabelInput, ImgInput], x)
    Discriminator_Model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.002, beta_1=0.5), metrics=['accuracy'])
    return Discriminator_Model

#Merge both the models to get cGAN
def CGAN(Generator_Model, Discriminator_Model):
    #We dont want to train it initially
    Discriminator_Model.trainable = False
    
    #Getting generator model inputs and outputs
    GeneratorLabelInput, GeneratorNoiseInput = Generator_Model.input
    GeneratorImgOutput = Generator_Model.output
    
    #Generators image go into discriminator
    DiscriminatorOutput = Discriminator_Model([GeneratorLabelInput, GeneratorImgOutput])
    
    #CGAN takes label and Generator noise from starting as input
    #and verdict from discriminator as output
    CGAN_Model = Model([GeneratorLabelInput, GeneratorNoiseInput], DiscriminatorOutput)
    
    CGAN_Model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.002, beta_1=0.5))
    return CGAN_Model

#Real sample, i.e., from the dataset , X_train
def real_sample(n, data, mapping):
    #Take some random indexes in dataset
    rand_ind = np.random.randint(0,data.shape[0],n)
    
    #Collect data at these random indexes
    X = data[rand_ind]
    Label = mapping[rand_ind]
    Y = np.ones((n,1))
    
    return X, Y, Label

#This just generates a noise vector along with a random label
def random_noise(n,Noise_Dim=100):
    random_val = np.random.randn(Noise_Dim*n)
    random_val = random_val.reshape(n, Noise_Dim)
    random_label = np.random.randint(0,NUM_CLASSES,n)
    random_label.reshape(n,1)
    
    return random_val, random_label

#This generates a fake sample using Generator's help
def fake_sample(n,Generator_Model,Noise_Dim=100):
    random_val , random_label = random_noise(n,Noise_Dim)
    
    X = Generator_Model.predict([random_label , random_val])
    Y = np.zeros((n,1))
    
    return X,Y,random_label

#A function to visualise our Generator's performance
#Expectation is that during early stages generator just gives some noise
#On later stage it should give characters that look real
def Generator_Image(Generator_Model,length,width,Noise_Dim=100):
    img, Y_fake, labels_fake = fake_sample(length*width,Generator_Model,Noise_Dim)
    #Making a subplot to show width X height number of images
    fig, axs = plt.subplots(height, width)
    for i in range(height):
        for j in range(width):
            n = i * width + j
            axs[i, j].imshow(img[n], cmap='gray')
            axs[i, j].axis('off')
            character = chr(mapping[int(labels_fake[n])][1])
            axs[i, j].set_title(character)
    plt.show()

def train(Generator_Model, Discriminator_Model, GAN_Model,Data, LabelMapping, Noise_Dim=50, Num_Epochs=6, Num_Batch=256):
    #Number of batched per epoch
    Batch_per_epoch = int(Data.shape[0] / Num_Batch)
    print('Batch per epoch : ',Batch_per_epoch)
    
    #Dividing batch because Discriminator first trains on real images
    #then on fake images
    Half_batch = int(Num_Batch/2)
    
    for i in range(Num_Epochs):
        print('________________________\nEpoch : %d/%d \n'%(i+1,Num_Epochs))
        for j in range(Batch_per_epoch):
        #Discriminator Training
            #Real Samples
            X_Real, Y_Real, Label_Real = real_sample(Half_batch, Data, LabelMapping)
            Discriminator_Loss1, _ = Discriminator_Model.train_on_batch([Label_Real, X_Real], Y_Real)
            
            #Fake Samples
            X_Fake,Y_Fake,Label_Fake = fake_sample(Half_batch,Generator_Model,Noise_Dim)
            Discriminator_Loss2, _ = Discriminator_Model.train_on_batch([Label_Fake, X_Fake], Y_Fake)
            
        #Generator Training
            random_val, random_label = random_noise(Num_Batch,Noise_Dim)
            Y = np.ones((Num_Batch, 1))
            Generator_Loss = GAN_Model.train_on_batch([random_label, random_val], Y)
            print('Batch: %d/%d, D_Loss_Real=%.3f, D_Loss_Fake=%.3f Gen_Loss=%.3f' %  (j+1, Batch_per_epoch, Discriminator_Loss1, Discriminator_Loss2, Generator_Loss))
        Generator_Image(Generator_Model,5,5,Noise_Dim)

        
def give_models(mode):
    if os.path.isfile(GENERATOR_PATH) and os.path.isfile(DISCRIMINATOR_PATH):
        print("Model Found !")
        with open(GENERATOR_PATH, 'rb') as file1, open(DISCRIMINATOR_PATH, 'rb') as file2:
            Gen = load_model(file1)
            Dis = load_model(file2)

    else:
        print("Model(s) Not Found !")
        print("Making the model.....")
        Gen = Generator()
        print("\nGenerator Is : \n")
        Gen.summary()
        #Generator model plot
        plot_model(Gen, show_shapes=True, show_layer_names=True, to_file='Generator_model.png')

        Dis = Discriminator()
        print("\nDiscriminator Is : \n")
        Dis.summary()
        #Discriminator model plot
        plot_model(Dis, show_shapes=True, show_layer_names=True, to_file='Discriminator_model.png')


    if mode=='Predict':
        return Gen
    
    CGAN_Model = CGAN(Gen,Dis)
    print("\nCGAN Model Is : \n")
    CGAN_Model.summary()
    #cGAN model plot
    plot_model(CGAN_Model, show_shapes=True,show_layer_names=True, to_file='cGAN_model.png')

    if mode=='Train':
        return Gen, Dis, CGAN_Model
    
    print("Invalid Input\n")



def train_models():
    #PREPROCESSING
    train_data = pd.read_csv(TRAIN_PATH_CSV, header = None)
    test_data = pd.read_csv(TEST_PATH_CSV, header = None)
    mapping = pd.read_csv(MAPPING_PATH,sep = ' ', header = None)

    print("Shape of train_data is :",train_data.shape)
    print("Shape of test_data is :",test_data.shape)

    train_data.rename(columns= {0: 'label'}, inplace = True)
    test_data.rename(columns= {0: 'label'}, inplace = True)

    Y_train = train_data['label']
    Y_train = Y_train.values.reshape(train_data.shape[0],1).astype('float32')

    X_train = train_data.drop(columns = 'label')
    X_train = X_train.values.reshape(train_data.shape[0],28,28,1).astype('float32')

    Y_test = test_data['label']
    Y_test = Y_test.values.reshape(test_data.shape[0],1).astype('float32')

    X_test = test_data.drop(columns = 'label')
    X_test = X_test.values.reshape(test_data.shape[0],28,28,1).astype('float32')

    mapping = mapping.values

    #Normalize
    X_train = X_train/255.0
    X_test = X_test/255.0

    #Gather the models
    Gen, Dis, CGAN_Model = give_models('Train')

    NUM_EPOCHS = int(input("Enter number of epochs (Default is 6): "))
    NUM_BATCH = int(input("Enter batches size (Default is 256): "))
    #training start
    train(Gen, Dis, CGAN_Model ,X_train, Y_train, Noise_Dim=100, Num_Epochs=NUM_EPOCHS, Num_Batch=NUM_BATCH)
    print("\n\n----------------------------------\nDone")

    Gen.save(GENERATOR_PATH)
    Dis.save(DISCRIMINATOR_PATH)

    return Gen, Dis