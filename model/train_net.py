"""Training NN for Face Recognition"""
import os
import sys
import math
import keras 
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Layer
from keras.layers import (Conv2D, 
                          BatchNormalization, 
                          ReLU, 
                          DepthwiseConv2D, 
                          Activation, 
                          Input, 
                          Add, 
                          Flatten, 
                          Dense, 
                          Lambda,
                          Softmax)
                          
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import time
from matplotlib import pyplot as plt

from MobileFaceNetModel import *
from generate_dataset import *

"""Custom Training Step for Face Rec Model"""
def train_model(model, train_dataset, opt):
    epochList = []
    lossList = []
    for epoch in range(NUM_EPOCHS):
        lossVal = 0
        print("\nStart of epoch %d" % (epoch,))
        # Iterate over the batches of the dataset.
        for step, (x_batch_train) in enumerate(train_dataset):
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            # print(step)
            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                # print("Batch_train_size:", x_batch_train.shape)
                logits = model(x_batch_train, training=True)  # Logits for this minibatch
                # print(logits.shape)
                # Compute the loss value for this minibatch.
                loss_value = triplet_loss(logits, logits)
            lossVal += loss_value
            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)
            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            opt.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 200 batches.
            if step % 20 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * 1))
        epochList.append(epoch)
        lossList.append(lossVal / len(train_dataset))
    
    plt.plot(epochList, lossList)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("training_loss.png")
    plt.close()

# default model
FRmodel = build_model()

# check to see if a previous best model was saved --> resote if it was
best_model_path=None
if os.path.exists("./bestmodel.txt"):
    with open('bestmodel.txt', 'r') as file:
        best_model_path = file.read()
    
if best_model_path != None and os.path.exists(best_model_path):
    print("Pre trained model found")
    FRmodel = keras.models.load_model(best_model_path, custom_objects={'triplet_loss': triplet_loss})
else:
    print('Saved model not found, loading untrained FaceNet')

input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)

# create three separate models for anchor, positive, and negative images --> using model.fit() implementation
# A = Input(shape=input_shape, name = 'anchor')
# P = Input(shape=input_shape, name = 'anchorPositive')
# N = Input(shape=input_shape, name = 'anchorNegative')

# enc_A = FRmodel(A)
# enc_P = FRmodel(P)
# enc_N = FRmodel(N)

# early stopping, and create checkpoint logs. 
early_stopping = EarlyStopping(monitor='loss', patience=10, min_delta=0.00005)
MARKER = 'facenet_%d'%(len(paths)) 
checkpoint_dir = './' + 'checkpoints/' + str(int(time.time())) + '/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

bst_model_path = checkpoint_dir + MARKER + '.h5'
tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

# using model.fit()
# tripletModel = Model([A, P, N], [enc_A, enc_P, enc_N])
# tripletModel.compile(optimizer = 'adam', loss = triplet_loss)
# gen = batch_generator(BATCH_SIZE)
# tripletModel.fit(gen, epochs=NUM_EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,callbacks=[early_stopping, tensorboard])

# using custom defined training step
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
FRmodel.compile(optimizer = opt, loss = triplet_loss)

data_set = generate_dataset()
train_dataset = tf.data.Dataset.from_tensor_slices((data_set))
train_dataset = train_dataset.shuffle(buffer_size=200)
print("train_dataset Size:", len(list(train_dataset)))
train_model(FRmodel, train_dataset, opt)

# save model
FRmodel.save(bst_model_path)
with open('bestmodel.txt','w') as file:
    file.write(bst_model_path)
