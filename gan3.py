import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
# import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
# Define dimensions

# plt.show(block=False)  # Non-blocking mode
# Load the MNIST dataset
# (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_x = np.load("x32.npy")
train_y = np.load("y32.npy")

input_shape = (32, 32, 3)  # Input shape for the discriminator

# train_images = np.load("gray.npy")
# print(train_images.shape)
# exit()
train_x = train_x.reshape(train_x.shape[0],input_shape[0],input_shape[1],input_shape[2]).astype('float32')
train_x = (train_x - 127.5) / 127.5  # Normalize the images to [-1, 1]

train_y = train_y.reshape(train_y.shape[0],input_shape[0],input_shape[1],input_shape[2]).astype('float32')
train_y = (train_y - 127.5) / 127.5  # Normalize the images to [-1, 1]

# Shuffle and batch the data
BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Assuming you have paired datasets: train_x and train_y
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)




def build_generator(input_shape=(32, 32, 3)):
    model = tf.keras.Sequential()

    # Use an explicit Input layer to take in the image 'x'
    model.add(layers.Input(shape=input_shape))

    # Apply convolutional layers to modify 'x' and generate 'y'
    model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))  # Downscale to (16, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))  # Downscale to (8, 8)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Now, upsample back to the original shape (32, 32, 3)
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))  # Upscale to (16, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))  # Upscale to (32, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Output layer: Transformed image of the same size (32x32 with 3 channels)
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))

    return model

generator = build_generator(input_shape=(input_shape))


def build_discriminator(input_shape=(32, 32, 3)):
    # Input layers for both x and y
    input_x = layers.Input(shape=input_shape)
    input_y = layers.Input(shape=input_shape)

    # Processing input_x
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(input_x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)

    # Processing input_y
    y = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(input_y)
    y = layers.LeakyReLU()(y)
    y = layers.Dropout(0.3)(y)

    y = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(y)
    y = layers.LeakyReLU()(y)
    y = layers.Dropout(0.3)(y)

    y = layers.Flatten()(y)

    # Concatenate the flattened x and y
    combined = layers.concatenate([x, y])

    # Dense layers for binary classification
    out = layers.Dense(128)(combined)
    out = layers.LeakyReLU()(out)
    out = layers.Dense(1)(out)  # Binary classification (real/fake)

    # Define the model with two inputs and one output
    model = tf.keras.Model(inputs=[input_x, input_y], outputs=out)

    return model

discriminator = build_discriminator(input_shape=(input_shape))



cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def load_best_model(directory, model_type):
    model_files = [f for f in os.listdir(directory) if model_type in f and f.endswith('.keras')]
    model_files.sort()  # Ensure the models are sorted by filename
    if model_files:
        best_model_file = model_files[-1]  # Assuming the highest epoch number is the best
        model_path = os.path.join(directory, best_model_file)
        print(f"Loading {model_type} model from: {model_path}")
        return load_model(model_path)
    else:
        print(f"No {model_type} models found in {directory}.")
        return None
# generator = load_best_model("gan", 'generator')
# discriminator = load_best_model("gan", 'discriminator')

# Optimizers for both networks
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


@tf.function
def train_step(real_x, real_y):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generator creates a generated 'y' from 'real_x'
        generated_y = generator(real_x, training=True)

        # Discriminator evaluates the real (x, y) and fake (x, generated_y) pairs
        real_output = discriminator([real_x, real_y], training=True)
        fake_output = discriminator([real_x, generated_y], training=True)

        # Calculate losses for generator and discriminator
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Calculate and apply gradients for both networks
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs,b):
    for epoch in range(epochs):
        for real_x, real_y in dataset:
            train_step(real_x, real_y)  # Train step with both inputs

        # Generate and save images every few epochs using real input images
        # generate_and_save_images(generator, epoch + 1, real_x)

        # Optionally save the model every 100 epochs
        if epoch % 100 == 0:
            save_models(epoch,b)

        # Optionally print the progress
        print(f'Epoch {epoch}/{epochs} complete')

# Function to save the generator and discriminator models
def save_models(epoch,b):
    generator.save(f'gan/generator_epoch_{epoch+b}.keras')
    discriminator.save(f'gan/discriminator_epoch_{epoch+b}.keras')
    print(f'Models saved at epoch {epoch}')
# Function to generate and display images during training
def generate_and_save_images(model, epoch, test_input):
    if epoch % 1 == 0:
        # Generate images using the generator model
        generated_images = model(test_input, training=False)

        # Limit the number of images to 16
        num_images = min(generated_images.shape[0], 16)

        # Create a 4x4 grid for the generated images
        fig = plt.figure(figsize=(4, 4))

        # Loop through the first 16 generated images and display them
        for i in range(num_images):
            plt.subplot(4, 4, i+1)
            image = generated_images[i] * 127.5 + 127.5  # Rescale the generated image to [0, 255]

            # Clip the values to ensure they're in the valid range [0, 255]
            image = np.clip(image, 0, 255)

            # Display the image as a colored image
            plt.imshow(image.astype(np.uint8))  # Convert to uint8 for proper visualization
            plt.axis('off')  # Turn off axis for cleaner image display



        plt.show()



# Train the GAN for a set number of epochs
EPOCHS = 1000
EPOCHS = EPOCHS +1
BATCH_SIZE = 32  # Define the batch size
b=0
train(train_dataset, EPOCHS,b)