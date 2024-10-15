import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load your pre-trained TensorFlow model (.keras or .h5)
model = tf.keras.models.load_model('gan\generator_epoch_300.keras')
model2 = tf.keras.models.load_model('gan\discriminator_epoch_300.keras')
# Function to normalize image to [-1, 1]
def normalize_image(image):
    image = np.array(image)
    image = image / 127.5 - 1.0
    return image

# Function to denormalize image from [-1, 1] back to [0, 255]
def denormalize_image(image):
    image = (image + 1.0) * 127.5
    return np.clip(image, 0, 255).astype(np.uint8)

# Load and preprocess the input image
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).resize((128, 128))  # Adjust size as needed
    image = np.array(image)
    image = normalize_image(image)
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Display input and generated images
def display_images(input_image, generated_image,input_image2):
    plt.figure(figsize=(8, 4))

    # Display original input image
    plt.subplot(1, 3, 1)
    plt.title('Input Image')
    plt.imshow(denormalize_image(input_image2[0]))
    plt.axis('off')

    # Display original input image
    plt.subplot(1, 3, 2)
    plt.title('Input Image')
    plt.imshow(denormalize_image(input_image[0]))
    plt.axis('off')

    # Display generated image
    plt.subplot(1, 3, 3)
    plt.title('Generated Image')
    plt.imshow(denormalize_image(generated_image[0]))
    plt.axis('off')

    plt.show()
score = 0
for i in range(0,500):
    # Path to your input image
    input_image_path = f'x/image_{i}.png'
    input_image_path2 = f'y/image_{i}.png'
    # Load and preprocess the image
    input_image = load_and_preprocess_image(input_image_path)
    input_image2 = load_and_preprocess_image(input_image_path2)
    # Run the model to generate the output
    generated_image = model.predict(input_image)
    prediction = model2.predict([input_image,input_image2])
    prediction2 = model2.predict([input_image,generated_image])
    # Print the prediction result
    if prediction>0:
        if prediction2>0:
            score+=1
            print("gan wins")
            # Print the prediction result
            print("Discriminator output:", prediction)
            # Display both input and generated images
            # display_images(input_image, generated_image, input_image2)

        else:
            print("gen loss")

print(score/i)