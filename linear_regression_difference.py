import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras import layers, models
from keras.optimizers import Adam

size = 64
dir_data = "image folder"  #place images folder name here
Ntrain = 40000  # 150000 may cause a memory error.
# Ntrain = 10000
nm_imgs = np.sort(os.listdir(dir_data))
nm_imgs_train = nm_imgs[:Ntrain]
img_shape = (size, size, 3)
optimizer = Adam(0.00007, 0.5)
noise_shape = (100,)


def get_npdata(nm_imgs_train):
    x = []
    for i, myid in enumerate(nm_imgs_train):
        image = load_img(dir_data + "/" + myid, target_size=img_shape[:2])
        image = img_to_array(image)/255.0
        x.append(image)
    x = np.array(x)
    return x


X_train = get_npdata(nm_imgs_train)
print("X_train.shape = {}".format(X_train.shape))


def build_generator(img_shape, noise_shape):
    input_noise = layers.Input(shape=noise_shape)
    d = layers.Dense(1024, activation="relu")(input_noise)
    d = layers.Dense(1024, activation="relu")(input_noise)
    d = layers.Dense(128 * 8 * 8, activation="relu")(d)
    d = layers.Reshape((8, 8, 128))(d)
    d = layers.Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), use_bias=False)(d)
    d = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(d)  # 16,16
    d = layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), use_bias=False)(d)
    d = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(d)  # 32,32
    if img_shape[0] == 64:
        d = layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), use_bias=False)(d)
        d = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(d)  # 64,64
    img = layers.Conv2D(3, (1, 1), activation='sigmoid', padding='same')(d)  # 32, 32
    model = models.Model(input_noise, img)
    model.summary()
    return model


generator = build_generator(img_shape, noise_shape)
generator.compile(loss='binary_crossentropy', optimizer=optimizer)


def get_noise(nsample=1, nlatent_dim=100):
    noise = np.random.normal(0, 1, (nsample, nlatent_dim))
    return noise


def plot_generated_images(noise, path_save=None, titleadd="", nsample=4):
    imgs = generator.predict(noise)
    fig = plt.figure(figsize=(40, 10))
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(1, nsample, i + 1)
        ax.imshow(img)
    fig.suptitle("Generated images " + titleadd, fontsize=30)
    plt.savefig(path_save, bbox_inches='tight', pad_inches=0)
    plt.close()


def build_discriminator(img_shape, noutput=1):
    input_img = layers.Input(shape=img_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(1, 1))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    out = layers.Dense(noutput, activation='sigmoid')(x)
    model = models.Model(input_img, out)
    return model


discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
discriminator.summary()

z = layers.Input(shape=(100,))
img = generator(z)
discriminator.trainable = False
valid = discriminator(img)
combined = models.Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)
combined.summary()


def train(models, X_train, noise_plot, dir_result, epochs=10000, batch_size=128):
    combined, discriminator, generator = models
    nlatent_dim = noise_plot.shape[1]
    half_batch = int(batch_size / 2)
    history = []
    for epoch in range(epochs):

        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx]
        noise = get_noise(half_batch, nlatent_dim)

        gen_imgs = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = get_noise(batch_size, nlatent_dim)
        valid_y = np.ones((batch_size, 1))
        g_loss = combined.train_on_batch(noise, valid_y)

        history.append({"D": d_loss[0], "G": g_loss})
        if epoch % 100 == 0:
            print("Epoch {:05.0f} [D loss: {:4.3f}, acc.: {:05.1f}%] [G loss: {:4.3f}]".format(
                epoch, d_loss[0], 100 * d_loss[1], g_loss))
        if epoch % 100 == 0:
            plot_generated_images(noise_plot, path_save=dir_result + "/image_{:05.0f}.png".format(epoch), titleadd="Epoch {}".format(epoch))
    return history


dir_result = "./result_GAN/"
if not os.path.exists(dir_result):
    os.makedirs(dir_result)
nsample = 4
noise_plot = get_noise(nsample=nsample, nlatent_dim=noise_shape[0])
_models = combined, discriminator, generator
#bs = 128  # faster epochs
bs = 64
history = train(_models, X_train, noise_plot, dir_result=dir_result, epochs=20000, batch_size=bs)





































