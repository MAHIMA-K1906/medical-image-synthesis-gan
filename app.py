import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import io

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Medical Image Synthesis using GAN",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Medical Image Synthesis using GAN")

st.write(
    "Generate synthetic medical images using a "
    "Generative Adversarial Network (GAN)."
)

st.info(
    "Dataset: ISIC 2018 | Image Size: 64 × 64 | "
    "Framework: TensorFlow + Streamlit"
)

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

IMG_SIZE = 64
CHANNELS = 3
LATENT_DIM = 100


# --------------------------------------------------
# GENERATOR
# --------------------------------------------------

def build_generator():

    model = tf.keras.Sequential(
        name="Generator"
    )

    model.add(
        layers.Dense(
            8 * 8 * 256,
            input_shape=(LATENT_DIM,)
        )
    )

    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Reshape((8, 8, 256))
    )

    model.add(
        layers.Conv2DTranspose(
            128,
            kernel_size=4,
            strides=2,
            padding="same"
        )
    )

    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            64,
            kernel_size=4,
            strides=2,
            padding="same"
        )
    )

    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            32,
            kernel_size=4,
            strides=2,
            padding="same"
        )
    )

    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2D(
            CHANNELS,
            kernel_size=3,
            padding="same",
            activation="tanh"
        )
    )

    return model


# --------------------------------------------------
# DISCRIMINATOR
# --------------------------------------------------

def build_discriminator():

    model = tf.keras.Sequential(
        name="Discriminator"
    )

    model.add(
        layers.Conv2D(
            64,
            kernel_size=4,
            strides=2,
            padding="same",
            input_shape=(
                IMG_SIZE,
                IMG_SIZE,
                CHANNELS
            )
        )
    )

    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv2D(
            128,
            kernel_size=4,
            strides=2,
            padding="same"
        )
    )

    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv2D(
            256,
            kernel_size=4,
            strides=2,
            padding="same"
        )
    )

    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())

    model.add(
        layers.Dense(1)
    )

    return model


# --------------------------------------------------
# CREATE MODELS
# --------------------------------------------------

@st.cache_resource
def create_models():

    generator = build_generator()
    discriminator = build_discriminator()

    return generator, discriminator


generator, discriminator = create_models()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("⚙️ GAN Settings")

epochs = st.sidebar.slider(
    "Training Epochs",
    min_value=1,
    max_value=20,
    value=5
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    [8, 16, 32],
    index=0
)

num_images = st.sidebar.slider(
    "Images to Generate",
    min_value=1,
    max_value=16,
    value=4
)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.subheader("📂 Upload Medical Images")

uploaded_files = st.file_uploader(
    "Upload ISIC 2018 images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# --------------------------------------------------
# PREPROCESS IMAGES
# --------------------------------------------------

def preprocess_images(files):

    images = []

    for file in files:

        try:

            image = Image.open(file).convert("RGB")

            image = image.resize(
                (IMG_SIZE, IMG_SIZE)
            )

            image = np.array(image)

            image = image.astype(
                np.float32
            )

            # Normalize to [-1, 1]
            image = (
                image / 127.5
            ) - 1.0

            images.append(image)

        except Exception:
            pass

    return np.array(images)


# --------------------------------------------------
# GAN LOSS
# --------------------------------------------------

cross_entropy = tf.keras.losses.BinaryCrossentropy(
    from_logits=True
)


def discriminator_loss(
    real_output,
    fake_output
):

    real_loss = cross_entropy(
        tf.ones_like(real_output),
        real_output
    )

    fake_loss = cross_entropy(
        tf.zeros_like(fake_output),
        fake_output
    )

    return real_loss + fake_loss


def generator_loss(fake_output):

    return cross_entropy(
        tf.ones_like(fake_output),
        fake_output
    )


# --------------------------------------------------
# OPTIMIZERS
# --------------------------------------------------

generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002,
    beta_1=0.5
)

discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002,
    beta_1=0.5
)


# --------------------------------------------------
# TRAINING STEP
# --------------------------------------------------

@tf.function
def train_step(images):

    noise = tf.random.normal(
        [tf.shape(images)[0], LATENT_DIM]
    )

    with tf.GradientTape() as gen_tape, \
         tf.GradientTape() as disc_tape:

        generated_images = generator(
            noise,
            training=True
        )

        real_output = discriminator(
            images,
            training=True
        )

        fake_output = discriminator(
            generated_images,
            training=True
        )

        gen_loss = generator_loss(
            fake_output
        )

        disc_loss = discriminator_loss(
            real_output,
            fake_output
        )

    gradients_of_generator = gen_tape.gradient(
        gen_loss,
        generator.trainable_variables
    )

    gradients_of_discriminator = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(
            gradients_of_generator,
            generator.trainable_variables
        )
    )

    discriminator_optimizer.apply_gradients(
        zip(
            gradients_of_discriminator,
            discriminator.trainable_variables
        )
    )

    return gen_loss, disc_loss


# --------------------------------------------------
# TRAIN BUTTON
# --------------------------------------------------

if st.button(
    "🚀 Train GAN",
    type="primary",
    use_container_width=True
):

    if not uploaded_files:

        st.warning(
            "⚠️ Please upload ISIC medical images first."
        )

    elif len(uploaded_files) < 4:

        st.warning(
            "⚠️ Please upload at least 4 images "
            "for training."
        )

    else:

        data = preprocess_images(
            uploaded_files
        )

        dataset = tf.data.Dataset.from_tensor_slices(
            data
        )

        dataset = dataset.shuffle(
            len(data)
        ).batch(
            batch_size,
            drop_remainder=False
        )

        progress = st.progress(0)

        status = st.empty()

        for epoch in range(epochs):

            gen_loss_value = 0
            disc_loss_value = 0

            for batch in dataset:

                gen_loss_value, disc_loss_value = train_step(
                    batch
                )

            progress.progress(
                (epoch + 1) / epochs
            )

            status.write(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Generator Loss: "
                f"{float(gen_loss_value):.4f} | "
                f"Discriminator Loss: "
                f"{float(disc_loss_value):.4f}"
            )

        st.success(
            "✅ GAN training completed!"
        )


# --------------------------------------------------
# GENERATE IMAGES
# --------------------------------------------------

st.divider()

st.subheader(
    "🧪 Generate Synthetic Medical Images"
)

if st.button(
    "✨ Generate Images",
    use_container_width=True
):

    noise = tf.random.normal(
        [num_images, LATENT_DIM]
    )

    generated_images = generator(
        noise,
        training=False
    )

    generated_images = (
        generated_images + 1
    ) / 2

    generated_images = tf.clip_by_value(
        generated_images,
        0,
        1
    )

    cols = st.columns(4)

    for i in range(num_images):

        image = generated_images[i].numpy()

        image = (
            image * 255
        ).astype(np.uint8)

        pil_image = Image.fromarray(
            image
        )

        with cols[i % 4]:

            st.image(
                pil_image,
                caption=f"Synthetic Image {i + 1}",
                use_container_width=True
            )

            buffer = io.BytesIO()

            pil_image.save(
                buffer,
                format="PNG"
            )

            st.download_button(
                "⬇️ Download",
                buffer.getvalue(),
                file_name=f"synthetic_{i + 1}.png",
                mime="image/png",
                key=f"download_{i}"
            )


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("📌 Project Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Dataset",
        "ISIC 2018"
    )

with col2:

    st.metric(
        "Model",
        "GAN"
    )

with col3:

    st.metric(
        "Image Size",
        "64 × 64"
    )

st.caption(
    "Medical Image Synthesis using GAN "
    "(ISIC 2018 Dataset) with Cloud Deployment"
)
