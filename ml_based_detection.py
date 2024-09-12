# Imports
import numpy as np
import tensorflow as tf
import pathlib
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def carregar_dados(data_dir, img_height, img_width, batch_size):

    # Carrega e prepara os datasets de treinamento e validação.

    data_dir = pathlib.Path(data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names

    return train_ds, val_ds, class_names

def construir_modelo(img_height, img_width, num_classes):

    # Constrói e retorna um modelo CNN simples para classificação de imagens.

    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model

def main():
    # Base de dados com 2 classes, desmatadas e normal
    data_dir = r"Caminho/para/base/de/dados"
    batch_size = 16
    img_height = 300
    img_width = 300

    # Carregar os dados de treino e validação
    train_ds, val_ds, class_names = carregar_dados(data_dir, img_height, img_width, batch_size)

    # Otimizar os datasets
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Construir o modelo
    num_classes = len(class_names)
    model = construir_modelo(img_height, img_width, num_classes)

    # Treinar o modelo
    epochs = 1
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    # Fazer previsões
    img_path = r'C:\Users\paulo.jesus\Desktop\floresta\floresta1.jpg'
    img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Adiciona uma dimensão de lote

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print('Esta floresta está {}'.format(class_names[np.argmax(score)]))

if __name__ == "__main__":
    main()
