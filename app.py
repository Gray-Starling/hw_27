import time
import os
import zipfile
import wget
import streamlit as st
import tensorflow as tf
import numpy as np
import shutil
from PIL import Image
import tensorflow as ts
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from keras.callbacks import ModelCheckpoint, Callback
import time

IMG_WIDTH, IMG_HEIGHT = 288, 288



st.sidebar.title("Классификация кошек и собак")
st.sidebar.write("Укажите желаемое количество эпох, на которых будет обучаться модель")
user_epochs = st.sidebar.slider('Количество эпох', 1, 20, 5)
st.sidebar.write("Выберите файл для классификации")
uploaded_file = st.sidebar.file_uploader("Choose a file")

def get_data():
    with st.spinner("Скачивание и распаковка данных..."):
        if not os.path.exists("cat-and-dog.zip"):
            url = "https://storage.yandexcloud.net/academy.ai/cat-and-dog.zip"
            wget.download(url)
        with zipfile.ZipFile("cat-and-dog.zip", 'r') as zip_ref:
            zip_ref.extractall("./temp")
        time.sleep(2)

    st.success("Данные скачаны и распакованы!")

def copy():
    TRAIN_CATS_PATH = './temp/training_set/training_set/cats'
    TRAIN_DOGS_PATH = './temp/training_set/training_set/dogs'

    TEST_CATS_PATH = './temp/test_set/test_set/cats'
    TEST_DOGS_PATH = './temp/test_set/test_set/dogs'

    COMBINED_DATASET_PATH = './temp/combined_dataset'

    if not os.path.exists(COMBINED_DATASET_PATH):
            os.mkdir(COMBINED_DATASET_PATH)

    if not os.path.exists(os.path.join(COMBINED_DATASET_PATH, 'cats')):
        os.mkdir(os.path.join(COMBINED_DATASET_PATH, 'cats'))

    if not os.path.exists(os.path.join(COMBINED_DATASET_PATH, 'dogs')):
        os.mkdir(os.path.join(COMBINED_DATASET_PATH, 'dogs'))
        
    def copy_files(src_dir, dst_dir):
        files = os.listdir(src_dir)
        for fname in files:
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            shutil.copyfile(src, dst)

    with st.spinner("Копирование данных в комбинированный датасет..."):
        copy_files(TRAIN_CATS_PATH, os.path.join(COMBINED_DATASET_PATH, 'cats'))
        copy_files(TEST_CATS_PATH, os.path.join(COMBINED_DATASET_PATH, 'cats'))
        copy_files(TRAIN_DOGS_PATH, os.path.join(COMBINED_DATASET_PATH, 'dogs'))
        copy_files(TEST_DOGS_PATH, os.path.join(COMBINED_DATASET_PATH, 'dogs'))
        time.sleep(1)
    st.success("Данные скопированы!")

def make_dataset(img_path):
    CLASS_LIST = sorted(os.listdir(img_path))
    CLASS_COUNT = len(CLASS_LIST)

    BASE_DIR = './dataset/'

    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)

    os.mkdir(BASE_DIR)

    train_dir = os.path.join(BASE_DIR, 'train')
    validation_dir = os.path.join(BASE_DIR, 'validation')
    test_dir = os.path.join(BASE_DIR, 'test')

    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)


    def create_dataset(
        img_path: str,         # Путь к файлам с изображениями классов
        new_path: str,         # Путь к папке с выборками
        class_name: str,       # Имя класса (оно же и имя папки)
        start_index: int,      # Стартовый индекс изображения, с которого начинаем подвыборку
        end_index: int         # Конечный индекс изображения, до которого создаем подвыборку

    ):

        # Полный путь к папке с изображениями класса
        src_path = os.path.join(img_path, class_name)
        # Полный путь к папке с новым датасетом класса
        dst_path = os.path.join(new_path, class_name)

        # Получение списка имен файлов с изображениями текущего класса
        class_files = os.listdir(src_path)

        # Создаем подпапку, используя путь
        os.mkdir(dst_path)

        # Перебираем элементы, отобранного списка с начального по конечный индекс
        for fname in class_files[start_index: end_index]:
            # Путь к файлу (источник)
            src = os.path.join(src_path, fname)
            # Новый путь расположения файла (назначение)
            dst = os.path.join(dst_path, fname)
            # Копируем файл из источника в новое место (назначение)
            shutil.copyfile(src, dst)


    with st.spinner("Создание тренировочной, валидационной и тестовой выборок..."):
        for class_label in range(CLASS_COUNT):    
            class_name = CLASS_LIST[class_label]
            create_dataset(img_path, train_dir, class_name, 0, 3000)
            create_dataset(img_path, validation_dir, class_name, 3000, 4000)
            create_dataset(img_path, test_dir, class_name, 4000, 5000)
        time.sleep(1)
    st.success("Выборки созданы")

    return train_dir, validation_dir, test_dir

def create_gen(train_dir, validation_dir):

    train_datagen = ImageDataGenerator(
        rescale=1./255,           # нормализация данных
        rotation_range=40,        # поворот 40 градусов
        width_shift_range=0.2,    # смещенние изображения по горизонтали
        height_shift_range=0.2,   # смещенние изображения по вертикали
        shear_range=0.2,          # случайный сдвиг
        zoom_range=0.2,           # случайное масштабирование
        horizontal_flip=True,     # отражение по горизонтали
        fill_mode='nearest'       # стратегия заполнения пустых пикселей при трансформации
    )

    datagen = ImageDataGenerator(rescale=1./255)

    with st.spinner("Подготовка генераторов данных..."):
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            batch_size=20,
            class_mode='categorical'
        )
        validation_generator = datagen.flow_from_directory(
            validation_dir,
            target_size=(IMG_WIDTH, IMG_HEIGHT),
            batch_size=20,
            class_mode='categorical'
        )
        time.sleep(1)
    st.success("Генераторы данных подготовлены!")

    return train_generator, validation_generator, datagen

def model_maker(end_index):
        base_model = MobileNet(include_top=False, input_shape = (IMG_WIDTH, IMG_HEIGHT, 3))
        base_model.trainable = True

        for layer in base_model.layers[:end_index]:
            layer.trainable = False

        input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        custom_model = base_model(input)
        custom_model = GlobalAveragePooling2D()(custom_model)
        custom_model = Dense(64, activation='relu')(custom_model)
        custom_model = Dropout(0.5)(custom_model)
        predictions = Dense(2, activation='softmax')(custom_model)

        return Model(inputs=input, outputs=predictions)

def get_model(train_generator, validation_generator):
    st.write(f"Эпох: {user_epochs}")

    model = model_maker(end_index=50)  # Используем вашу функцию создания модели
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    save_best = ModelCheckpoint(filepath="./best_weights.keras",
                            monitor="val_accuracy",
                            save_best_only=True,
                            mode="auto",
                            verbose=1)

    class TrainingProgressCallback(Callback):
        def __init__(self, progress_bar):
            self.progress_bar = progress_bar

        def on_epoch_end(self, epoch, logs=None):
            # Обновление прогресса на каждой эпохе
            progress = (epoch + 1) / self.params['epochs']
            self.progress_bar.progress(progress)
    
    # Настройка индикатора прогресса
    training_progress = st.progress(0)

    # Настройка колбэка для обновления прогресса
    progress_callback = TrainingProgressCallback(training_progress)

    with st.spinner("Обучение модели..."):
        model.fit(
            train_generator,
            epochs=user_epochs,
            validation_data=validation_generator,
            callbacks=[save_best, progress_callback]
        )

    model.save('final_model.h5')

    st.success("Обучение завершено!")

    return model

if st.button('Обучить модель'):
    IMAGE_PATH = './temp/combined_dataset/'
    get_data()
    copy()
    train_dir, validation_dir, test_dir = make_dataset(IMAGE_PATH)
    train_generator, validation_generator, datagen = create_gen(train_dir, validation_dir)

    st.success("Данные успешно подготовлены и готовы к обучению!")

    model = get_model(train_generator, validation_generator)


    model.load_weights('best_weights.keras')

    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=20,
        class_mode='categorical'
    )

    test_loss, test_acc = model.evaluate(test_generator, steps=50)
    st.write(f'Точность на контрольной выборке: {test_acc * 100:.0f}%')
    with st.spinner("Классифицируем файл..."):
        if uploaded_file is not None:
    # Открываем загруженный файл с помощью PIL
            img = Image.open(uploaded_file)

            # Меняем размер изображения
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))

            # Преобразуем изображение в массив NumPy
            img_array = np.array(img)

            # Нормализуем пиксели (делим на 255)
            img_array = img_array.astype('float32') / 255.0

            # Расширяем размерность, чтобы создать batch (добавляем 1 дополнительную размерность)
            img_array = np.expand_dims(img_array, axis=0)

            # st.image(img, caption="Загруженное изображение", use_column_width=True)

            prediction = model.predict(img_array)

            class_index = np.argmax(prediction)

            if class_index == 0:
                st.success("Это кошка")
            else:
                st.success("Это собака")
