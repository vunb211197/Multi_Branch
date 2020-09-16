
from keras.models import Model, Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, DepthwiseConv2D, GlobalAveragePooling2D, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint

from make_data import make_data
import numpy as np


class AgeGenderRaceNetwork():

    def __init__(self, trainable):
        if trainable:
            self.build_model()
            self.train_model()
        else:
            self.model = load_model('models/my_model.h5')

    def build_model(self):
        input = Input(shape=(96, 96, 3))
        ageBranch = age_base_network(1, input)
        genderBranch = gender_base_network(1, input)
        raceBranch = race_base_network(5, input)

        self.model = Model(inputs=input, outputs=[ageBranch, genderBranch, raceBranch])

        self.model.summary()

    def train_model(self):
        losses = {"age_output": "mean_squared_error", "gender_output": "binary_crossentropy",
                  "race_output": "sparse_categorical_crossentropy"}
        lossWeights = {"age_output": 0.03,
                       "gender_output": 1.0, "race_output": 1.0}

        # lấy được tập train và tập val
        X_data, age_labels, gender_labels, race_labels = make_data()

        (X_train, X_val, age_train, age_val, gender_train, gender_val, race_train, race_val) = train_test_split(
            X_data, age_labels, gender_labels, race_labels, test_size=0.2, random_state=123)
        # tạo thêm data agumentation
        #tạm thời bỏ bước này vì tự nhiên thấy không quan trọng
        # image_aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1,
        #                                shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

        # khởi tạo các tham số để compile models
        EPOCHS = 50
        opt = optimizers.Adam(1e-4)
        losses = {"age_output": "mean_squared_error", "gender_output": "binary_crossentropy",
                  "race_output": "categorical_crossentropy"}
        lossWeights = {"age_output": 0.03,
                       "gender_output": 1.0, "race_output": 1.0}
        # thực hiện complile model
        self.model.compile(loss=losses, loss_weights=lossWeights,
                           optimizer=opt, metrics=['accuracy'])

        self.model.summary()

        # tiến hành train model
        filepath = "models/weights-{epoch:02d}-{val_accuracy:.2f}.h5"
        # Model Checkpoint
        cpt_save = ModelCheckpoint(
            filepath, save_best_only=True, monitor='val_accuracy', mode='max')
        callbacks_list = [cpt_save]
        
        print(X_train.shape)
        print(X_val.shape)
        print(age_train.shape)
        print(age_val.shape)
        print(gender_train.shape)
        print(gender_val.shape)
        print(race_train.shape)
        print(race_val.shape)

        print("Training......")
        self.model.fit(X_train, {"age_output": age_train, "gender_output": gender_train, "race_output": race_train}, validation_data=(X_val, {"age_output": age_val, "gender_output": gender_val, "race_output": race_val}), batch_size=32,epochs=EPOCHS, verbose=1,callbacks=callbacks_list)
        # lưu model sau khi đã trên xong
        self.model.save('models/my_model.h5')

    def predict(self, img):
        y_predict = self.model.predict(img.reshape(1, 96, 96, 3))
        print(y_predict[0])


def age_base_network(classes, input):
    # DepthWiseCONV => CONV => RELU => POOL
    x = DepthwiseConv2D(kernel_size=(3, 3), padding="same", activation='relu')(input)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), padding="same", activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding="same", activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (3, 3), padding="same", activation='relu')(x)
    x = BatchNormalization()(x)
    x = (MaxPooling2D(pool_size=(2, 2)))(x)

    x = Flatten()(x)
    x = Dense(classes, activation='linear', name="age_output")(x)

    print("1")
    print(x)

    return x


def gender_base_network(classes, input):
    # CONV => RELU => POOL
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(input)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)

    # CONV => RELU => POOL
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # CONV => RELU => POOL
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(classes, activation='sigmoid', name="gender_output")(x)

    print("2")
    print(x)
    return x


def race_base_network(classes, input):
    # CONV => RELU => POOL
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(input)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)

    # CONV => RELU => POOL
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # CONV => RELU => POOL
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(classes, activation='softmax', name="race_output")(x)

    print("3")
    print(x)

    return x
