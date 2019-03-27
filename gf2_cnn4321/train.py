import keras
from keras import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Activation, Dropout
import LasterDataExp.cnn4321.Integrate_raw_data as ird


def DLNet(input_size):
    model = Sequential()
    # encoder
    model.add(Conv2D(64, (3, 3), input_shape=(input_size,input_size,4)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.75))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(20, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()
    keras.utils.plot_model(model, to_file='output/model.png', show_shapes=True, show_layer_names=True, rankdir='TB')
    return model


if __name__ == '__main__':

    # # 自定义模型
    input_size=9
    bands = [4, 3, 2, 1]

    train_tiff = ird.Tiff(fea_src="output/train_pos_label.txt", bands=bands,size=input_size).get_beatch(100)
    validate_tiff = ird.Tiff(fea_src="output/validate_pos_label.txt",bands=bands,size=input_size).get_beatch(100)

    model=DLNet(input_size=input_size)
    checkpoint = ModelCheckpoint(filepath="output/four_bands_DLnet_model.h5", monitor='val_acc', mode='auto', save_best_only='True')
    tensorboard=TensorBoard(log_dir='output/logs/DLnet_%d' % input_size)
    model.fit_generator(train_tiff,steps_per_epoch=100,epochs=120,verbose=2,
                        validation_data=validate_tiff,validation_steps=20,callbacks=[checkpoint,tensorboard])#


