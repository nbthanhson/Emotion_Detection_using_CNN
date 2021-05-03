from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras.models import Sequential

model = Sequential()

#Add the first Block
model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', activation = 'relu', input_shape = train_data.shape[1:]))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

#Add the second Block
model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

#Add the third Block
model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

#Add the four Block
model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

#Add the Flatten layer
model.add(Flatten())

#Add the ouput layer
model.add(Dense(units = 256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 7, activation = 'softmax'))

#Display model summary
model.summary()

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Early stop training
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1)

#Save the best model
best_model = ModelCheckpoint(filepath = 'Emotion_Detection_bestmodel.h5', monitor = 'val_loss', verbose = 1, save_best_only = True)

#Reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 5, verbose = 1, min_lr = 0.0001)

#Start training
model_history = model.fit(
                        data_generator.flow(train_data, train_label, batch_size = 64),
                        epochs = 50,
                        validation_data = val_data,
                        callbacks = [early_stopping, best_model, reduce_lr],
                        shuffle = True)

#Load the best model
keras.models.load_model(filepath = 'Emotion_Detection_bestmodel.h5')

model.evaluate(test_x, test_y)
