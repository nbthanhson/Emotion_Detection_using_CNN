import matplotlib.pyplot as plt

# Get training loss and validation loss from model history
history_dict = model_history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

# Diplay a chart of training loss and validation loss
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training loss', 'Val loss'], loc='center right')

plt.show()

# Get training accuracy and validation loss from model history
history_dict = model_history.history
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

# Diplay a chart of training accuracy and validation accuracy
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy)
plt.plot(epochs, val_accuracy)

plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training accuracy', 'Val accuracy'], loc='center right')

plt.show()
