import util
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def main():
    x_train, y_train = util.load_mnist()
    x_test, y_test = util.load_mnist(kind = "t10k")
    
    x_train = x_train.reshape(-1, 28, 28, 1) 
    x_test = x_test.reshape(-1, 28, 28, 1) 
    
    model = tf.keras.Sequential([
        layers.Conv2D(filters=8, kernel_size=5, padding="same", strides=2, activation="relu", input_shape=(28, 28, 1)),
        layers.Conv2D(filters=16, kernel_size=3, padding="same", strides=2, activation="relu", input_shape=(28, 28, 1)),
        layers.Conv2D(filters=32, kernel_size=3, padding="same", strides=2, activation="relu", input_shape=(28, 28, 1)), 
        layers.Conv2D(filters=32, kernel_size=3, padding="same", strides=2, activation="relu", input_shape=(28, 28, 1)),
        layers.AveragePooling2D(1),
        layers.Flatten(),
        layers.Dense(10, activation="relu"),
        layers.Softmax()
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))

    plt.figure()
    plt.plot(history.history['accuracy'], "-d", label="training accuracy")
    plt.plot(history.history['val_accuracy'], "-d", label="testing accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1])
    plt.title("Accuracy over epoch")
    plt.legend(loc='lower right')
    plt.savefig("output/CNN.png")
    plt.close()
    return

if __name__ == "__main__":
    main()