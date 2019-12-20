import numpy as np
from tensorflow import keras


# Base model
def baseModel(TotalData, TotalLabels):
    totalLen = len(TotalData)
    trainLen = int(totalLen*(2/3))

    trainData = TotalData[0:trainLen]
    testData = TotalData[trainLen: totalLen]

    trainLabels = TotalLabels[0:trainLen]
    testLabels = TotalLabels[trainLen: totalLen]

    train = np.zeros((len(trainData), 28, 28))
    test = np.zeros((len(testData), 28, 28))

    for i in range(len(trainData)):
        train[i, :, :] = trainData[i]

    for i in range(len(testData)):
        test[i, :, :] = testData[i]

    labels = np.zeros((len(trainLabels),))
    for i in range(len(trainLabels)):
        labels[i] = trainLabels[i]

    testlabelsF = np.zeros((len(testLabels),))
    for i in range(len(testLabels)):
        testlabelsF[i] = testLabels[i]

    labels = labels.astype("uint8")
    testlabelsF = testlabelsF.astype("uint8")

    # creating a model
    # first we need to flatten the data
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(29, activation="softmax")
    ])

    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train, labels, epochs=25)
    _, acc = model.evaluate(test, testlabelsF)
    print(acc)
