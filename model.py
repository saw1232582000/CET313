from lib import *
from dataPreprocessing import *


def buildModel(trainData: TrainData):
    early_stopping = callbacks.EarlyStopping(
        min_delta=0.001,  # minimium amount of change to count as an improvement
        patience=20,  # how many epochs to wait before stopping
        restore_best_weights=True,
    )
    # Initialising the NN
    annModel = Sequential()

    # layers
    annModel.add(
        Dense(units=16, kernel_initializer="uniform", activation="relu", input_dim=12)
    )
    annModel.add(Dense(units=8, kernel_initializer="uniform", activation="relu"))
    annModel.add(Dropout(0.25))
    annModel.add(Dense(units=4, kernel_initializer="uniform", activation="relu"))
    annModel.add(Dropout(0.5))
    annModel.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

    # Compiling the ANN
    annModel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the ANN
    history = annModel.fit(
        trainData.X_train,
        trainData.y_train,
        batch_size=32,
        epochs=500,
        callbacks=[early_stopping],
        validation_split=0.2,
    )

    val_accuracy = np.mean(history.history["val_accuracy"])
    print("\n%s: %.2f%%" % ("Accuracy percentage", val_accuracy * 100))

    history_df = pd.DataFrame(history.history)
    mplot.plot(history_df.loc[:, ["loss"]], "#18c6ed", label="Training loss")
    mplot.plot(history_df.loc[:, ["val_loss"]], "#c7e03a", label="Validation loss")
    mplot.title("Loss Rate")
    mplot.xlabel("Epochs")
    mplot.ylabel("Loss")
    mplot.legend(loc="best")
    mplot.show()

    history_df = pd.DataFrame(history.history)
    mplot.plot(history_df.loc[:, ["accuracy"]], "#18c6ed", label="Training accuracy")
    mplot.plot(
        history_df.loc[:, ["val_accuracy"]], "#c7e03a", label="Validation accuracy"
    )
    mplot.title("Accuracy Rate")
    mplot.xlabel("Epochs")
    mplot.ylabel("Accuracy")
    mplot.legend()
    mplot.show()
