from lib import *
from dataPreprocessing import *


def testannModel(trainData: TrainData):
    
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
    # Predicting the test set results
    y_pred = annModel.predict(trainData.X_test)
    y_pred = y_pred > 0.5
    np.set_printoptions()

    # confusion matrix
    cmap1 = sbn.diverging_palette(275, 150, s=40, l=65, n=6)
    mplot.subplots(figsize=(12, 8))
    cf_matrix = confusion_matrix(trainData.y_test, y_pred)
    custom_palette = sbn.color_palette("viridis", as_cmap=True)
    sbn.heatmap(
        cf_matrix / np.sum(cf_matrix),
        cmap=custom_palette,
        annot=True,
        annot_kws={"size": 15},
    )
    mplot.show()
    print("------------------------------------\n")
    print(classification_report(trainData.y_test, y_pred))
    print("\n------------------------------------")
