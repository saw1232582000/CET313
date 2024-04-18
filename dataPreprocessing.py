from lib import *


class TrainData:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


def preprocessData(data):
    # assigning values to features as X and target as y
    X = data.drop(["DEATH_EVENT"], axis=1)
    y = data["DEATH_EVENT"]

    # Set up a standard scaler for the features
    col_names = list(X.columns)
    s_scaler = preprocessing.StandardScaler()
    X_df = s_scaler.fit_transform(X)
    X_df = pd.DataFrame(X_df, columns=col_names)
    print("-----------------------------------------\n")
    print(X_df.describe().T)
    print("\n-----------------------------------------")

    mplot.show()
    unique_categories = y.unique()
    palette = sbn.color_palette("tab20", len(unique_categories))
    mplot.figure(figsize=(20, 10))
    sbn.boxenplot(data=X_df, palette=palette)
    mplot.xticks(rotation=90)
    mplot.show()


def getTrainData(data) -> TrainData:
    # assigning values to features as X and target as y
    X = data.drop(["DEATH_EVENT"], axis=1)
    y = data["DEATH_EVENT"]

    # Set up a standard scaler for the features
    col_names = list(X.columns)
    s_scaler = preprocessing.StandardScaler()
    X_df = s_scaler.fit_transform(X)
    X_df = pd.DataFrame(X_df, columns=col_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.25, random_state=7
    )
    return TrainData(X_train, X_test, y_train, y_test)
