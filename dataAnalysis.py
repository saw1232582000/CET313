from lib import *


def analysisData(data):

    cols = ["#18c6ed", "#c7e03a"]
    ####################################### Analyzing Data Balance #######################################
    sbn.countplot(
        x=data["DEATH_EVENT"], palette=cols, legend=False, hue=data["DEATH_EVENT"]
    ).set_title("Data Balance", color="#774571")
    mplot.legend([], frameon=False)
    mplot.show()

    ####################################### Analyzing corelation matrix #######################################
    
    cmap = sbn.diverging_palette(275, 150, s=40, l=65, n=9)
    corrmat = data.corr()
    mplot.subplots(figsize=(18, 18))
    sbn.heatmap(corrmat, cmap=cmap, annot=True, square=True)
    mplot.show()

    ####################################### Analyzing age distrivution #######################################

    mplot.figure(figsize=(20, 12))
    Days_of_week = sbn.countplot(
        x=data["age"], data=data, hue="DEATH_EVENT", palette=cols
    )
    Days_of_week.set_title("Age Distribution", color="#774571")
    mplot.show()

    ####################################### Analyzing swarmplot of non binary features #######################################
    
    feature = [
        "age",
        "creatinine_phosphokinase",
        "ejection_fraction",
        "platelets",
        "serum_creatinine",
        "serum_sodium",
        "time",
    ]
    for i in feature:
        mplot.figure(figsize=(8, 8))
        sbn.swarmplot(x=data["DEATH_EVENT"], y=data[i], color="red", alpha=0.5)
        sbn.boxenplot(x=data["DEATH_EVENT"], y=data[i], palette=cols,color='red')
        mplot.show()

    ####################################### Analyzing kdeplot of age and time #######################################
    sbn.kdeplot(x=data["time"], y=data["age"], hue=data["DEATH_EVENT"], palette=cols)
    data.describe().T
    mplot.show()
