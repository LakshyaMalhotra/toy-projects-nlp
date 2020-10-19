import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../data/IMDB_Dataset.csv")
    df.sentiment = df.sentiment.map({"positive": 1, "negative": 0})
    
    df["kfold"] = -1
    y = df.sentiment.values
    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f_, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f_
    df["id"] = range(1, len(df)+1)
    df.to_csv("../data/train_folds.csv", index=False)

    print(df.head(5))
    print(df.kfold.value_counts())