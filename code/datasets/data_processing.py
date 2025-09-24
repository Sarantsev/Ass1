import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def save_data_splits(data_dir):
    data = load_breast_cancer()
    X = data.data
    y = data.target

    raw_df = pd.DataFrame(X, columns=data.feature_names)
    raw_df['target'] = y
    raw_df.to_csv(os.path.join(data_dir, 'raw.csv'), index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_df = pd.DataFrame(X_train, columns=data.feature_names)
    train_df['target'] = y_train
    train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)

    test_df = pd.DataFrame(X_test, columns=data.feature_names)
    test_df['target'] = y_test
    test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)

if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
    os.makedirs(data_dir, exist_ok=True)
    save_data_splits(data_dir)