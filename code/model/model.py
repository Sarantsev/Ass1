from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
import pandas as pd
from datasets.data_processing import save_data_splits

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
os.makedirs(data_dir, exist_ok=True)
save_data_splits(data_dir)

train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

X_train = train_df.drop('target', axis=1).values
y_train = train_df['target'].values
X_test = test_df.drop('target', axis=1).values
y_test = test_df['target'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../deployment/api/models'))
os.makedirs(output_dir, exist_ok=True)
joblib.dump(model, os.path.join(output_dir, 'model.pkl'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
