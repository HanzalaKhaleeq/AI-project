import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB

# Load data
def load_data():
    train = pd.read_csv('data/isolet1+2+3+4.data', header=None)
    X = train.iloc[:, :-1].values
    y = train.iloc[:, -1].astype(int).values
    return X, y

X_train, y_train = load_data()
# Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
# Save model
with open('models/nb.pkl', 'wb') as f:
    pickle.dump(nb, f)
print("Naive Bayes model saved to models/nb.pkl")