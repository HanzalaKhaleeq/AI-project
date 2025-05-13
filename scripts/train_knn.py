import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier

# Load data
def load_data():
    train = pd.read_csv('data/isolet1+2+3+4.data', header=None)
    X = train.iloc[:, :-1].values
    y = train.iloc[:, -1].astype(int).values
    return X, y

X_train, y_train = load_data()

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Save model
with open('models/knn.pkl', 'wb') as f:
    pickle.dump(knn, f)

print("KNN model saved to models/knn.pkl")