import pandas as pd
import pickle
from sklearn.cluster import KMeans

# Load data
def load_data():
    train = pd.read_csv('data/isolet1+2+3+4.data', header=None)
    X = train.iloc[:, :-1].values
    return X

X_train = load_data()
# Train K-Means
kmeans = KMeans(n_clusters=26, random_state=42)
kmeans.fit(X_train)
# Save model
with open('models/kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
print("K-Means model saved to models/kmeans.pkl")