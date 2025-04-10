import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

training_csv = ("C:/Users/Asus/Desktop/Licenta/csvs/UNSW_NB15_training-set.csv")
testing_csv = ("C:/Users/Asus/Desktop/Licenta/csvs/UNSW_NB15_testing-set.csv")

training_set = pd.read_csv(training_csv)
testing_set = pd.read_csv(testing_csv)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(testing_set.drop(columns=['label', 'id', 'proto', 'service', 'state', 'attack_cat']))

#try to use PCA to reduce dimensions and plot the dataset
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

plt.scatter(df_pca[:,0], df_pca[:, 1], cmap = "coolwarm", alpha = 0.5)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection of Multi-Dimensional Data")
plt.show()