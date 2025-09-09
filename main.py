# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Step 2: Load Dataset
data = pd.read_csv("dataset/Mall_Customers.csv")

# Step 3: Clean the Data
data.drop("CustomerID", axis=1, inplace=True)
data["Gender"] = data["Gender"].map({"Male": 0, "Female": 1})

# Step 4: Scale the Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Step 5: Elbow Method to find optimal number of clusters
inertia = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(scaled_data)
    inertia.append(model.inertia_)

# Optional: Save elbow plot (optional for GitHub)
plt.figure(figsize=(6,4))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method For Optimal k")
plt.grid(True)
plt.tight_layout()
plt.savefig("images/elbow_plot.png")  # Optional
plt.show()

# Step 6: Apply KMeans Clustering (Assuming k=5 from elbow)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Step 7: Add Cluster Labels to Data
data["Cluster"] = clusters

# Step 8: Final Cluster Visualization
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=data["Annual Income (k$)"],
    y=data["Spending Score (1-100)"],
    hue=data["Cluster"],
    palette="Set1",
    s=100
)
plt.title("Customer Segmentation (5 Clusters)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.savefig("images/cluster_plot.png")  # Optional
plt.show()
