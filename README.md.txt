# 🛍️ Customer Segmentation using K-Means Clustering

This project performs customer segmentation based on spending patterns and annual income using **K-Means Clustering**, a popular unsupervised machine learning algorithm.

---

## 📂 Dataset

- Source: [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial)
- Contains:
  - Gender
  - Age
  - Annual Income (k$)
  - Spending Score (1–100)

---

## 🧠 Objective

The goal is to **group customers into similar segments** so that businesses can target each group with appropriate marketing strategies.

---

## 🛠️ Tech Stack

| Tool | Use |
|------|-----|
| Python | Programming Language |
| Pandas | Data loading & processing |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | Machine Learning (KMeans, StandardScaler) |

---

## 🚀 Project Workflow

1. **Data Cleaning**: Removed customer ID, encoded gender
2. **Feature Scaling**: Standardized features using `StandardScaler`
3. **Elbow Method**: Chose optimal clusters (k=5)
4. **KMeans Clustering**: Grouped customers into 5 segments
5. **Visualization**: Displayed clusters based on income and spending score

---

## 📊 Output

### 📌 Final Cluster Plot:
![Customer Clusters](images/cluster_plot.png)

### 📌 Elbow Method (to find k):
![Elbow Method](images/elbow_plot.png)

---

## 📈 Cluster Insights

Each cluster represents a customer group such as:

- 💸 High Income, High Spending
- 🧑‍💼 Low Income, Low Spending
- 🎯 Balanced Customers

These segments can help businesses create targeted marketing campaigns.

---

## ▶️ How to Run

1. Clone the repository  
2. Install dependencies:  
