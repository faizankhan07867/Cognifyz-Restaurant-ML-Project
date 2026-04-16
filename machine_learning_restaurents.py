# ==========================================================
# 🍽 COGNIFYZ RESTAURANT ML PROJECT
# Full Python Code (Task 1 + Task 2 + Task 3)
# Author: Faizan Khan
# ==========================================================

# ----------------------------
# Import Libraries
# ----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("restaurants.csv")

# Clean Column Names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("✅ Dataset Loaded Successfully:", df.shape)

# ==========================================================
# 🧩 TASK 1 : Predict Restaurant Ratings
# ==========================================================

print("\n🧩 TASK 1 : Predict Restaurant Ratings")

# Features
X = df[['average_cost_for_two', 'price_range', 'votes']]

# Target
y = df['aggregate_rating']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("MAE :", round(mean_absolute_error(y_test, y_pred), 3))
print("R2 Score :", round(r2_score(y_test, y_pred), 3))

# ==========================================================
# 🧩 TASK 2 : Restaurant Recommendation System
# ==========================================================

print("\n🧩 TASK 2 : Restaurant Recommendation System")

# Combine Features
df['combined_features'] = df['cuisines'].astype(str) + " " + df['city'].astype(str)

# Convert Text to Matrix
cv = CountVectorizer()
matrix = cv.fit_transform(df['combined_features'])

# Similarity Matrix
similarity = cosine_similarity(matrix)

# Recommendation Function
def recommend_restaurant(name):
    name = name.lower()

    if name not in df['restaurant_name'].str.lower().values:
        print("❌ Restaurant Not Found")
        return

    index = df[df['restaurant_name'].str.lower() == name].index[0]

    scores = list(enumerate(similarity[index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    print(f"\n🍽 Recommended Restaurants for {name.title()}:\n")

    for i in sorted_scores:
        print(df.iloc[i[0]]['restaurant_name'])

# Example Recommendation
recommend_restaurant("domino's pizza")

# ==========================================================
# 🧩 TASK 3 : Restaurant Segmentation
# ==========================================================

print("\n🧩 TASK 3 : Restaurant Segmentation")

# Select Features
features = df[['average_cost_for_two', 'aggregate_rating', 'votes']]

# Scale Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(scaled_data)

print("✅ Clustering Completed")

# ==========================================================
# 📊 Dashboard Charts
# ==========================================================

plt.style.use("ggplot")

fig = plt.figure(figsize=(20,12))
fig.suptitle("🍽 Restaurant Analytics Dashboard", fontsize=18, fontweight='bold')

# 1 Scatter Plot
plt.subplot(2,3,1)
plt.scatter(
    df['average_cost_for_two'],
    df['aggregate_rating'],
    c=df['cluster'],
    cmap='viridis',
    s=60,
    alpha=0.7
)
plt.title("1. Restaurant Clusters")
plt.xlabel("Cost for Two")
plt.ylabel("Rating")

# 2 Bar Chart
plt.subplot(2,3,2)
df['cluster'].value_counts().plot(
    kind='bar',
    color=['#ff6b6b','#4ecdc4','#1a535c']
)
plt.title("2. Cluster Count")
plt.xlabel("Cluster")
plt.ylabel("Restaurants")

# 3 Pie Chart
plt.subplot(2,3,3)
df['cluster'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=90,
    shadow=True,
    colors=['#ff9f1c','#2ec4b6','#e71d36']
)
plt.title("3. Cluster Distribution")
plt.ylabel("")

# 4 Histogram
plt.subplot(2,3,4)
plt.hist(
    df['aggregate_rating'],
    bins=12,
    color='#4361ee',
    edgecolor='black'
)
plt.title("4. Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")

# 5 Average Rating per Cluster
plt.subplot(2,3,5)
df.groupby('cluster')['aggregate_rating'].mean().plot(
    kind='bar',
    color=['#8338ec','#3a86ff','#ff006e']
)
plt.title("5. Avg Rating per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Rating")

# 6 Average Cost per Cluster
plt.subplot(2,3,6)
df.groupby('cluster')['average_cost_for_two'].mean().plot(
    kind='bar',
    color=['#06d6a0','#118ab2','#ef476f']
)
plt.title("6. Avg Cost per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Cost")

plt.tight_layout()
plt.show()

# ==========================================================
# Project Completed
# ==========================================================

print("\n🎉 ALL TASKS COMPLETED SUCCESSFULLY")
