#linear regression
# ---------------- GRAPHICAL REPRESENTATION ----------------
import matplotlib.pyplot as plt

# Graph 1: Actual vs Predicted Values
plt.figure(figsize=(8, 5))
plt.scatter(actual, predicted, alpha=0.7, label="Predictions")

low = min(min(actual), min(predicted))
high = max(max(actual), max(predicted))

plt.plot([low, high], [low, high], linestyle="--",
         label="Perfect Prediction")

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Graph 2: Prediction Errors
errors = [a - p for a, p in zip(actual, predicted)]

plt.figure(figsize=(8, 5))
plt.hist(errors, bins=20, edgecolor="black")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Distribution of Linear Regression Errors")
plt.grid(True)
plt.tight_layout()
plt.show()

# Install libraries before running:
# python -m pip install matplotlib numpy scipy








# logistic regression
# ---------------- GRAPHICAL REPRESENTATION ----------------
import matplotlib.pyplot as plt

# Graph 1: Confusion Matrix
labels = sorted(set(ytest + pred))

matrix = [
    [sum(1 for a, b in zip(ytest, pred)
         if a == actual_label and b == predicted_label)
     for predicted_label in labels]
    for actual_label in labels
]

plt.figure(figsize=(7, 5))
plt.imshow(matrix, interpolation="nearest")
plt.title("Logistic Regression: Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)
plt.colorbar()

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, str(matrix[i][j]),
                 ha="center", va="center")

plt.tight_layout()
plt.show()

# Graph 2: Actual vs Predicted Class Counts
actual_counts = [ytest.count(label) for label in labels]
predicted_counts = [pred.count(label) for label in labels]

positions = list(range(len(labels)))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar([p - width / 2 for p in positions],
        actual_counts, width, label="Actual")
plt.bar([p + width / 2 for p in positions],
        predicted_counts, width, label="Predicted")

plt.xticks(positions, labels)
plt.xlabel("Survival Class")
plt.ylabel("Number of Passengers")
plt.title("Titanic: Actual vs Predicted Class Counts")
plt.legend()
plt.tight_layout()
plt.show()

# Install libraries before running:
# python -m pip install matplotlib numpy scipy






#decision tree
# ---------------- GRAPHICAL REPRESENTATION ----------------
import matplotlib.pyplot as plt

labels = sorted(set(y + pred))

matrix = [
    [sum(1 for a, b in zip(y, pred)
         if a == actual_label and b == predicted_label)
     for predicted_label in labels]
    for actual_label in labels
]

# Graph 1: Confusion Matrix
plt.figure(figsize=(7, 5))
plt.imshow(matrix, interpolation="nearest")
plt.title("Decision Tree: Confusion Matrix")
plt.xlabel("Predicted Income Class")
plt.ylabel("Actual Income Class")
plt.xticks(range(len(labels)), labels, rotation=15)
plt.yticks(range(len(labels)), labels)
plt.colorbar()

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, str(matrix[i][j]),
                 ha="center", va="center")

plt.tight_layout()
plt.show()

# Graph 2: Actual vs Predicted Class Distribution
actual_counts = [y.count(label) for label in labels]
predicted_counts = [pred.count(label) for label in labels]
positions = list(range(len(labels)))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar([p - width / 2 for p in positions],
        actual_counts, width, label="Actual")
plt.bar([p + width / 2 for p in positions],
        predicted_counts, width, label="Predicted")

plt.xticks(positions, labels, rotation=15)
plt.ylabel("Number of Records")
plt.title("Decision Tree: Class Distribution")
plt.legend()
plt.tight_layout()
plt.show()

# Install libraries before running:
# python -m pip install matplotlib numpy scipy





#random forest
# ---------------- GRAPHICAL REPRESENTATION ----------------
import matplotlib.pyplot as plt

labels = sorted(set(y + pred))

matrix = [
    [sum(1 for a, b in zip(y, pred)
         if a == actual_label and b == predicted_label)
     for predicted_label in labels]
    for actual_label in labels
]

# Graph 1: Confusion Matrix
plt.figure(figsize=(7, 5))
plt.imshow(matrix, interpolation="nearest")
plt.title("Random Forest: Confusion Matrix")
plt.xlabel("Predicted Income Class")
plt.ylabel("Actual Income Class")
plt.xticks(range(len(labels)), labels, rotation=15)
plt.yticks(range(len(labels)), labels)
plt.colorbar()

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, str(matrix[i][j]),
                 ha="center", va="center")

plt.tight_layout()
plt.show()

# Graph 2: Actual vs Predicted Class Counts
actual_counts = [y.count(label) for label in labels]
predicted_counts = [pred.count(label) for label in labels]
positions = list(range(len(labels)))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar([p - width / 2 for p in positions],
        actual_counts, width, label="Actual")
plt.bar([p + width / 2 for p in positions],
        predicted_counts, width, label="Predicted")

plt.xticks(positions, labels, rotation=15)
plt.ylabel("Number of Records")
plt.title("Random Forest: Class Distribution")
plt.legend()
plt.tight_layout()
plt.show()

# Install libraries before running:
# python -m pip install matplotlib numpy scipy




#boosting
# ---------------- GRAPHICAL REPRESENTATION ----------------
import matplotlib.pyplot as plt

labels = ["<=50K", ">50K"]

matrix = [
    [sum(1 for a, b in zip(y, pred)
         if a == actual_label and b == predicted_label)
     for predicted_label in labels]
    for actual_label in labels
]

# Graph 1: Confusion Matrix
plt.figure(figsize=(7, 5))
plt.imshow(matrix, interpolation="nearest")
plt.title("AdaBoost-Style Boosting: Confusion Matrix")
plt.xlabel("Predicted Income Class")
plt.ylabel("Actual Income Class")
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)
plt.colorbar()

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, str(matrix[i][j]),
                 ha="center", va="center")

plt.tight_layout()
plt.show()

# Graph 2: Actual vs Predicted Class Counts
actual_counts = [y.count(label) for label in labels]
predicted_counts = [pred.count(label) for label in labels]
positions = list(range(len(labels)))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar([p - width / 2 for p in positions],
        actual_counts, width, label="Actual")
plt.bar([p + width / 2 for p in positions],
        predicted_counts, width, label="Predicted")

plt.xticks(positions, labels)
plt.ylabel("Number of Records")
plt.title("Boosting: Class Distribution")
plt.legend()
plt.tight_layout()
plt.show()

# Install libraries before running:
# python -m pip install matplotlib numpy scipy






# heirarchical clustering
# ---------------- GRAPHICAL REPRESENTATION ----------------
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Assign each row its final cluster number
cluster_ids = [0] * len(X)

for cluster_number, cluster in enumerate(clusters):
    for row_index in cluster:
        cluster_ids[row_index] = cluster_number

# Graph 1: Cluster Visualization
# Uses the first two standardized features for a 2D view.
plt.figure(figsize=(8, 6))
plt.scatter(
    [row[0] for row in X],
    [row[1] for row in X],
    c=cluster_ids,
    s=60
)

plt.xlabel("Standardized Fresh Spending")
plt.ylabel("Standardized Milk Spending")
plt.title("Wholesale Customers: Hierarchical Clusters")
plt.grid(True)
plt.tight_layout()
plt.show()

# Graph 2: Hierarchical Clustering Dendrogram
# Single linkage, consistent with the experiment.
linkage_matrix = linkage(X, method="single", metric="euclidean")

plt.figure(figsize=(12, 6))
dendrogram(
    linkage_matrix,
    truncate_mode="lastp",
    p=15,
    leaf_rotation=45,
    leaf_font_size=9
)

plt.xlabel("Cluster or Sample")
plt.ylabel("Euclidean Distance")
plt.title("Hierarchical Clustering Dendrogram (Single Linkage)")
plt.tight_layout()
plt.show()

# Install libraries before running:
# python -m pip install matplotlib numpy scipy


