import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# 0.9775
df = pd.read_csv("public_data.csv", sep=",")

data_lists = df.values.tolist()

Num = len(data_lists)
N = len(data_lists[0]) - 1

labels = []
C1 = 0
C2 = 0
C3 = 0

g1 = 0
g2 = 0
g3 = 0
g4 = 0
g5 = 0
g6 = 0
g7 = 0
g8 = 0
g9 = 0
g10 = 0
g11 = 0
g12 = 0
g13 = 0
g14 = 0
g15 = 0
diff = []
diff2 = []
H = []
F = []
D1 = []
D2 = []
D3 = []
CCC = 0

for row in data_lists:
    if row[1] > row[2] + row[3] + row[4]:
        g1 = g1 + 1
        labels.append(1 * 4 + 1)

    elif row[2] > row[1] + row[3] + row[4]:
        if row[1] - row[3] < -30:
            g2 = g2 + 1
            labels.append(2 * 4 + 1)
        elif row[1] - row[3] > 30:
            g3 = g3 + 1
            labels.append(2 * 4 + 2)
        else:
            if row[4] < 10:
                g4 = g4 + 1
                labels.append(2 * 4 + 3)
            else:
                g5 = g5 + 1
                labels.append(2 * 4 + 4)

    elif row[3] > row[1] + row[2] + row[4]:
        if row[2] - row[4] < -30:
            g6 = g6 + 1
            labels.append(3 * 4 + 1)
        elif row[2] - row[4] > 30:
            g7 = g7 + 1
            labels.append(3 * 4 + 2)
        else:
            if row[1] < 10:
                g8 = g8 + 1
                labels.append(3 * 4 + 3)
            else:
                g9 = g9 + 1
                labels.append(3 * 4 + 4)

    elif row[4] > row[1] + row[2] + row[3]:
        g10 = g10 + 1
        labels.append(4 * 4 + 1)

    else:
        D1.append(row[1])
        D2.append(row[4])
        if row[1] - row[4] > 50:

            if row[4] <= 10:
                g11 = g11 + 1
                labels.append(5 * 4 + 1)
            else:
                g12 = g12 + 1
                labels.append(5 * 4 + 2)
        elif row[4] - row[1] > 50:
            if row[1] <= 10:
                g13 = g13 + 1
                labels.append(5 * 4 + 3)
            else:
                g14 = g14 + 1
                labels.append(5 * 4 + 4)
        else:
            g15 = g15 + 1
            labels.append(5 * 4 + 5)

            C3 += 1


print(g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15)
df2 = pd.DataFrame({
    "id": range(1, Num + 1),
    "label": labels
})
df2.to_csv("public_submission.csv", index=False)


'''
plt.figure(figsize=(10, 8))
plt.scatter(D1, D2, alpha=0.6, s = 1)
plt.title('D1 D2')
plt.xlabel('D1')
plt.ylabel('D2')
plt.grid(True)
plt.savefig('dimension_plot.png')
'''


# Some data
# (k = 15, random_state = 0) 0.8828
# (k = 15, random_state = 1) 0.8793
# (k = 15, random_state = 2) 0.8124
# (k = 15, random_state = 3) 0.8833
# (k = 15, random_state = 4) 0.8828
# (k = 15, random_state = 5) 0.8852
# (k = 15, random_state = 6) 0.8270

'''
df = pd.read_csv("public_data.csv")
X = df.iloc[:, 1:]

n = X.shape[1]
k = 15

kmeans = KMeans(n_clusters=k, random_state=6)
kmeans.fit(X)

labels = kmeans.labels_

output_df = pd.DataFrame({
    "id": df.iloc[:, 0],
    "label": labels
})

output_df.to_csv("public_submission.csv", index=False)
'''
