import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score

df = pd.read_csv("private_data.csv", sep=",")

data_lists = df.values.tolist()
Num = len(data_lists)
N = len(data_lists[0]) - 1

labels = []
g = []
GG = []
C1 = 0
C2 = 0
C3 = 0
D1 = []
D2 = []
for i in range(10 * (N + 1)):
    g.append(0)

for row in data_lists:
    pos = -1
    sum = 0
    for i in range(N):
        sum += row[i + 1]
    for i in range(N):
        if row[i + 1] * 2 > sum:
            pos = i + 1
    if pos > 1 and pos < N:
        Det = 0

        if pos <= N // 2:
            Det = pos + 2
        else:
            Det = pos - 2
        if row[pos - 1] - row[pos + 1] < -30:
            g[10 * pos + 1] += 1
            labels.append(10 * pos + 1)
        elif row[pos - 1] - row[pos + 1] > 30:
            g[10 * pos + 2] += 1
            labels.append(10 * pos + 2)
        else:
            if row[Det] < 10:
                g[10 * pos + 3] += 1
                labels.append(10 * pos + 3)
            else:
                g[10 * pos + 4] += 1
                labels.append(10 * pos + 4)
    elif pos == 1:
        g[10 * 1 + 1] += 1
        labels.append(10 * 1 + 1)
    elif pos == N:
        g[10 * N + 1] += 1
        labels.append(10 * N + 1)
    else:
        if row[N // 2 - 1] - row[N // 2 + 2] > 70:
            if row[N // 2 + 2] <= 10:
                g[1] += 1
                labels.append(1)
            else:
                g[2] += 1
                labels.append(2)
        elif row[N // 2 + 2] - row[N // 2 - 1] > 70:
            if row[N // 2 - 1] <= 10:
                g[3] += 1
                labels.append(3)
            else:
                g[4] += 1
                labels.append(4)
        else:
            g[5] += 1
            labels.append(5)
'''
plt.figure(figsize=(10, 8))
plt.scatter(D1, D2, alpha=0.6, s = 1)
plt.title('D1 D2')
plt.xlabel('D1')
plt.ylabel('D2')
plt.grid(True)
plt.savefig('dimension_plot_2.png')
'''

print(g)


df2 = pd.DataFrame({
    "id": range(1, Num + 1),
    "label": labels
})
df2.to_csv("private_submission.csv", index=False)

