import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score

'''
df = pd.read_csv("public_data.csv", sep=",")

data_lists = df.values.tolist()

Num = len(data_lists)
N = len(data_lists[0]) - 1

l = 0
r = 5000
tmp = []
labels = []
for _ in range(200):
    print(_)
    mid = (l + r) / 2
    tmp = []
    labels = []
    WA = 0
    for i in range(Num):
        pos = -1
        cnt = 0
        for j in range(len(tmp)):
            dis = 0
            for k in range(N):
                dis += (tmp[j][k + 1] - data_lists[i][k + 1]) * (tmp[j][k + 1] - data_lists[i][k + 1])
            if dis < mid * mid:
                cnt = cnt + 1
                pos = j
        if cnt > 1:
            WA = 1
            break
        if cnt == 1:
            labels.append(j)
        else:
            labels.append(len(tmp))
            tmp.append(data_lists[i])
            if len(tmp) > 15:
                break
    if WA == 1:
        r = mid
    else:
        if len(tmp) > 15:
            l = mid
        else:
            r = mid
print(l, r)
print(len(labels))
tmp = []
labels = []
WA = 0
for i in range(Num):
    pos = -1
    for j in range(len(tmp)):
        dis = 0
        for k in range(N):
            dis += (tmp[j][k + 1] - data_lists[i][k + 1]) * (tmp[j][k + 1] - data_lists[i][k + 1])
        if dis < 10000 * 10000:
            pos = j
    if pos != -1:
        labels.append(j)
    else:
        labels.append(len(tmp))
        print(len(tmp))
        tmp.append(data_lists[i])

print(len(tmp))

df2 = pd.DataFrame({
    "id": range(1, Num + 1),
    "label": labels
})
df2.to_csv("public_submission.csv", index=False)
'''



# 0.9557
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
        if row[Det] <= 6:
            X = row[pos - 1] - row[pos + 1]

            if X > 50:
                g[10 * pos + 1] += 1
                labels.append(10 * pos + 1)
            elif X < -50:
                g[10 * pos + 2] += 1
                labels.append(10 * pos + 2)
            else:
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
        if row[N // 2 - 1] <= 6 and row[N // 2 + 2] <= 6:
            C1 = C1 + 1
        if row[N // 2 - 1] <= 6:
            g[1] += 1
            labels.append(1)
            C2 = C2 + 1
        elif row[N // 2 + 2] <= 6:
            g[2] += 1
            labels.append(2)
            C3 = C3 + 1
        else:

            #print(row)
            GG.append(row[N // 2 - 1] - row[N // 2 + 2])
            if row[N // 2 - 1] - row[N // 2 + 2] > 50:
                g[3] += 1
                labels.append(3)
            elif row[N // 2 - 1] - row[N // 2 + 2] < -50:
                g[4] += 1
                labels.append(4)
            else:
                g[5] += 1
                labels.append(5)


GG.sort()
print(GG)
print(g)
print(C1, C2, C3)


df2 = pd.DataFrame({
    "id": range(1, Num + 1),
    "label": labels
})
df2.to_csv("private_submission.csv", index=False)




'''
df = pd.read_csv("public_data.csv")
X = df.iloc[:, 1:]
print(X)
for i in range(len(X)):
    sum = 0
    for j in range(len(X[i])):
        sum += X[i][j]
    print(i, sum)

n = X.shape[1]
k = 4 * n - 3
k = 11

kmeans = KMeans(n_clusters=k, random_state=69)
kmeans.fit(X)

labels = kmeans.labels_

output_df = pd.DataFrame({
    "id": df.iloc[:, 0],
    "label": labels
})

output_df.to_csv("public_submission.csv", index=False)
'''

