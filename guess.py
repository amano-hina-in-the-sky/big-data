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
D23 = []
CCC = 0
C = 120
for row in data_lists:
    gg = 0
    if row[1] == 0:
        gg = gg
    elif row[2] - row[1] > C:
        gg += 16
    elif row[1] - row[2] > C:
        gg += 32
    else:
        gg += 48

    if row[3] - row[2] > C:
        gg += 4
    elif row[2] - row[3] > C:
        gg += 8
    else:
        gg += 12

    if row[4] == 0:
        gg = gg
    elif row[3] - row[4] > C:
        gg += 1
    elif row[4] - row[3] > C:
        gg += 2
    else:
        gg += 3

    labels.append(gg)

plt.figure(figsize=(10, 8))
plt.scatter(D1, D2, alpha=0.6, s = 1)
plt.title('D1 D2')
plt.xlabel('D1')
plt.ylabel('D2')
plt.grid(True)
plt.savefig('dimension_plot.png')
print(C1, C2, C3)
print(g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15)
diff.sort()
diff2.sort()
H.sort()
F.sort()
D23.sort()
#print(diff)
#print(diff2)
#print(H)
#print(F)
#print(D23)
print(CCC)
df2 = pd.DataFrame({
    "id": range(1, Num + 1),
    "label": labels
})
df2.to_csv("public_submission.csv", index=False)



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

