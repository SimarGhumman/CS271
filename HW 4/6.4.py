from sklearn.tree import DecisionTreeClassifier

X_malware = [
    [120, 7, 32],
    [120, 7, 28],
    [120, 6, 34],
    [130, 5, 33],
    [100, 6, 35],
    [100, 5, 27],
    [100, 6, 32],
    [120, 6, 33],
    [100, 8, 32],
    [110, 6, 34]
]

X_benign = [
    [120, 4, 22],
    [130, 5, 23],
    [140, 5, 26],
    [100, 5, 21],
    [110, 6, 20],
    [140, 7, 20],
    [140, 3, 28],
    [100, 4, 21],
    [100, 4, 24],
    [120, 7, 25]
]

X = X_malware + X_benign
y = [1] * 10 + [0] * 10

A_indices = [0, 1, 2, 4, 9]
B_indices = [2, 4, 6, 8, 9]
C_indices = [0, 1, 5, 7, 9]

clf_A = DecisionTreeClassifier()
clf_A.fit([X[i][:2] for i in A_indices], [y[i] for i in A_indices])

clf_B = DecisionTreeClassifier()
clf_B.fit([[X[i][0], X[i][2]] for i in B_indices], [y[i] for i in B_indices])

clf_C = DecisionTreeClassifier()
clf_C.fit([[X[i][1], X[i][2]] for i in C_indices], [y[i] for i in C_indices])

samples = [
    [100, 7, 27],
    [130, 7, 28],
    [115, 4, 30],
    [105, 4, 35],
    [140, 6, 20]
]

predictions = []

for sample in samples:
    votes = []
    votes.append(clf_A.predict([sample[:2]])[0])
    votes.append(clf_B.predict([[sample[0], sample[2]]])[0])
    votes.append(clf_C.predict([[sample[1], sample[2]]])[0])

    predictions.append(1 if votes.count(1) > votes.count(0) else 0)

print(predictions)
