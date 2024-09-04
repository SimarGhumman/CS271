import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE

malware_samples = list(range(1, 41))
malware_hmm = [-2.5502, -2.4916, -2.4591, -2.3937, -2.5805, -2.4426, -2.5148, -2.4417, -2.4508, -12.4561,
               -2.5332, -2.4849, -2.6171, -2.5150, -14.4404, -2.5892, -2.4532, -2.4831, -2.5505, -2.4479,
               -2.4743, -2.5167, -2.5318, -2.3913, -2.6346, -2.5553, -2.5426, -2.3792, -13.5807, -2.5571,
               -2.5179, -2.5161, -2.6699, -2.4019, -2.4906, -2.5358, -12.5585, -2.3902, -2.5675, -2.4462]
malware_ssd = [0.458, 0.398, 0.381, 0.387, 0.412, 0.944, 0.989, 0.402, 0.490, 0.479,
               0.491, 0.488, 0.455, 0.432, 0.408, 0.425, 0.421, 1.397, 0.368, 0.426,
               0.383, 0.466, 0.458, 0.437, 0.436, 0.460, 0.477, 0.431, 1.408, 0.429,
               0.355, 0.398, 0.478, 0.397, 0.447, 0.929, 0.419, 0.478, 0.416, 0.400]
malware_ogs = [0.3097, 0.8671, 0.2878, 0.3369, 0.3344, 0.2908, 0.2814, 0.3266, 0.3223, 0.7914,
               0.4302, 0.3293, 0.8409, 0.3612, 0.2755, 0.3998, 0.3486, 0.3550, 0.3432, 0.3346,
               0.3506, 0.3861, 0.3588, 0.3407, 0.4723, 0.3527, 0.3269, 0.3136, 0.3260, 0.3729,
               0.3303, 0.3033, 0.3950, 0.3360, 0.3694, 0.9425, 0.3946, 0.3299, 0.3566, 0.3332]
malware_labels = ['malware'] * 40

benign_samples = list(range(1, 41))
benign_hmm = [-20.1718, -13.8231, -12.2302, -23.7316, -9.4449, -33.5896, -148.4577, -11.9680, -8.0129, -14.7196,
              -12.9691, -35.6650, -14.8911, -33.0356, -14.0974, -12.8733, -16.8113, -30.8435, -9.0773, -22.3555,
              -21.6937, -14.2945, -21.9569, -27.6297, -19.0987, -22.5225, -31.2162, -148.9674, -42.8055, -51.2141,
              -21.3982, -17.8242, -169.1587, -45.4216, -22.7345, -15.0389, -13.6486, -14.1127, -15.7107, -33.7041]
benign_ssd = [0.930, 0.854, 0.928, 0.924, 0.801, 0.917, 0.908, 0.916, 0.930, 0.979,
              0.927, 0.882, 0.972, 0.865, 0.827, 0.953, 0.870, 0.915, 0.938, 0.848,
              0.858, 0.906, 0.999, 0.863, 0.927, 0.940, 0.908, 0.993, 1.036, 0.949,
              1.002, 0.992, 1.018, 1.059, 0.858, 0.857, 0.851, 0.848, 0.875, 1.037]
benign_ogs = [0.6909, 0.7998, 0.7324, 0.7543, 0.6843, 0.7021, 0.8879, 0.7166, 0.6830, 0.7142,
              0.6771, 0.6901, 0.8415, 0.7811, 0.6921, 0.7454, 0.6873, 0.8512, 0.7999, 0.7783,
              0.7068, 0.6834, 0.7130, 0.7892, 0.7036, 0.7543, 0.7014, 0.8182, 0.9166, 0.7801,
              0.8206, 0.8215, 0.9209, 0.8323, 0.7568, 0.6616, 0.6433, 0.6434, 0.6644, 0.7858]
benign_labels = ['benign'] * 40

malware_df = pd.DataFrame({
    'Sample': malware_samples,
    'HMM': malware_hmm,
    'SSD': malware_ssd,
    'OGS': malware_ogs,
    'Label': malware_labels
})

benign_df = pd.DataFrame({
    'Sample': benign_samples,
    'HMM': benign_hmm,
    'SSD': benign_ssd,
    'OGS': benign_ogs,
    'Label': benign_labels
})

df_alternating = pd.concat([malware_df.iloc[:20], benign_df.iloc[:20], malware_df.iloc[20:], benign_df.iloc[20:]], ignore_index=True)

train_data = df_alternating.iloc[:40]
test_data = df_alternating.iloc[40:]

clf = SVC(kernel='linear')
clf.fit(train_data[['HMM', 'SSD', 'OGS']], train_data['Label'])

predictions = clf.predict(test_data[['HMM', 'SSD', 'OGS']])
accuracy = accuracy_score(test_data['Label'], predictions)

weights = clf.coef_[0]

print("Accuracy:", accuracy)
print("Weights for HMM, SSD, OGS respectively:", weights)

features = ['HMM', 'SSD', 'OGS']

while len(features) > 1:
    clf = SVC(kernel='linear')

    selector = RFE(clf, n_features_to_select=len(features) - 1, step=1)
    selector.fit(train_data[features], train_data['Label'])

    clf.fit(train_data[features], train_data['Label'])
    feature_weights = clf.coef_[0]
    feature_with_min_weight = features[abs(feature_weights).argmin()]

    predictions = clf.predict(test_data[features])
    accuracy = accuracy_score(test_data['Label'], predictions)

    print(f"Features used: {features}")
    print(f"Weights: {feature_weights}")
    print(f"Accuracy: {accuracy}\n")

    features.remove(feature_with_min_weight)

clf = SVC(kernel='linear')
clf.fit(train_data[features], train_data['Label'])
feature_weights = clf.coef_[0]
predictions = clf.predict(test_data[features])
accuracy = accuracy_score(test_data['Label'], predictions)
print(f"Features used: {features}")
print(f"Accuracy: {accuracy}")
print(f"Weights: {feature_weights}\n")
