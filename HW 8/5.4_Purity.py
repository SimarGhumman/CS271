from math import log

# Function to calculate the entropy of a single cluster using probabilities of classes
def calculate_entropy_natural_log(cluster, total_points):
    entropy = -sum((count / total_points) * log(count / total_points) for count in cluster if count > 0)
    return entropy

# Function to calculate the purity of the entire clustering
def calculate_purity(clusters):
    total_points = sum(sum(cluster.values()) for cluster in clusters)
    # Purity is the sum of the maximum class count in each cluster divided by the total number of points
    return sum(max(cluster.values()) for cluster in clusters) / total_points

# Function to convert cluster data to the format needed for entropy calculation
def convert_clusters_to_probability(clusters):
    total_points = sum(sum(cluster.values()) for cluster in clusters)
    return [[count / total_points for count in cluster.values()] for cluster in clusters]

# Data for the left and right image clusters
left_clusters = [
    {'brown': 6, 'red': 1, 'blue': 1},  # Top cluster
    {'brown': 1, 'red': 1, 'blue': 3},  # Bottom cluster
    {'brown': 1, 'red': 7, 'blue': 1}   # Right cluster
]

right_clusters = [
    {'brown': 3, 'red': 3, 'blue': 2},  # Top cluster
    {'blue': 1, 'brown': 2, 'red': 2},  # Bottom cluster
    {'red': 4, 'brown': 3, 'blue': 2}   # Right cluster
]

# Convert cluster data to probabilities
left_probabilities = convert_clusters_to_probability(left_clusters)
right_probabilities = convert_clusters_to_probability(right_clusters)

# Calculate entropy for the left and right image clusters
left_entropies = [calculate_entropy_natural_log(cluster, sum(cluster)) for cluster in left_probabilities]
right_entropies = [calculate_entropy_natural_log(cluster, sum(cluster)) for cluster in right_probabilities]

# Calculate average entropy and purity for the left and right images
left_avg_entropy = sum(left_entropies) / len(left_entropies)
left_purity = calculate_purity(left_clusters)

right_avg_entropy = sum(right_entropies) / len(right_entropies)
right_purity = calculate_purity(right_clusters)

# Print the results
print(f"Left image - Average Entropy: {left_avg_entropy}, Purity: {left_purity}")
print(f"Right image - Average Entropy: {right_avg_entropy}, Purity: {right_purity}")
