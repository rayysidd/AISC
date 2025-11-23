import numpy as np

class ksfom:
    def __init__(self,num_features,num_clusters,learning_rate=0.5):
        self.num_features = num_features
        self.num_clusters = num_clusters
        self.lr = learning_rate
        # Initialize weights randomly (Cluster centers)
        self.weights = np.random.rand(num_clusters, num_features)

    def compute_dist(self,x,w):
        return np.sqrt(np.sum((x-w)**2))
    
    def train(self,data,epochs=100):

        for epoch in range(epochs):
            current_lr = self.lr * (1-epoch/epochs)

            for input_vector in data:
                distances = [self.compute_dist(input_vector,w) for w in self.weights]
                winner_index=np.argmin(distances)

                # W_new = W_old + LR * (Input - W_old)
                self.weights[winner_index]+= self.lr*(input_vector-self.weights[winner_index])

    def predict(self,data):
        results = []
        for vector in data:
            distances = [self.compute_dist(vector, w) for w in self.weights]
            cluster_id = np.argmin(distances)
            results.append(cluster_id + 1)
        return results
    
def normalize_data(data):
    # Min-Max Normalization to 0-1 range
    data = np.array(data, dtype=float)
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    return (data - min_vals) / (max_vals - min_vals)



# Batch A1: Countries (Renewable Energy)
# Raw Data: [Renewable%, Emissions, GDP, Efficiency, Population]
raw_data_A1 = [
    [55, 3.2, 45000, 85, 10],
    [20, 9.0, 30000, 70, 50],
    [10, 12.5, 15000, 40, 80],
    [75, 1.5, 50000, 90, 5],
    [25, 8.0, 20000, 60, 40],
    [65, 2.0, 48000, 88, 8]
]

# 1. Normalize
norm_data = normalize_data(raw_data_A1)

# 2. Initialize & Train (4 Clusters as per PS)
som = ksfom(num_features=5, num_clusters=4, learning_rate=0.5)
som.train(norm_data, epochs=100)

# 3. Output
clusters = som.predict(norm_data)
print("Country Clusters:", clusters)


# Batch B2: CSE Students (Skills)
# Raw Data: [CGPA, ProgScore, TechEvents, Internship, CommScore]
raw_data_B2 = [
    [9.2, 95, 5, 6, 8],
    [7.5, 80, 3, 4, 5],
    [6.0, 55, 1, 0, 4],
    [8.0, 70, 2, 3, 9],
    [5.8, 50, 1, 1, 6],
    [7.0, 65, 4, 5, 7]
]

norm_data = normalize_data(raw_data_B2)
# 4 Clusters as per PS
som = ksfom(num_features=5, num_clusters=4) 
som.train(norm_data)
print("Student Clusters assigned:", som.predict(norm_data))