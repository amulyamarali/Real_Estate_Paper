import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic data
n_samples = 1000
data = {
    'Size (sqft)': np.random.randint(800, 3000, n_samples),
    'Bedrooms': np.random.randint(1, 6, n_samples),
    'Bathrooms': np.random.uniform(1, 3, n_samples),
    'Neighborhood': np.random.choice(['Suburb', 'Urban', 'Rural'], n_samples),
    'SchoolRating': np.random.randint(1, 11, n_samples),
}

df = pd.DataFrame(data)

# Create a synthetic knowledge graph using NetworkX
G = nx.Graph()
G.add_node('Suburb', NodeType='Neighborhood', AvgHousePrice=250000)
G.add_node('Urban', NodeType='Neighborhood', AvgHousePrice=350000)
G.add_node('Rural', NodeType='Neighborhood', AvgHousePrice=200000)
G.add_edge('Suburb', 'Urban', Relationship='Adjacent')
G.add_edge('Urban', 'Rural', Relationship='Adjacent')

# show the complete knowledge graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=800, font_size=10, font_color='black')

plt.axis('off')
plt.show()


# Feature engineering using the knowledge graph
df['AvgHousePrice_Neighborhood'] = df['Neighborhood'].apply(
    lambda x: nx.get_node_attributes(G, 'AvgHousePrice')[x]
)

# Split data into features and target
X = df[['Size (sqft)', 'Bedrooms', 'Bathrooms', 'AvgHousePrice_Neighborhood']]
y = df['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Predict house prices
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
