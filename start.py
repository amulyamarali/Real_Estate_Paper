import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Create a bipartite graph
G = nx.Graph()

# Sample users and items
users = ["User1", "User2", "User3", "User4"]
items = ["ItemA", "ItemB", "ItemC", "ItemD", "ItemE"]

# Create a random utility matrix
utility_matrix = np.array([
    [5, 0, 4, 0, 0],
    [0, 3, 0, 2, 0],
    [0, 0, 0, 0, 4],
    [0, 0, 0, 0, 0]
])

# Add nodes for users and items
G.add_nodes_from(users, bipartite=0, label="User")
G.add_nodes_from(items, bipartite=1, label="Item")

# Define user-item interactions (edges) based on the utility matrix
user_item_interactions = []
for user_idx, user in enumerate(users):
    for item_idx, item in enumerate(items):
        if utility_matrix[user_idx, item_idx] > 0:
            user_item_interactions.append((user, item))

G.add_edges_from(user_item_interactions)

# Draw the bipartite graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=800, font_size=10, font_color='black')

# Separate and label users and items in the graph
user_pos = {node: (pos[node][0], pos[node][1] + 0.1) for node in users}
item_pos = {node: (pos[node][0], pos[node][1] - 0.1) for node in items}

user_labels = {user: user for user in users}
item_labels = {item: item for item in items}

nx.draw_networkx_labels(G, user_pos, labels=user_labels, font_size=12, font_color='blue')
nx.draw_networkx_labels(G, item_pos, labels=item_labels, font_size=12, font_color='green')

plt.axis('off')
plt.show()
