import pandas as pd

from kmodes.kmodes import KModes
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

# Read in the data
accident_df = pd.read_csv("data/accident.CSV", encoding="windows-1252")

case_id = "ST_CASE"

accident_df.set_index(case_id, inplace=True)

accident_useful_columns = {
  "ROUTE": "Route Signing",
  "RUR_URB": "Land Use",
  "FUNC_SYS": "Functional System",
  "RD_OWNER": "Ownership",
  "HARM_EV": "Harmful Event",
  "MAN_COLL": "Manner of Collision",
  "RELJCT2": "Relation to Junction",
  "TYP_INT": "Intersection Type",
  "REL_ROAD": "Relation to Trafficway",
  "WRK_ZONE": "Work Zone",
  "LGT_COND": "Light Condition",
  "WEATHER": "Weather",
  "SCH_BUS": "School Bus"
}

# Filter out the columns we don't need
accident_df = accident_df.filter(items=accident_useful_columns.keys())

# Create numpy matrix
accident_mat = accident_df.to_numpy()

# k-modes clustering
best_cost = float("inf")
best_clusters = []
best_km = None
for n_clusters in range(2, 10):
  km = KModes(n_clusters=n_clusters, init="Huang", n_jobs=16, verbose=1)
  clusters = km.fit_predict(accident_mat)
  if km.cost_ < best_cost:
    best_cost = km.cost_
    best_clusters = clusters
    best_km = km

print(f"Best init: {best_km.init}")
print(f"Best n: {best_km.n_clusters}")
print(f"Best cost: {best_cost}")
print(f"Best clusters: {best_clusters}")

# t-SNE visualization
plt.clf()
tsne = TSNE(n_components=2, learning_rate="auto", verbose=1)
z = tsne.fit_transform(accident_mat)
plt.figure(figsize=(20,20))
plt.margins(0)
plt.axis('off')
fig = plt.scatter(
  z[:,0], z[:,1],
  c=clusters,
  cmap='hsv',
  alpha=0.8,
  s=20,
  lw=0,
  edgecolor='white'
)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig("tsne.png", transparent=False)

print(clusters)

