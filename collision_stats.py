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

accident_value_lookup = {
  "ROUTE": ["", "Interstate", "Other US Route", "State Route", "County Route", "Local Street", "Other", "Unknown"],
  "RUR_URB": ["", "Urban", "Rural", "Unknown"],
  "FUNC_SYS": ["", "Interstate", "Principal Arterial", "Minor Arterial", "Major Collector", "Minor Collector", "Local", "Unknown"],
  "RD_OWNER": ["", "State", "Local", "Federal", "Other", "Unknown"],
  "HARM_EV": ["", "Pedestrian", "Bicyclist", "Motorcycle", "Other Motor Vehicle", "Fixed Object", "Railroad Vehicle", "Animal", "Other", "Unknown"],
  "MAN_COLL": ["", "Head-On", "Rear-End", "Angle", "Sideswipe", "Other", "Unknown"],
  "RELJCT2": ["", "At Intersection", "Not at Intersection", "Unknown"],
  "TYP_INT": ["", "Four-Way Intersection", "T-Intersection", "Y-Intersection", "Traffic Circle", "Roundabout", "Unknown"],
  "REL_ROAD": ["", "On Roadway", "Entering Roadway", "Exiting Roadway", "Off Roadway", "Unknown"],
  "WRK_ZONE": ["", "No", "Yes", "Unknown"],
  "LGT_COND": ["", "Daylight", "Dark - Lighted", "Dark - Not Lighted", "Unknown"],
  "WEATHER": ["", "Clear", "Rain", "Sleet", "Snow", "Fog", "Severe Crosswinds", "Blowing Sand, Soil, Dirt", "Other", "Unknown"],
  "SCH_BUS": ["", "No", "Yes"]
}

# Filter out the columns we don't need
accident_df = accident_df.filter(items=accident_useful_columns.keys())

# Print out value counts
for col, label in accident_useful_columns.items():
  print(label)
  value_labels = accident_value_lookup[col]
  for i in range(len(value_labels)):
    print(f"{i}: {value_labels[i]}")
  print(accident_df[col].value_counts(normalize=True) * 100)

