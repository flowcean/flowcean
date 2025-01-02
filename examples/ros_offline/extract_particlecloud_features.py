import sys

import polars as pl
from features import extract_features_from_message

sys.setrecursionlimit(1000000)

data = pl.read_json("cached_ros_data.json")

particle_cloud = data[0, 3]

number_of_messages = len(particle_cloud)

all_features = []

for i in range(number_of_messages):
    message_dictionary = particle_cloud[i]

    message_time = message_dictionary["time"]
    particles_dict = message_dictionary["value"]
    list_of_particles = particles_dict["particles"]

    features = extract_features_from_message(
        list_of_particles, eps=0.3, min_samples=5
    )

    features["time"] = message_time
    all_features.append(features)

features_df = pl.DataFrame(all_features)
time_values = features_df["time"].to_list()

new_data = {}

for col in features_df.columns:
    if col == "time":
        continue

    # Extract the feature values
    feature_values = features_df[col].to_list()

    # Combine each feature value with its corresponding time into a dictionary
    dict_list = [
        {"time": t, "value": val}
        for t, val in zip(time_values, feature_values, strict=False)
    ]

    # Put the entire dict_list as a single entry - a list of all structs.
    new_data[col] = [dict_list]

final_df = pl.DataFrame(new_data)

print(final_df.head())
print(final_df.shape)
