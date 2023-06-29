# import numpy as np
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import LabelEncoder, StandardScaler

# # Assuming you have a dataset in a CSV file called 'data.csv'
# data = pd.read_csv('Wholesale_customers_data.csv')

# # Extract the features you want to use for clustering
# features = ['Channel', 'Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
# data = data[features]

# # Preprocess categorical variables (channel and region) using label encoding
# categorical_features = ['Channel', 'Region']
# label_encoder = LabelEncoder()
# for feature in categorical_features:
#     data[feature] = label_encoder.fit_transform(data[feature])

# # Perform feature scaling on numerical variables
# numerical_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
# scaler = StandardScaler()
# data[numerical_features] = scaler.fit_transform(data[numerical_features])

# # Specify the number of clusters you want to create
# num_clusters = 3

# # Create an instance of the KMeans algorithm
# kmeans = KMeans(n_clusters=num_clusters)

# # Fit the algorithm to your preprocessed data
# kmeans.fit(data)

# # Obtain the cluster labels for each data point
# labels = kmeans.labels_

# # Assign cluster labels to the original dataset
# data['Cluster'] = labels

# # Calculate the average values of each feature for each cluster
# cluster_means = data.groupby('Cluster').mean()

# # Identify the cluster(s) with high average values for certain features
# high_value_clusters = cluster_means[
#     (cluster_means['Grocery'] > 0.5) & (cluster_means['Detergents_Paper'] > 0.5)
# ]

# # Print the market opportunities based on the identified clusters
# for cluster in high_value_clusters.index:
#     print(f"Cluster {cluster}: Potential market opportunity for grocery and detergent papers")

# # Optional: You can also save the cluster means to a CSV file for further analysis
# cluster_means.to_csv('cluster_means.csv')





# # Path: marketOpp.py
# # import numpy as np
# # import pandas as pd
# # from sklearn.cluster import KMeans
# # from sklearn.preprocessing import StandardScaler

# # Assuming you have a dataset in a CSV file called 'data.csv'
# data = pd.read_csv('Wholesale_customers_data.csv')

# # Extract the features you want to use for clustering
# features = ['Channel', 'Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
# data = data[features]

# # Preprocess categorical variables (channel and region) using label encoding
# categorical_features = ['Channel', 'Region']
# label_encoder = LabelEncoder()
# for feature in categorical_features:
#     data[feature] = label_encoder.fit_transform(data[feature])

# # Perform feature scaling on numerical variables
# numerical_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
# scaler = StandardScaler()
# data[numerical_features] = scaler.fit_transform(data[numerical_features])

# # Specify the number of clusters you want to create
# num_clusters = 5

# # Create an instance of the KMeans algorithm
# kmeans = KMeans(n_clusters=num_clusters)

# # Fit the algorithm to your preprocessed data
# kmeans.fit(data)

# # Obtain the cluster labels for each data point
# labels = kmeans.labels_

# # Convert labels to a pandas Series
# labels_series = pd.Series(labels, name='Cluster')

# # Assign cluster labels to the original dataset
# data = pd.concat([data, labels_series], axis=1)

# # Calculate the average values of each feature for each cluster
# cluster_means = data.groupby('Cluster').mean()

# # Sort the market opportunities in descending order for each feature
# sorted_market_opportunities = cluster_means.apply(lambda x: x.sort_values(ascending=False))

# # Print the sorted market opportunities for each feature
# for feature in sorted_market_opportunities.columns:
#     print(f"Market opportunities for {feature}:")
#     for cluster, value in sorted_market_opportunities[feature].items():
#         print(f"Cluster {cluster}: {value}")
#     print()

# # Optional: You can also save the cluster means to a CSV file for further analysis
# cluster_means.to_csv('cluster_means.csv')

# # # Identify the cluster(s) with high average values for certain features
# # high_value_clusters = cluster_means[
# #     (cluster_means['Grocery'] > 0.5) & (cluster_means['Detergents_Paper'] > 0.5)
# # ]

# # # Print the market opportunities based on the identified clusters
# # for cluster in high_value_clusters.index:
# #     print(f"Cluster {cluster}: Potential market opportunity for grocery and detergent papers")

# # # Optional: You can also save the cluster means to a CSV file for further analysis
# # cluster_means.to_csv('cluster_means.csv')



import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Assuming you have a dataset in a CSV file called 'data.csv'
data = pd.read_csv('Wholesale_customers_data.csv')

# Extract the features you want to use for clustering
features = ['Channel', 'Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
data_features = data[features]

# Preprocess categorical variables (Channel and Region) using label encoding
categorical_features = ['Channel', 'Region']
label_encoder = LabelEncoder()
for feature in categorical_features:
    if data_features[feature].isnull().sum() > 0:
        data_features[feature] = data_features[feature].fillna('Unknown')
    data_features[feature] = label_encoder.fit_transform(data_features[feature])


# Perform feature scaling on numerical variables
numerical_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
scaler = StandardScaler()
data_features[numerical_features] = scaler.fit_transform(data_features[numerical_features])

# Group the data by 'Channel' and 'Region'
grouped_data = data_features.groupby(['Channel', 'Region'])

# Initialize an empty DataFrame to store the top market opportunities
top_market_opportunities = pd.DataFrame()

# Iterate over each group and perform clustering
for group, group_data in grouped_data:
    # Extract the numerical features for the current group
    group_data_numeric = group_data[numerical_features]

    # Specify the number of clusters you want to create for each group
    num_clusters = len(numerical_features)  # Set the number of clusters to the total number of features

    # Create an instance of the KMeans algorithm
    kmeans = KMeans(n_clusters=num_clusters)

    # Fit the algorithm to the preprocessed data for the current group
    kmeans.fit(group_data_numeric)

    # Obtain the cluster labels for each data point in the current group
    labels = kmeans.labels_

    # Assign cluster labels to the original dataset for the current group
    group_data['Cluster'] = labels

    # Calculate the average values of each feature for each cluster in the current group
    cluster_means = group_data.groupby('Cluster').mean()

    # Sort the market opportunities in descending order for each feature in the current group
    sorted_market_opportunities = cluster_means.mean(axis=1).sort_values(ascending=False)

    # Get all the market opportunities for the current group
    top_market_opportunities_group = sorted_market_opportunities

    # Append the top market opportunities for the current group to the overall results
    top_market_opportunities = pd.concat([top_market_opportunities, top_market_opportunities_group.to_frame().T])

# Reset the index of 'top_market_opportunities'
top_market_opportunities.reset_index(drop=True, inplace=True)

# Add 'Channel' and 'Region' columns to the market opportunities DataFrame
top_market_opportunities[['Channel', 'Region']] = data_features[['Channel', 'Region']]

# Print the top market opportunities for each 'Channel' and 'Region' combination
for idx, row in top_market_opportunities.iterrows():
    channel = row['Channel']
    region = row['Region']
    if not pd.isnull(channel):
        channel = label_encoder.inverse_transform([int(channel)])[0]
    if not pd.isnull(region):
        region = label_encoder.inverse_transform([int(region)])[0]

    if pd.notnull(channel) and pd.notnull(region):
        print(f"Top market opportunities for Channel: {channel}, Region: {region}")
        top_features = row.drop(['Channel', 'Region'])  # Get all the features and their values

        feature_names = data_features.columns[~((data_features.columns == 'Channel') | (data_features.columns == 'Region'))]
        for feature, value in top_features.items():
            feature_name = feature_names[int(feature)] # type: ignore
            print(f"{str(feature_name)}: {value}")
        print()

# Save the top market opportunities to a CSV file
top_market_opportunities.to_csv('top_market_opportunities.csv', index=False)
