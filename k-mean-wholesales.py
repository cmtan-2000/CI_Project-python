
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

df = pd.read_csv('Wholesale_customers_data.csv')
df['Total'] = df['Fresh'] + df['Milk'] + df['Grocery'] + df['Frozen'] + df['Detergents_Paper'] + df['Delicassen']
# print (df.head())

def get_dummies(source_df, dest_df, col):
    dummies = pd.get_dummies(source_df[col], prefix=col)

    print ('Quantities for %s column' % col)
    for col in dummies:
        print ('%s: %d' % (col, np.sum(dummies[col])))
    print()

    dest_df = dest_df.join(dummies)
    return dest_df

df = get_dummies(df, df, 'Channel')
df.drop(['Channel', 'Channel_2'], axis=1, inplace=True)
df = get_dummies(df, df, 'Region')
df.drop(['Region', 'Region_3'], axis=1, inplace=True)
df.rename(index=str, columns={'Channel_1': 'Channel_Horeca', 'Region_1': 'Region_Lisbon', 'Region_2': 'Region_Oporto'}, inplace=True)
print (df.head())

plt.hist(df['Total'], bins=32)
plt.xlabel('Total Purchases')
plt.ylabel('Number of Customers')
plt.title('Histogram of Customer Size')
plt.show()
plt.close()

sc = StandardScaler()
sc.fit(df)
X = sc.transform(df)

def plot_kmeans(pred, centroids, x_name, y_name, x_idx, y_idx, k):

    for i in range(0, k):
        plt.scatter(df[x_name].loc[pred == i], df[y_name].loc[pred == i], s=6,c=colors[i], marker= markers[i], label='Cluster %d' % (i + 1)) # type: ignore

    centroids = sc.inverse_transform(kmeans.cluster_centers_) # type: ignore

    # plot_kmeans(pred, centroids, 'Frozen', 'Detergents_Paper', 3, 4, k)
    plt.scatter(centroids[:, x_idx], centroids[:, y_idx], marker='x', s=180, linewidths=3,color='k', zorder=10) # type: ignore

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()
    plt.show()
    plt.close()

markers = ('s', 'o', 'v', '*', 'D', '+', 'p', '<', '>', 'x')
colors = ('C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9')
    
k = 3
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
pred = kmeans.predict(X)

for i in range(0, k):
    x = len(pred[pred == i])
    print ('Cluster %d has %d members' % ((i + 1), x))
    centroids = sc.inverse_transform(kmeans.cluster_centers_)
    plot_kmeans(pred, centroids, 'Frozen', 'Detergents_Paper', 3, 4, k)

df = df.loc[df['Total'] <= 75000]

sc = StandardScaler()
sc.fit(df)
X = sc.transform(df)

K = range(1, 20)
mean_distortions = []
for k in K:
    np.random.seed(555)
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
    kmeans.fit(X)
    mean_distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis = 1))/ X.shape[0])

plt.plot(K, mean_distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting K w/ Elbow Method')
plt.show()
plt.close()

np.random.seed(555)  # sets seed, makes it repeatable
k = 6
kmeans = KMeans(n_clusters=k)  # , init='random')
kmeans.fit(X)
pred = kmeans.predict(X)

for i in range(0, k):
    x = len(pred[pred == i])
    print ('Cluster %d has %d members' % ((i + 1), x))

centroids = sc.inverse_transform(kmeans.cluster_centers_)
plot_kmeans(pred, centroids, 'Frozen', 'Detergents_Paper', 3, 4, k)


def print_cluster_data(cluster_number):
    print ('\nData for cluster %d' % cluster_number)
    cluster = df.loc[pred == cluster_number - 1, :]
    # print cluster1.head()
    num_in_cluster = float(len(cluster.index))
    num_horeca = float(np.sum(cluster['Channel_Horeca']))
    num_retail = float(num_in_cluster - num_horeca)
    print ('Percent Horeca: %.2f, Percent Retail: %.2f' % (num_horeca / num_in_cluster * 100.0, num_retail / num_in_cluster * 100.0))

    num_lisbon = float(np.sum(cluster['Region_Lisbon']))
    num_oporto = float(np.sum(cluster['Region_Oporto']))
    num_other = num_in_cluster - num_lisbon - num_oporto
    print ('Percent Lisbon: %.2f, Percent Oporto: %.2f, Percent Other: %.2f' % (num_lisbon / num_in_cluster * 100.0, num_oporto / num_in_cluster * 100.0, num_other / num_in_cluster * 100.0))

    avg_cust_size = np.sum(cluster['Total']) / num_in_cluster
    print ('Average Customer Size is: %.2f for %d Customers' % (avg_cust_size, num_in_cluster))

print_cluster_data(cluster_number=1)
print_cluster_data(cluster_number=3)
print_cluster_data(cluster_number=5)