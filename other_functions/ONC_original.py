import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


def optimal_number_clusters(corr_mtrx: pd.DataFrame) -> list[list[str]]:
    clustered_corr_matrix, clusters_dict, silh = clusterKMeansTop(corr_mtrx)
    listed_clusters = []
    # clustered_corr_matrix.to_csv('clustered_correlation_matrix.csv')
    for key in clusters_dict:
        listed_clusters.append(clusters_dict[key])
    return listed_clusters


def clusterKMeansBase(corr0, maxNumClusters=10, n_init=10):
    print(f'Runnning clusterKMeansBase with maxNumClusters={maxNumClusters} and n_init={n_init}.')
    dist, silh = ((1 - corr0.fillna(0)) / 2.) ** .5, pd.Series()  # distance matrix
    for init in range(n_init):
        for i in range(2, maxNumClusters + 1):  # find optimal num clusters
            kmeans_ = KMeans(n_clusters=i, n_init=1)
            kmeans_ = kmeans_.fit(dist)
            silh_ = silhouette_samples(dist, kmeans_.labels_)
            stat = (silh_.mean() / silh_.std(), silh.mean() / silh.std())
            if np.isnan(stat[1]) or stat[0] > stat[1]:
                silh, kmeans = silh_, kmeans_
            if i % 50 == 0:
                print(f'{init + 1} of {n_init}: Iteration {i} done...')
    # n_clusters = len(np.unique(kmeans.labels_))
    newIdx = np.argsort(kmeans.labels_)
    corr1 = corr0.iloc[newIdx]  # reorder rows
    corr1 = corr1.iloc[:, newIdx]  # reorder columns
    clstrs = {i: corr0.columns[np.where(kmeans.labels_ == i)[0]].tolist() for i in
              np.unique(kmeans.labels_)}  # cluster members
    print(f'{len(clstrs)} Clusters found.')
    silh = pd.Series(silh, index=dist.index)
    return corr1, clstrs, silh


def makeNewOutputs(corr0, clstrs, clstrs2):
    clstrsNew, newIdx = {}, []
    for i in clstrs.keys():
        clstrsNew[len(clstrsNew.keys())] = list(clstrs[i])
    for i in clstrs2.keys():
        clstrsNew[len(clstrsNew.keys())] = list(clstrs2[i])
    map(newIdx.extend, clstrsNew.values())
    corrNew = corr0.loc[newIdx, newIdx]
    dist = ((1 - corr0.fillna(0)) / 2.) ** .5
    kmeans_labels = np.zeros(len(dist.columns))
    for i in clstrsNew.keys():
        idxs = [dist.index.get_loc(k) for k in clstrsNew[i]]
        kmeans_labels[idxs] = i
    silhNew = pd.Series(silhouette_samples(dist, kmeans_labels), index=dist.index)
    return corrNew, clstrsNew, silhNew


# ------------------------------------------------------------------------------
def clusterKMeansTop(corr0, maxNumClusters=10, n_init=10):
    print(f'Runnning clusterKMeansTop with maxNumClusters={maxNumClusters} and n_init={n_init}.')
    corr1, clstrs, silh = clusterKMeansBase(corr0, maxNumClusters=corr0.shape[1] - 1, n_init=n_init)
    clusterTstats = {i: np.mean(silh[clstrs[i]]) / np.std(silh[clstrs[i]]) for i in clstrs.keys()}
    tStatMean = np.mean(list(clusterTstats.values()))
    redoClusters = [i for i in clusterTstats.keys() if clusterTstats[i] < tStatMean]
    if len(redoClusters) <= 2:
        return corr1, clstrs, silh
    else:
        keysRedo = []
        map(keysRedo.extend, [clstrs[i] for i in redoClusters])
        corrTmp = corr0.loc[keysRedo, keysRedo]
        corr2, clstrs2, silh2 = clusterKMeansTop(corrTmp, maxNumClusters=corrTmp.shape[1] - 1, n_init=n_init)
        # Make new outputs, if necessary
        corrNew, clstrsNew, silhNew = makeNewOutputs(corr0, {
            i: clstrs[i] for i in clstrs.keys() if i not in redoClusters}, clstrs2)
        newTstatMean = np.mean(
            [np.mean(silhNew[clstrsNew[i]]) / np.std(silhNew[clstrsNew[i]]) for i in clstrsNew.keys()])
        if newTstatMean <= tStatMean:
            return corr1, clstrs, silh
        else:
            return corrNew, clstrsNew, silhNew