import faiss
import time
import numpy as np
from data.data_loader import Pseudolabel_Dataset


class Kmeans_model:
    def __init__(self, k):
        self.k = k

    def cluster(self, data, val_data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb, xv = preprocess_features(data, val_data, pca=256)

        # cluster the data
        # I ,loss, val_I = run_kmeans(data , val_data, self.k, verbose)
        I ,loss, val_I = run_kmeans(xb , xv, self.k, verbose)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        self.val_lists = [[] for i in range(self.k)]
        for i in range(len(val_data)):
            self.val_lists[val_I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss
    
    def cluster_ectractlabels(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features_extractlabels(data)

        # cluster the data
        I ,loss = kmeans_extractlabels(xb , self.k, verbose)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss
    
def run_kmeans(x, val_data, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    print(n_data,d)

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.verbose = True
    clus.max_points_per_centroid = 100
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)

    all_data  = np.concatenate((x, val_data), axis=0)
    _, I = index.search(all_data, 1)

    stats = clus.iteration_stats
    clustering_loss = stats.at(stats.size() - 1).obj


    print(clustering_loss)

    return [int(n[0]) for n in I[:n_data]]  ,clustering_loss, [int(n[0]) for n in I[n_data:]]  


def preprocess_features(npdata,valdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)
    valdata = mat.apply_py(valdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    val_row_sums = np.linalg.norm(valdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]
    valdata = valdata / val_row_sums[:, np.newaxis]

    return npdata, valdata


    
def stft_cluster_assign(images_lists, dataset):
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    return Pseudolabel_Dataset(image_indexes, pseudolabels, dataset)

    
def kmeans_extractlabels(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    print(n_data,d)

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.verbose = True
    clus.max_points_per_centroid = 100
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    stats = clus.iteration_stats
    clustering_loss = stats.at(stats.size() - 1).obj
    # losses = faiss.vector_to_array(clus.iteration_stats)
    # if verbose:
    #     print('k-means loss evolution: {0}'.format(losses))

    print(clustering_loss)

    return [int(n[0]) for n in I]  ,clustering_loss  # , losses[-1]

def preprocess_features_extractlabels(npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata

    