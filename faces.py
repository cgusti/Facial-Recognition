"""
Author: Claudia Gusti
Date: 11/5/2022
Description: midterm
"""

import numpy as np
import matplotlib.pyplot as plt
import time

import util
# TODO: change cluster_5350 to cluster if you do the extra credit
from cluster_5350.cluster import *

######################################################################
# helper functions
######################################################################

def build_face_image_points(X, y):
    """
    Translate images to (labeled) points.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets

    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """

    n,d = X.shape

    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in range(n):
        images[y[i]].append(X[i,:])

    points = []
    for face in images:
        count = 0
        for im in images[face]:
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def generate_points_2d(N, seed=1234):
    """
    Generate toy dataset of 3 clusters each with N points.

    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed

    Returns
    --------------------
        points -- list of Points, dataset
    """
    np.random.seed(seed)

    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]

    label = 0
    points = []
    for m,s in zip(mu, sigma):
        label += 1
        for i in range(N):
            x = util.random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))

    return points


######################################################################
# main
######################################################################

def main():
    ### ========== TODO: START ========== ###
    # # part 1: explore LFW data set
    # # TODO: display samples images and "average" image
    X, y = util.get_lfw_data()
    print(f"number of sample in total dataset: {X.shape}")
    # print("plotting first 5 individual images")
    for i in range(5):
        util.show_image(X[i])
    print("plotting mean of image dataset")
    images_mean = X.mean(axis=0)
    util.show_image(images_mean)
    # # TODO: display top 12 eigenfaces
    # print("plotting eigenfaces...")
    U, mu = util.PCA(X)
    util.plot_gallery(np.array([util.vec_to_image(U[:,i]) for i in range(12)]))
    # # TODO: try lower-dimensional representations
    l_values = [1, 10, 50, 100, 500, 1288]
    for l in l_values:
        Z, Ul = util.apply_PCA_from_Eig(X, U, l, mu)
        X_rec = util.reconstruct_from_PCA(Z, Ul, mu)
        print(f"plotting gallery using {l} dimensions ")
        util.plot_gallery(X_rec[:12, :])
    ### ========== TODO: END ========== ###


    #===============================================
    # (Optional) part 2: test Cluster implementation
    # centroid: [ 1.04022358  0.62914619]
    # medoid:   [ 1.05674064  0.71183522]




    ### ========== TODO: START ========== ###
    X1, y1 = util.limit_pics(X, y, [4, 6, 13, 16], 40)
    points = build_face_image_points(X1, y1)

    # part 3a: cluster faces
    np.random.seed(1234)
    k_means_scores = []
    k_medoids_scores = []

    st_kmeans = time.time()
    for i in range(10):
        k_means_scores.append(kMeans(points, 4).score())
    et_kmeans = time.time()
    elapsed_time_kmeans = et_kmeans - st_kmeans

    st_kmedoids = time.time()
    for i in range(10):
        k_medoids_scores.append(kMedoids(points, 4).score())
    et_kmedoids = time.time()
    elapsed_time_kmedoids = et_kmedoids - st_kmedoids

    print(f"k-means ave: {np.mean(k_means_scores)} \n k-means min: {np.min(k_means_scores)} \n "
          f"k-means max: {np.max(k_means_scores)} \n elapsed time_kmeans: {elapsed_time_kmeans}")
    print(f"k-medoids ave: {np.mean(k_medoids_scores)} \n k-medoids min: {np.min(k_medoids_scores)} \n "
          f"k-medoids max: {np.max(k_medoids_scores)} \n elapsed time_kmedoids: {elapsed_time_kmedoids}")


    # part 3b: explore effect of lower-dimensional representations on clustering performance
    np.random.seed(1234)

    X2, y2 = util.limit_pics(X, y, [4, 13], 40)
    l_values = [x for x in range(1, 43, 2)]
    U, mu = util.PCA(X)

    k_means_scores_PCA = []
    k_medoids_scores_PCA = []

    for l in l_values:
        Z, Ul = util.apply_PCA_from_Eig(X2, U, l, mu)
        # print(f"value of Z.shape with {l} dimensions: {Z.shape}")
        k_means_points = build_face_image_points(Z, y2)
        k_medoids_points = build_face_image_points(Z, y2)

        k_means_scores_PCA.append(kMeans(k_means_points, 2, init='cheat').score())
        k_medoids_scores_PCA.append(kMedoids(k_medoids_points, 2, init='cheat').score())


    print("plotting effects of lower-dimesional representations")
    plt.plot(l_values, k_means_scores_PCA, 'g*', label ='k-means')
    plt.plot(l_values, k_medoids_scores_PCA, 'r*', label='k-medoids')
    plt.xlabel('dimensions')
    plt.ylabel('scores')
    plt.show()

    # part 3c: determine "most discriminative" and "least discriminative" pairs of images
    unique_classes = np.unique(y)
    unique_classes_count = len(unique_classes) #19
    cluster_pair_scores = np.zeros((unique_classes_count, unique_classes_count)) #size = (19,19)
    np.random.seed(1234)

    for i in range(unique_classes_count):
        for j in range(unique_classes_count):
            if i == j:
                cluster_pair_scores[i][j] = np.NAN
            else:
                X_pair, y_pair = util.limit_pics(X, y, [i, j], 40)
                k_means_points = build_face_image_points(X_pair, y_pair)
                score = kMeans(k_means_points, 2, init='cheat').score()
                cluster_pair_scores[i][j] = score

    print(f'\n{cluster_pair_scores}\n')

    #find index of max score:
    ind = np.unravel_index(np.nanargmax(cluster_pair_scores, axis=None), cluster_pair_scores.shape)
    i, j = ind
    highest_score_pair = cluster_pair_scores[ind]
    print(f"highest_score_pair {highest_score_pair}")
    first_img, label = util.limit_pics(X, y, [i], 1)
    second_img, label = util.limit_pics(X, y, [j], 1)

    print("show two most differet faces:")
    util.show_image(first_img)
    util.show_image(second_img)

    #find index of min score
    ind = np.unravel_index(np.nanargmin(cluster_pair_scores, axis=None), cluster_pair_scores.shape)
    i, j = ind
    lowest_score_pair = cluster_pair_scores[ind]
    print(f"lowest score pare: {lowest_score_pair}")
    first_img, label = util.limit_pics(X, y, [i], 1)
    second_img, label = util.limit_pics(X, y, [j], 1)

    print("show two most similar faces:")
    util.show_image(first_img)
    util.show_image(second_img)

    ### ========== TODO: END ========== ###


if __name__ == "__main__":
    main()
