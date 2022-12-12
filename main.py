import os

from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import numpy as np
import pandas as pd
import pickle
import cv2 as cv

model = pickle.load(open('model/finalized_model.sav', 'rb'))


def vectorize(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    _, descriptors = sift.detectAndCompute(gray, None)
    classes = model.predict(descriptors)
    hist, _ = np.histogram(classes, np.arange(1024))
    return hist / hist.sum()


def db_create(folder_dir):
    vectors, links = [], []
    for image in os.listdir(folder_dir):
        if image.endswith(".jpg"):
            vectors.append(vectorize(cv.imread('images/' + image)))
            links.append(image)
    return pd.DataFrame({"vector": vectors, "link": links})


def get_k_neighbours(vector, df, number_of_neighbours):
    neigh = NearestNeighbors(n_neighbors=number_of_neighbours, metric=lambda a, b: distance.cosine(a, b))
    neigh.fit(df['vector'].to_numpy().tolist())
    return neigh.kneighbors([vector], number_of_neighbours, return_distance=False)


def get_neighbours_links(df, neighbors):
    similar = df.iloc[neighbors[0]]
    return similar['link'].to_numpy().tolist()
