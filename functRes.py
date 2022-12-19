import os
import base64
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import pandas as pd
import cv2 as cv
import torch
from torchvision.models import resnet50


def preprocess(img):
    c = torch.FloatTensor(cv.resize(img, (224, 224))).permute(2, 0, 1)
    return torch.unsqueeze(c, 0)/255


def vectorResNet(image):
    model = resnet50(True)
    model = model.eval()
    inp = preprocess(image)
    with torch.no_grad():
        out = model(inp)
    emb = out[0].numpy()
    print(emb.shape)
    return emb


def db_createResNet(folder_dir):
    vectors, links = [], []
    c = 0
    for image in os.listdir(folder_dir):
        if image.endswith(".jpg"):
            vectors.append(base64.b64encode((vectorResNet(cv.imread('images/' + image)))))
            links.append(image)
        c += 1
        print(c)

    return pd.DataFrame({"vector": vectors, "link": links})


def get_k_neighbours(vector, df, number_of_neighbours):
    neigh = NearestNeighbors(n_neighbors=number_of_neighbours, metric=lambda a, b: distance.cosine(a, b))
    neigh.fit(df['vector'].to_numpy().tolist())
    return neigh.kneighbors([vector], number_of_neighbours, return_distance=False)


def get_neighbours_links(df, neighbors):
    similar = df.iloc[neighbors[0]]
    return similar['link'].to_numpy().tolist()


# db = db_createResNet('images')
# db.to_csv('outRes.csv')
