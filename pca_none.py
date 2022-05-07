from glob import glob
import random
import math
import numpy as np
from PIL import Image
import time
from random import sample 
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import cv2
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix



sob_normal_train_img_fn = glob('./mura_data/RGB/20220407/data/train_data/normal/*.png')
sob_mura_train_img_fn = glob('./mura_data/RGB/20220407/data/train_data/defect/*.png')#
sob_normal_test_img_fn = glob('./mura_data/RGB/20220407/data/test_data/normal/*.png')#8261
sob_mura_img_fn = glob('./mura_data/RGB/20220407/data/test_data/defect/*.png')#305
print(len(sob_normal_train_img_fn))
print(len(sob_mura_train_img_fn))
print(len(sob_normal_test_img_fn))
print(len(sob_mura_img_fn))


random.seed(time.time())

random.shuffle(sob_normal_test_img_fn)
sob_normal_test_img_fn = sob_normal_test_img_fn[:1000]
random.shuffle(sob_mura_train_img_fn)
sob_mura_train_img_fn = sob_mura_train_img_fn[:695]
sob_mura_img_fn.extend(sob_mura_train_img_fn)

print("none")
print(len(sob_normal_train_img_fn))
print(len(sob_normal_test_img_fn))
print(len(sob_mura_img_fn))

N,w=10000,128
read_img = lambda fn: np.array(Image.open(fn).resize((w,w)).convert('L')).ravel()
load_imgs = lambda fn_list: np.array([read_img(fn) for fn in fn_list]).astype("float32")

sob_normal_train_imgs = load_imgs(sample(sob_normal_train_img_fn,N))
sob_normal_test_imgs = load_imgs(sob_normal_test_img_fn)
sob_mura_imgs = load_imgs(sample(sob_mura_img_fn,1000))

print(sob_normal_train_imgs.shape)




#pca = PCA(n_components=512)
pca = PCA(n_components=256)
sob_normal_train_PCA = pca.fit(sob_normal_train_imgs)
print(sob_normal_train_PCA)
def compute_dist(x,pca):
    x_pca = pca.transform(x)
    x_inv = pca.inverse_transform(x_pca)

    return np.linalg.norm(x_inv-x,axis=1)

self_dist = compute_dist(sob_normal_train_imgs, pca)
print("self_dist.shape:",self_dist.shape)
print("normal_train_imgs:",self_dist.mean(),self_dist.std())

ndist = compute_dist(sob_normal_test_imgs, pca)
print("ndist.shape:",ndist.shape)
print("normal_test_imgs",ndist.mean(),ndist.std())#8261

mdist = compute_dist(sob_mura_imgs, pca)
print("mdist.shape",mdist.shape)
print("mura_imgs",mdist.mean(),mdist.std())#1000

dists = np.concatenate([ndist,mdist])
print(len(dists))


pred = []
for i in dists:
    if i>= (self_dist.mean()+self_dist.std()):
        pred.append(1)
    else:
        pred.append(0)
label = [0]*ndist.shape[0]+[1]*mdist.shape[0]

print(confusion_matrix(label, pred))
print("recall:",recall_score(label,pred))
print("precision:",precision_score(label,pred))

print("acc",accuracy_score(label,pred))



print("auc:",roc_auc_score([0]*ndist.shape[0]+[1]*mdist.shape[0],dists))






