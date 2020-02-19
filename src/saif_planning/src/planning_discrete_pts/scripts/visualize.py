from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import glob 
import sys 
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from keras import backend as K
import tensorflow as tf 
# import seaborn as sns
# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_context('notebook', font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})

from image_representation import Embedder, EmbedderV
from vae import vae_occ, vae


def scatter(x, labels, subtitle=None):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
        
    if subtitle != None:
        plt.suptitle(subtitle)
        
    plt.show()

def prepare_data(fol_path='data/target_1', num_samples=100):

        fol_list = glob.glob(fol_path)
        data_list = []

        
        print("folder list: " + str(fol_list))
        for fol in [fol_list[0]]:
            anchor_paths = glob.glob(fol + '/anchors/*.jpg')
            warped_paths = glob.glob(fol + '/warped/*.jpg')
            occluded_paths = glob.glob(fol + '/occluded/*.jpg')

            print(anchor_paths)

            positive_paths = anchor_paths + warped_paths
            
            anchor_imgs = [cv2.imread(img) for img in anchor_paths]
            warped_imgs = [cv2.imread(img) for img in warped_paths]
            occluded_imgs = [cv2.imread(img) for img in occluded_paths]

            positives = anchor_imgs + warped_imgs

            print(np.shape(np.array(anchor_imgs)))

            
        return anchor_imgs, warped_imgs, occluded_imgs

def toFeatureRepresentation(model, img, img_shape=(480,640,3), prep=True):
        img = np.expand_dims(img, axis=0)
        if prep == True:
            img = preprocess_input(img)
        return np.array(model.predict(img)).flatten()

#def toEmbedding(load_path, img):


def eucDist(a, b):
    print("a")
    print(a[0])
    return np.sqrt(np.dot(a - b, np.transpose(a - b)))


def getTriplets(path='data/triplets/'):
    paths = glob.glob(path + '*.jpg')

    num_triplets = int(len(paths)/3)
    triplet_list = []

    print(num_triplets)

    for n in range(num_triplets):
        a = cv2.imread(path + 'a_{}.jpg'.format(n))
        w = cv2.imread(path + 'w_{}.jpg'.format(n))
        o = cv2.imread(path + 'occ_{}.jpg'.format(n))

        triplet_list.append([a, w, o])

    return triplet_list


def visualizeTriplets(triplets, model):

    fig, axes = plt.subplots(10, 3)

    print(len(triplets))
    num_triplets = len(triplets)

    #cv2.imshow('im', cv2.imread('liquid.jpg'))

    triplets = np.array(triplets)

    # plt.imshow(triplets[0, 0])
    # plt.show()

    for t in range(num_triplets):
        a_embed = K.eval(toFeatureRepresentation(model, triplets[t, 0])[0])
        w_embed = K.eval(toFeatureRepresentation(model, triplets[t, 1])[0])
        o_embed = K.eval(toFeatureRepresentation(model, triplets[t, 2])[0])

        aw_dist = eucDist(a_embed[0], w_embed[0])
        ao_dist = eucDist(a_embed[0], o_embed[0])
        
        #print(np.shape(triplets[10][0]))
        axes[t, 0].imshow(triplets[t, 0])
        axes[t, 1].imshow(triplets[t][1])
        axes[t, 2].imshow(triplets[t][2])

        axes[t, 1].set_title(aw_dist)
        axes[t, 2].set_title(ao_dist)

        #axes[0, 0].imshow(cv2.imread('liquid.jpg'))

    plt.show()


def showDistances(anchor, others, model):

    fig, axes = plt.subplots(len(others), 2)


    for t in range(len(others)):
        a_embed = K.eval(toFeatureRepresentation(model, anchor)[0])
        o_embed = K.eval(toFeatureRepresentation(model, others[t])[0])

        ao_dist = eucDist(a_embed[0], o_embed[0])
        
        #print(np.shape(triplets[10][0]))
        axes[t, 0].imshow(anchor)
        axes[t, 1].imshow(others[t])


        axes[t, 1].set_title(ao_dist)


        #axes[0, 0].imshow(cv2.imread('liquid.jpg'))

    plt.show()

def PCA(X, k=2, num_features=1000):
    print("shape of X: {}".format(np.shape(X)))
    X = X.reshape(np.shape(X)[0], num_features)
    print("shape of X: {}".format(np.shape(X)))

    X_mean = np.average(X, axis=0)

    print("shape of X_mean: {}".format(np.shape(X_mean)))

    # for i in range(np.shape(X)[1]):
    #     X[:, i] = X[:, i] - X_mean 

    for i in range(np.shape(X)[0]):
        X[i, :] = X[i, :] - X_mean 

    
    C = np.matmul(np.transpose(X), X)

    print("shape of C: {}".format(np.shape(C)))

    w, V = np.linalg.eig(C)

    sorted_indices = w.argsort()

    # print(w)
    # print(len(w))
    # print(sorted(w))
    
    #plt.scatter(range(len(w)), np.log(sorted(w)))
    #plt.show()

    V_sorted = V[:, sorted_indices[::-1]]

    print("shape of V_sorted: {}".format(np.shape(V_sorted)))

    pca_mat = np.matmul(X, V_sorted)
    print("shape of pca_mat: {}".format(np.shape(pca_mat)))

    return pca_mat[:, 0:k]


def plotEmbeddings(embeddings, labels, num_features=1000):
    pca_embed = PCA(embeddings, num_features=num_features) # , num_features=2048

    
    #scatter(embedded, labels)
    #plt.scatter(np.array(embedded)[:, :, 2], np.array(embedded[:, :, 1]), c=labels)
    plt.scatter(pca_embed[:, 0], pca_embed[:, 1], c=labels)
    plt.show()

def prepareLabels(label_list):
    labels = []
    for label in label_list:
        if label == 1:
            labels.append(np.array([1, 0, 0, 1]))
        elif label == 2:
            labels.append(np.array([0, 1, 0, 1]))
        elif label == 3:
            labels.append(np.array([0, 0, 1, 1]))
        elif label == 4:
            labels.append(np.array([1, 0, 1, 1]))
        elif label == 5:
            labels.append(np.array([1, 1, 0, 1]))
        else:
            labels.append(np.array([0, 1, 1, 1]))

    return labels

def toEmbeddings(paths, model):
    imgs = [cv2.imread(p) for p in paths]
    return np.array([K.eval(toFeatureRepresentation(model, im)[0]) for im in imgs])

if __name__ == "__main__":

    #print(len(inception_model.layers))

    #init_op = tf.initialize_all_variables()

    #with tf.Session() as sess:
    #sess.run(init_op)
    inception_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(480,640,3), pooling='avg', classes=1000)


    model = Embedder()
    model.load()

    # vae_model = vae()
    # vae_model.load()

    v_model = EmbedderV()
    v_model.load()

    triplets = np.array(getTriplets(path='data/test_triplets/'))
    print(triplets)
    #visualizeTriplets(triplets, model)

    #a, w, o = prepare_data()

    print(np.shape(triplets))

    num_triplets = 5
    # a_paths = sorted(glob.glob('data/triplets/a_*.jpg'))[0:num_triplets]
    # w_paths = sorted(glob.glob('data/triplets/w_*.jpg'))[0:num_triplets]
    # occ_paths = sorted(glob.glob('data/triplets/occ_*.jpg'))[0:num_triplets]


    target_num = 1

    a_paths = sorted(glob.glob('data/target_{}/anchors/*.jpg'.format(target_num)))
    w_paths = sorted(glob.glob('data/target_{}/warped/*.jpg'.format(target_num)))
    occ_paths = sorted(glob.glob('data/target_{}/occluded/*.jpg'.format(target_num)))

    a = [cv2.imread(f) for f in a_paths]
    w = [cv2.imread(f) for f in w_paths]
    o = [cv2.imread(f) for f in occ_paths]

    

    print('flag1')

    test_paths = sorted(glob.glob('data/test_imgs/*.jpg'))
    test_paths_B = sorted(glob.glob('data/test_imgs_B/*.jpg'))
    test_paths_occ = sorted(glob.glob('data/test_occ_imgs/*.jpg'))

    test_imgs = [cv2.imread(p) for p in test_paths]
    test_imgs_B = [cv2.imread(p) for p in test_paths_B]
    test_imgs_occ = [cv2.imread(p) for p in test_paths_occ]

    ## embedder
    num_features = 1000
    #embedded = np.array([K.eval(toFeatureRepresentation(model, im)[0]) for im in a + w + o])
    #test_embeddings = toEmbeddings(test_paths, model)
    #test_embeddings_B = toEmbeddings(test_paths_B, model)
    test_embeddings_occ = toEmbeddings(test_paths_occ, model)

    ## embedderV
    # num_features = 2048
    # #embedded = np.array([K.eval(toFeatureRepresentation(v_model, im)[0]) for im in a + w + o])
    # #test_embeddings = toEmbeddings(test_paths, v_model)
    # test_embeddings_B = toEmbeddings(test_paths_B, v_model)

    # ## vae
    # num_features = 1000
    # embedded = np.array([K.eval(toFeatureRepresentation(vae_model, im, False)[0]) for im in a + w + o])

    ### inception
    # num_features = 2048
    # # # embedded = np.array([toFeatureRepresentation(inception_model, im) for im in a + w + o])
    # # test_embeddings = np.array([toFeatureRepresentation(inception_model, im) for im in test_imgs])
    # test_embeddings_B = np.array([toFeatureRepresentation(inception_model, im) for im in test_imgs_B])

    labels = [np.array([1, 0, 0, 1]) for i in range(len(a))] + [np.array([0, 1, 0, 1]) for i in range(len(w))] + [np.array([0, 0, 1, 1]) for i in range(len(o))]


    
    

    test_labels = prepareLabels([1 for i in range(33)] + [3 for i in range(4)] + [4 for i in range(12)] + [2 for i in range(9)])
    test_labels_B = prepareLabels([2 for i in range(33)] + [3 for i in range(13)] + [1 for i in range(33)] + [4 for i in range(12)])
    test_labels_occ = prepareLabels([5 for i in range(5)] + [1 for i in range(12)] + [2 for i in range(8)] + [6 for i in range(10)] + [3 for i in range(4)] + [4 for i in range(10)] + [5 for i in range(12)])


    #plotEmbeddings(embedded, labels, num_features)
    #plotEmbeddings(test_embeddings, test_labels, num_features)
    #plotEmbeddings(test_embeddings_B, test_labels_B, num_features)
    #plotEmbeddings(test_embeddings_occ, test_labels_occ, num_features)

    image_seq = [cv2.imread(img) for img in sorted(glob.glob('data/image_seq/*.jpg'))]

    showDistances(image_seq[0], image_seq, model)
    #a, w, o = triplets[:, 0, :, :, :], triplets[:, 1, :, :, :], triplets[:, 2, :, :, :]

    # embedded = np.array([K.eval(toFeatureRepresentation(model, im)[0]) for im in a + w + o])

    # labels = np.concatenate((np.zeros(len(a)), np.ones(len(w)), np.ones(len(o)) * 2))

    # labels = [np.array([1, 0, 0, 1]) for i in range(num_triplets)] + [np.array([0, 1, 0, 1]) for i in range(num_triplets)] + [np.array([0, 0, 1, 1]) for i in range(num_triplets)]

    # print(np.shape(np.array(embedded)))
    # print("embeddings: " + str(embedded[0]))
    # #embedded = np.array([emb[0:2] for emb in embedded])
    # #print("embeddings: " + str(embedded[0]))
    # print(np.shape(embedded))

    # a_embedded = [K.eval(toFeatureRepresentation(model, im)[0]) for im in a]
    # w_embedded = [K.eval(toFeatureRepresentation(model, im)[0]) for im in w]
    # o_embedded = [K.eval(toFeatureRepresentation(model, im)[0]) for im in o]

    print('flag2')
    #embedded = np.array(a_embedded + w_embedded + o_embedded)

    #pca_embed = PCA(np.transpose(embedded))
    

    # aw_dists = [eucDist(a_embedded[0], w_emb) for w_emb in w_embedded]
    # ao_dists = [eucDist(a_embedded[0], o_emb) for o_emb in o_embedded]
    # print(aw_dists, ao_dists)