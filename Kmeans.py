from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

#определяем наяальные параметры системы
seed = 42
n_data = 500
n_clusters = 4
n_centers = 4
# выберем дельту, при которой смещения центроидов можно начать считать незначительными 
delta = 0.0001
flag = True
#в зависимости от параметров системы создаем датасет
blobs, blob_labels = make_blobs(n_samples = n_data, n_features = 2, centers = n_centers, random_state = seed, cluster_std = 2)
blobs *= 0.6

# выбираем начальные случайные центроиды и заносим их координаты в массив
Centroid_indexes = np.random.choice(n_data, n_centers, replace = False)
Centroids = blobs[Centroid_indexes]
prev_Centroids = Centroids
##for i in range(10):   
while flag:          
    # создаем матрицу расстояний
    Dist = (Centroids[:, 0][:, np.newaxis] - blobs[:, 0])**2
    Dist += (Centroids[:, 1][:, np.newaxis] - blobs[:, 1])**2
    Dist = np.sqrt(Dist) 
    
    # массив, где индекс - это номер точки, а значение - это номер кластера
    clust_ind = np.empty(n_data)
    for i in range(n_data):
        clust_ind[i] = np.argmin(Dist[:, i])
    
    # теперь заново пересчитаем кординаты центроидов
    
    #сначала очистим массив с координатами предыдущих центроидов
    prev_Centroids = Centroids.copy()
    Centroids.fill(0)
    
    #создадим массив, в котором индекс - это номер кластера, а значение - количество точек в кластере
    clust_pow = np.empty(n_clusters)
    
    #считаем новые координаты центроидов
    for i in range(n_data):
        if clust_ind[i] == 0:
            Centroids[0, 0] += blobs[i, 0]
            Centroids[0, 1] += blobs[i, 1]
            clust_pow[0] += 1
        if clust_ind[i] == 1:
            Centroids[1, 0] += blobs[i, 0]
            Centroids[1, 1] += blobs[i, 1]
            clust_pow[1] += 1
        if clust_ind[i] == 2:
            Centroids[2, 0] += blobs[i, 0]
            Centroids[2, 1] += blobs[i, 1]
            clust_pow[2] += 1
        if clust_ind[i] == 3:
            Centroids[3, 0] += blobs[i, 0]
            Centroids[3, 1] += blobs[i, 1]
            clust_pow[3] += 1
        if clust_ind[i] == 4:
            Centroids[4, 0] += blobs[i, 0]
            Centroids[4, 1] += blobs[i, 1]
            clust_pow[4] += 1
    cnt = 0       
    for i in range(n_clusters):
        Centroids[i, 0] /= clust_pow[i]        
        Centroids[i, 1] /= clust_pow[i]
        if (abs(Centroids[i,0] - prev_Centroids[i,0]) <= delta) and (abs(Centroids[i,1] - prev_Centroids[i,1]) <= delta):
            flag = False
 
    
    
    #нарисуем результат кластеризации на каждой итерации
    plt.figure(figsize = (6,6))
    for i in range(n_data):
        if clust_ind[i] == 0:    
            plt.scatter(blobs[i,0], blobs[i,1], s = 60, color = '#FF9933', edgecolors = 'white', linewidth = 0.8)
        if clust_ind[i] == 1:
            plt.scatter(blobs[i,0], blobs[i,1], s = 60, color = '#3300FF', edgecolors = 'white', linewidth = 0.8)
        if clust_ind[i] == 2:    
            plt.scatter(blobs[i,0], blobs[i,1], s = 60, color = '#FF3333', edgecolors = 'white', linewidth = 0.8)
        if clust_ind[i] == 3:    
            plt.scatter(blobs[i,0], blobs[i,1], s = 60, color = '#00CC33', edgecolors = 'white', linewidth = 0.8)
        if clust_ind[i] == 4:    
               plt.scatter(blobs[i,0], blobs[i,1], s = 60, color = '#33FFFF', edgecolors = 'white', linewidth = 0.8)
    plt.xlabel('признак 1')
    plt.ylabel('признак 2')    
    plt.grid(lw = 2)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])



plt.figure(figsize = (6,6))
for i in range(n_data):
    if clust_ind[i] == 0:    
        plt.scatter(blobs[i,0], blobs[i,1], s = 60, color = '#FF9933', edgecolors = 'white', linewidth = 0.8)
    if clust_ind[i] == 1:
        plt.scatter(blobs[i,0], blobs[i,1], s = 60, color = '#3300FF', edgecolors = 'white', linewidth = 0.8)
    if clust_ind[i] == 2:    
        plt.scatter(blobs[i,0], blobs[i,1], s = 60, color = '#FF3333', edgecolors = 'white', linewidth = 0.8)
    if clust_ind[i] == 3:    
        plt.scatter(blobs[i,0], blobs[i,1], s = 60, color = '#00CC33', edgecolors = 'white', linewidth = 0.8)
    if clust_ind[i] == 4:    
       plt.scatter(blobs[i,0], blobs[i,1], s = 60, color = '#33FFFF', edgecolors = 'white', linewidth = 0.8)
plt.xlabel('признак 1')
plt.ylabel('признак 2')    
plt.grid(lw = 2)
plt.xlim([-10, 10])
plt.ylim([-10, 10])
        



