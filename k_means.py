# 编写人: Bonnie
# 开发时间：2023/8/9 0:55

import numpy as np
import pandas as pd

class KMeans:
    def __init__(self, data, K):
        self.data=data
        self.K=K

    #训练,传入迭代次数
    def train(self,num_iterations):
        #样本数
        num_samples=self.data.shape[0]
        #随机选择K个中心点
        centroids=KMeans.centroids_init(self.data, self.K)
        closest_centroid_index=np.empty((num_samples,1))
        for _ in range(num_iterations):
            #得到当前每一个样本点到K个中心点的距离，找到最近的中心点
            closest_centroid_index=KMeans.centroids_closest(self.data,centroids)
            centroids=KMeans.centroids_update(self.data, closest_centroid_index, self.K)#更新
        return centroids,closest_centroid_index

    @staticmethod
    def centroids_init(data,K):
        num_samples=data.shape[0]
        random_ids=np.random.permutation(num_samples)
        #随机取的K个聚类中心
        centroids=data[random_ids[:K],:]
        return centroids

    #寻找最近的中心簇
    @staticmethod
    def centroids_closest(data,centroids):
        num_samples=data.shape[0]#样本数
        num_centroids=centroids.shape[0]#中心簇数
        #初始化使每个样本点储存的中心簇索引为0
        closest_centroid_index=np.zeros((num_samples,1))#num_samples个索引
        for i in range(num_samples):
            distance=np.zeros((num_centroids,1))#初始化离三个中心簇的距离
            for centroids_index in range(num_centroids):
                distance_diff=data[i,:]-centroids[centroids_index,:]#得到一个x,y之差的矩阵
                distance[centroids_index]=np.sum(distance_diff**2)#距离的平方
            closest_centroid_index[i]=np.argmin(distance)#返回最小距离索引
        return closest_centroid_index

    #中心点位置更新
    @staticmethod
    def centroids_update(data, closest_centroid_index, K):
        num_features=data.shape[1]#特征值
        centroids=np.zeros((K, num_features))
        for index in range(K):
            #筛选出 与id匹配的样本
            closest_index=closest_centroid_index==index
            #各列特征值求均值即为中心簇的特征值
            centroids[index]=np.mean(data[closest_index.flatten(),:],axis=0)
        return centroids
