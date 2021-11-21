### Clustering
分别用K均值K_means和模糊C均值FCM算法对Iris鸢尾花数据集聚类以及图像聚类分割

Iris鸢尾花数据集：http://archive.ics.uci.edu/ml/datasets/Iris
### Contents
- [算法流程图](#算法流程图)
- [结果展示](#结果展示)
  - [K_means](#K_mean算法在Iris数据集聚类和图片聚类分割结果)
  - [FCM](#FCM算法在Iris数据集聚类结果)

### 算法流程图
<div align=center>
<img src="https://github.com/Luxlios/Figure/blob/main/Clustering/flow_chart.png" height="600">
</div>
左边为K均值K_means算法流程图，右边为模糊C均值FCM算法流程图

### 结果展示
#### K_mean算法在Iris数据集聚类和图片聚类分割结果
鸢尾花数据集有四维特征，这里只展示前三维。
<div align=center>
<img src="https://github.com/Luxlios/Figure/blob/main/Clustering/K_means.png" height="300">
</div>
原图片如下。
<div align=center>
<img src="https://github.com/Luxlios/Figure/blob/main/Clustering/hh.png" height="300">
</div>
聚类分割结果如下，簇数分别为4、6、8。
<div align=center>
<img src="https://github.com/Luxlios/Figure/blob/main/Clustering/K_means_img_segment.png" height="300">
</div>

#### FCM算法在Iris数据集聚类结果
鸢尾花数据集有四维特征，这里只展示前三维。同时，由于FCM算法得到的是隶属度矩阵，为了展示，取每个样本隶属度最大的作为其聚类结果。
<div align=center>
<img src="https://github.com/Luxlios/Figure/blob/main/Clustering/FCM.png" height="300">
</div> 
