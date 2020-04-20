# k-Nearext-Neighbor_notes
```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangxing
# datetime:2020/4/19 11:15 上午
# software: PyCharm
from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistanceIndicies = distance.argsort()  # 返回数组从小到大的索引值
    classCount = {}
    for i in range(k):
        voteIlable = labels[sortedDistanceIndicies[i]]
        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


group, labels = createDataSet()
print(classify0([0, 0], group, labels, 3))
```

### 算法步骤：
    1.计算已知类别数据集中的点与当前之间的距离
    2.按照距离递增次序排序
    3.选取与当前点距离最小的k个点
    4.确定前k个点所在的类别的出现频率
    5.返回前k个点出现频率最高的类别作为当前点的预测分类
