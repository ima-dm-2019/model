from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import Counter
from sklearn.decomposition import LatentDirichletAllocation

class Model():
    """a simple model in this project"""
    def __init__(self, k=9):
        self.k = k
        self.best_result = None
        self.context_diff = None
        self.type_title = {}
        self.type_list = None
        pass

    def _pca(self, weightarray):
        """降维"""
        pca = PCA(n_components=0.99)  # 保留百分之99的主成分
        return pca.fit_transform(weightarray)

    def fit_predict(self, X, data):
        """进行聚类"""
        """
        :param    X     tf_idf矩阵
        :param    data     各个文本的词集组成的列表
        :return    best_result     聚类结果以数字表示
        """
        clusterer = KMeans(self.k, init='k-means++')
        silhouette_num = 0
        for _ in range(30):
            X = self._pca(X)
            result = clusterer.fit_predict(X)
            avg_silhouette = silhouette_score(X, result)  # 计算平均轮廓系数
            if avg_silhouette > silhouette_num:
                silhouette_num = avg_silhouette
                best_result = result
        self.best_result = best_result
        self._type_context(data)
        return best_result

    def _type_context(self,  data):
        """将不同簇的内容分开"""
        count = Counter(self.best_result)
        self.type_list = count.most_common()
        self.context_diff = [[] for _ in range(self.k)]
        for type_index, type_value in enumerate(self.best_result):
            for common_index, common_value in enumerate(self.type_list):
                if type_value == common_value[0]:
                    self.context_diff[common_index].append(data[type_index])

    def return_title(self):
        """实现主题提取"""
        """
        :return   聚类结果中每个数字代表的簇的主题，使用得分最高的前十个词表示
        """
        lda = LatentDirichletAllocation(1)
        for index_context, value_context in enumerate(self.context_diff):
            vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵
            transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
            X = vectorizer.fit_transform(value_context)
            tf_feature_names = vectorizer.get_feature_names()
            tf_idf_new = transformer.fit_transform(X)  # 计算tf-idf
            lda.fit(tf_idf_new)
            self.type_title[str(self.type_list[index_context][0])] = \
                " ".join([tf_feature_names[i] for i in lda.components_[0].argsort()[:-10:-1]])
        return self.type_title