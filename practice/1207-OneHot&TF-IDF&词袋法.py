# 基于python原生代码(不依赖任何第三方库的情况)实现一下序号化转换、文本/Token哑编码转换、文本词袋法转换、文本TF-IDF转换的实现
import math
import os
import jieba


docs_str_1 = [
    "今天天气不错",
    "今天天气不行",
    "今天出门踩了狗屎",
    "今天天气不错，今天出去玩",
    "今天天气不错，今天出去玩吧？，那今天出去玩",
    "今天天气不错，今天想出去玩，今天出不出去玩呢？今天不出去玩",
]
# from sklearn.feature_extraction.text import CountVectorizer
# import numpy
# 序号化转换第一步，建立词典 0--pad，1--unk，之后开始构建词典
dic = {
    "PAD": 0,
    "UNK": 1,
    "今天": 2,
    "天气": 3,
    "不错": 4,
    "出去玩": 5,
    "踩": 6,
    "玩": 7,
    "出不出去": 8,
    "想出去": 9,
}

class TestMain:

    @staticmethod
    def resolve_word_2_vector(doc_lst):
        return [dic.get(doc, 1) for doc in doc_lst ]


    @staticmethod
    def resolve_one_hot(doc_lst):
        size = len(dic)
        res = []
        for doc in doc_lst:
            cache = [0] * size
            cache[dic.get(doc, 1)] = 1
            res.append(cache)
        return res

    @staticmethod
    def resolve_ci_dai(doc_lst):
        size = len(dic)
        freq = {}
        for word in doc_lst:
            freq[word] = freq.get(word, 0) + 1
        cache = [0] * size
        # 填充词袋向量
        for word, count in freq.items():
            index = dic.get(word, 1)  # 不存在映射到 UNK
            cache[index] = count
        return cache

    @staticmethod
    def resolve_tf_idf(doc_lst,idf):
        size = len(dic)
        freq = {}
        for word in doc_lst:
            freq[word] = freq.get(word, 0) + 1
        cache = [0] * size
        # 填充词袋向量
        # 2️⃣ TF × IDF
        for word, count in freq.items():
            index = dic.get(word, 1)
            idf_value = idf.get(word, 0)  # 如果没出现过，给0
            cache[index] = count * idf_value
        return cache


    @staticmethod
    def compute_idf(docs):
        N = len(docs)
        df = {}
        for doc in docs:
            for word in set(doc):
                df[word] = df.get(word, 0) + 1
        idf = {}
        for word, count in df.items():
            idf[word] = math.log((N + 1) / (count + 1)) + 1
        return idf


    # 序号化方法
    def test_xu_hao(self, docs_str):
        jieba.load_userdict(os.path.join(os.path.dirname(__file__), "customer_dict.word"))
        # 调用jieba分词
        docs = [jieba.lcut(doc) for doc in docs_str]
        print(docs)
        # 将docs数据转为序号字典内的数据
        datas = [self.resolve_word_2_vector(doc_lst) for doc_lst in docs]
        print(datas)

    # One-Hot方法
    def test_one_hot(self, docs_str):
        jieba.load_userdict(os.path.join(os.path.dirname(__file__), "customer_dict.word"))
        # 调用jieba分词
        docs = [jieba.lcut(doc) for doc in docs_str]
        datas = [self.resolve_one_hot(doc_lst) for doc_lst in docs]
        print(datas)

    def test_ci_dai(self, docs_str):
        jieba.load_userdict(os.path.join(os.path.dirname(__file__), "customer_dict.word"))
        # 调用jieba分词
        docs = [jieba.lcut(doc) for doc in docs_str]
        datas = [self.resolve_ci_dai(doc_lst) for doc_lst in docs]
        print(datas)

    def test_tf_idf(self, docs_str):
        jieba.load_userdict(os.path.join(os.path.dirname(__file__), "customer_dict.word"))
        # 调用jieba分词
        docs = [jieba.lcut(doc) for doc in docs_str]
        idf = self.compute_idf(docs)
        datas = [self.resolve_tf_idf(doc_lst, idf) for doc_lst in docs]
        print(datas)


if __name__ == '__main__':
    s = TestMain()
    # 测试序号化方法
    # s.test_xu_hao(docs_str_1)
    # 测试One-Hot方法
    # s.test_one_hot(docs_str_1)
    # 测试词袋法方法
    # s.test_ci_dai(docs_str_1)
    # 测试TF-IDF方法
    s.test_tf_idf(docs_str_1)
