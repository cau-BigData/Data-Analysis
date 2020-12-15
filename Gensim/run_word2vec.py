#Topic modelling for humans with gensim. seojaehoon

#단어들 => corpus (리스트의 리스트)
import nltk
from gensim.models.word2vec import Word2Vec
import string
import os

###stopword 목록 다운로드
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))
translator = str.maketrans('', '', string.punctuation)

###데이터셋 읽어오기
text = ''
for root, subdirs, files in os.walk('/content/drive/My Drive/gensim/popular/'):
    for subdir in subdirs:
        for root, subdirs, files in os.walk(root+subdir):
            for file in files:
                with open(root+'/'+file, 'r') as f:
                    text += f.read()

### each.translate(translator) == 특수문자 제거
### x.lower() == 소문자화
### if x.lower() not in stop_words == 불용어제거
clean = [[x.lower() for x in each.translate(translator).split() if x.lower() not in stop_words] for each in text.split('.\n')]

print(clean)
print("------------------------------------------------------------")
#window크기 5, 최소 출현수 5, skip-gram, 10000번 학습
model = Word2Vec(clean, window = 20, min_count=7, sg=1, iter=10000)

print(list(model.wv.vocab.keys()))
print("vocab length : %d"%len(model.wv.vocab))

#유사 의미 찾기
# print(model.wv.most_similar("good"))

#
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

X = model.wv[model.wv.vocab]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], color='r')
plt.show()

print("끝")
