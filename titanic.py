import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# print(os.listdir("../input"))
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

import warnings
warnings.filterwarnings('ignore')

# 【データフレームとして読み込み】 データフレーム＝行列データ　Excelシートのようなもの
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_gender_submission = pd.read_csv('../input/gender_submission.csv')

# df_train.head(5)
# 【shapeで行数と列数を確認】
# print(df_train.shape)
# print(df_test.shape)
# print(df_gender_submission.shape)
#　【カラムの確認】
# print(df_train.columns)
# print('-'*10)
# print(df_test.columns)
# 【各列の欠損値（NaN)の数とデータ型の確認】
# df_train.info()
# print('-'*10)
# df_test.info()
# df_train.isnull().sum()
# df_test.isnull().sum()
# 統計量の表示
# df_train と df_test を縦に連結

df_full = pd.concat([df_train, df_test],axis=0,ignore_index=True,sort=True)

# print(df_full.shape) # df_fullの行数と列数を確認
# df_full.describe() # df_fullの要約統計量
# describe percentiles 引数により任意の分位点を指定して表示
# 下記は10%～80%を表示
# df_train.describe(percentiles=[.1,.2,.3,.4,.5,.6,.7,.8])
# オブジェクト型の要素数、ユニーク数、最頻値の出現回数を表示
# ユニーク数を把握が大事　要素数に対してユニークが少なければ重複あり
# df_full.describe(include = 'O')

# sns.countplot(x='Survived',data=df_train)
# plt.title('死亡者数と生存者の数')
# plt.xticks([0,1],['死亡者','生存者'])

# #死亡者と生存者数を表示
# display(df_train['Survived'].value_counts())

# # 死亡者と生存者割合を表示
# display(df_train['Survived'].value_counts()/len(df_train['Survived']))

# print(mpl.matplotlib_fname()) #設定ファイルを表示（matplotlibrcは後で作ります）
# print(mpl.rcParams['font.family']) #現在使用しているフォントを表示
# print(mpl.get_configdir()) #設定ディレクトリを表示
# print(mpl.get_cachedir()) #キャッシュディレクトリを表示
# print('-------------------------------')

# 男女別の生存者数を可視化
# sns.countplot(x='Survived',hue='Sex',data=df_train)
# plt.xticks([0.0,1.0],['死亡','生存'])
# plt.title('男女別の死亡者と生存者の数')
# plt.show()
# sns.countplot(x='Sex',hue='Survived',data=df_train)

# # Sex とSurvivedをクロス集計する
# display(pd.crosstab(df_train['Sex'], df_train['Survived']))
# # クロス集計しSexごとに正規化
# display(pd.crosstab(df_train['Sex'],df_train['Survived'],normalize='index'))

# チケットクラス別の生存者数を可視化
# sns.countplot(x='Pclass',hue='Survived',data=df_train)
# plt.title('チケットクラス別の死亡者と生存者の数')
# plt.legend(['死亡','生存'])
# plt.show()

# # PclassとSurvivedをクロス集計する
# display(pd.crosstab(df_train['Pclass'],df_train['Survived']))
# # クロス集計しPclassごとに正規化する
# display(pd.crosstab(df_train['Pclass'],df_train['Survived'],normalize='index'))

# 年齢の分布
# 全体のヒストグラム
# sns.distplot(df_train['Age'].dropna(),kde=False,bins=30,label='全体')

# # 死亡者のヒストグラム
# sns.distplot(df_train[df_train['Survived']==0].Age.dropna(),kde=False,bins=30,label='死亡')

# # 生存者のヒストグラム
# sns.distplot(df_train[df_train['Survived']==1].Age.dropna(),kde=False,bins=30,label='生存')

# plt.title('乗船者の年齢の分布')
# plt.legend()#凡例を表示

# # 年齢を8等分　CategoricalAge　という変数を作成
# df_train['CategoricalAge'] = pd.cut(df_train['Age'],8)

# # CategoricalAge と　Survived をクロス集計する
# display(pd.crosstab(df_train['CategoricalAge'], df_train['Survived']))

# # クロス集計し、CategoricalAge　ごとに正規化する
# display(pd.crosstab(df_train['CategoricalAge'],df_train['Survived'],normalize='index'))

# 1番若いカテゴリは生存者のほうが割合大きい　その他は死亡者の割合が高い　60歳以上は8割がた死ぬ

# sns.countplot(x='SibSp',data=df_train)
# plt.title('同乗している兄弟・配偶者の数')

# df_train['SibSp_Over'] = [i if i<=1 else 2 for i in df_train['SibSp']]

# sns.countplot(x='SibSp_Over',hue='Survived',data=df_train)
# plt.legend(['死亡','生存'])
# plt.xticks([0,1,2],['0人','1人','2人以上'])
# plt.title('同乗している兄弟・配偶者の数別の死亡者数と生存者の数')
# plt.show()

# # SibSp と　Survivedをクロス集計
# display(pd.crosstab(df_train['SibSp_Over'],df_train['Survived']))

# # クロス集計しSibSpごとに正規化
# display(pd.crosstab(df_train['SibSp_Over'],df_train['Survived'],normalize='index'))

# 兄弟配偶者が　1人＞0人＞2人　のとおりに生存率高い

# sns.countplot(x='Parch',data=df_train)
# plt.title('同乗している両親・子供の数')

# df_train['Parch_Over'] = [i if i <= 2 else 3 for i in df_train['Parch']]

# # Parch_Over ごとに集計し可視化
# sns.countplot(x='Parch_Over',hue='Survived',data=df_train)
# plt.title('同乗している両親子供別の死亡者と生存者の数')
# plt.legend(['死亡','生存'])
# plt.xticks([0,1,2,3],['0人','1人','2人','3人'])
# plt.xlabel('Parch')
# plt.show()

# # Parch と　Survivedをクロス集計する
# display(pd.crosstab(df_train['Parch_Over'],df_train['Survived']))

# # クロス集計しParchごとに正規化する
# display(pd.crosstab(df_train['Parch_Over'],df_train['Survived'],normalize='index'))

# # 生存者割合　1人＞2人＞0人＞3人以上」
# 一人で乗船している人の生存率が低そう

# #SibSpとParchが同乗している家族の数。1を足すと家族の人数となる
# df_train['FamilySize']=df_train['SibSp']+ df_train['Parch']+ 1
# # IsAloneを0とし、2行目でFamilySizeが2以上であれば1にしている
# df_train['IsAlone'] = 0
# df_train.loc[df_train['FamilySize'] >= 2, 'IsAlone'] = 1
# # IsAloneごとに可視化
# sns.countplot(x='IsAlone', hue = 'Survived', data = df_train)
# plt.xticks([0, 1], ['1人', '2人以上'])
# plt.legend(['死亡', '生存'])
# plt.title('１人or２人以上で乗船別の死亡者と生存者の数')
# plt.show()
# # IsAloneとSurvivedをクロス集計する
# display(pd.crosstab(df_train['IsAlone'], df_train['Survived']))
# # クロス集計しIsAloneごとに正規化する
# display(pd.crosstab(df_train['IsAlone'], df_train['Survived'], normalize = 'index'))

# 一人の人は死亡率が高い

# sns.distplot(df_train['Fare'].dropna(),kde=False,hist=True)
# plt.title('運賃の分布')

#　ほとんどの人の運賃は0－100に固まっている
# qcut を用いて運賃を各カテゴリの人数が等しくなるように分割し生存者の割合を確認

# df_train['CategoricalFare'] = pd.qcut(df_train['Fare'], 4)
# df_train[['CategoricalFare','Survived']].groupby(['CategoricalFare'], as_index=False).mean()

# # Categoricak Survived クロス
# display(pd.crosstab(df_train['CategoricalFare'],df_train['Survived']))

# # クロス集計し　CategoricalFareごとに正規化
# display(pd.crosstab(df_train['CategoricalFare'],df_train['Survived'],normalize='index'))

# # 運賃が高くなるにつれ生存率が上がっている

##　名前
# 継承は　大文字アルファベッド＋小文字アルファベッド＋どっと　正規表現を使用して抽出
set(df_train.Name.str.extract('([A-Za-z]+)\.',expand=False))

#　Collections.Counter　を利用して数える
import collections
compellation = collections.Counter(df_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False))
# .most_common(): で数が多い順に並べ替え
# for comp in compellation.most_common():
#     print(comp)
#　敬称をTitle列に入れ年齢の平均値を算出
df_train['Title'] = df_train.Name.str.extract('([A-Za-z]+)\.',expand=False)
df_test['Title'] = df_train.Name.str.extract('([A-Za-z]+)\.',expand=False)
# .mean() で平均値を算出
# df_train.groupby('Title').mean()['Age']

# # 年齢を特徴量として使えるように調整
def title_to_num(title):
    if title == 'Master':
        return 1
    elif title == 'Miss':
        return 2
    elif title == 'Mr':
        return 3
    elif title == 'Mrs':
        return 4
    else:
        return 5

# # リスト内包表記で変換

df_train['Title_num'] = [title_to_num(i) for i in df_train['Title']]
df_test['Title_num'] = [title_to_num(i) for i in df_test['Title']]

#　識別機に学習させて予測させる

#　年齢の補完 とりあえず平均の30
df_train['Age'] = df_train['Age'].fillna(30)
df_test['Age'] = df_train['Age'].fillna(30)

#　欠損地をCで埋め
df_train.loc[df_train['PassengerId'].isin([62,830]),'Embarked']='C'

# 運賃の補完
# df_test[df_test['Fare'].isnull()]

df_train[['Pclass','Fare']].groupby('Pclass').mean()
df_test.loc[df_test['PassengerId'] == 1044,'Fare'] = 13.675550
df_test[df_test['PassengerId'] == 1044]

# 性別を0－1に変換
# 辞書を作成 中かっこで
genders = {'male':0,'female':1}

df_train['Sex'] = df_train['Sex'].map(genders)
df_test['Sex'] = df_test['Sex'].map(genders)

#　Embarked　乗船した港の変換
# ダミー変数
df_train = pd.get_dummies(df_train, columns=['Embarked'])
df_test = pd.get_dummies(df_test,columns=['Embarked'])

# 不要な列の削除
df_train.drop(['Name','Cabin','Ticket','Title'], axis=1, inplace=True)
df_test.drop(['Name','Cabin','Ticket','Title'], axis=1, inplace=True)

# 予測　ランダムフォレスト
X_train = df_train.drop(['PassengerId', 'Survived'], axis=1)
# 不要な列を削除
Y_train = df_train['Survived']
# Y_trainは、df_trainのSurvived列
X_test  = df_test.drop('PassengerId', axis=1).copy()
from sklearn.ensemble import RandomForestClassifier

# ランダムフォレストのインスタンス生成
forest = RandomForestClassifier(random_state=1)


# df_train.head(5)
# Title減らした

# X_train から　Y_train 予測するように学習
forest.fit(X_train,Y_train)

# 正解率
acc_log = round(forest.score(X_train,Y_train)*100,2)
# print(round(acc_log,2),'%')

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# 3分割交差検証を指定し、インスタンス化
kf = KFold(n_splits=3)
# kf.split(X_train.Ytrain)で、X_trainとY_trainを3分割し、交差検証をする
for train_index, test_index in kf.split(X_train, Y_train):
    X_cv_train = X_train.iloc[train_index]
    X_cv_test = X_train.iloc[test_index]
    Y_cv_train = Y_train.iloc[train_index]
    Y_cv_test = Y_train.iloc[test_index]
    forest = RandomForestClassifier(random_state=1)
    forest.fit(X_cv_train, Y_cv_train) # 学習
    predictions = forest.predict(X_cv_test) # 予測
    # acuuracyを表示
    print(round(accuracy_score(Y_cv_test,forest.predict(X_cv_test))*100,2))


# 学習と予測を行う
forest = RandomForestClassifier(random_state=1)
forest.fit(X_train, Y_train)
Y_prediction = forest.predict(X_test)
submission = pd.DataFrame({
                'PassengerId': df_test['PassengerId'],
                'Survived': Y_prediction})
submission.to_csv('submission.csv', index=False)