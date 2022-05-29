import numpy as np
import pandas as pd
from scipy import stats

# sklearn 相关库
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

# 不显示红色警告
import warnings

warnings.filterwarnings('ignore')


def OneHot(x):
    '''
        功能：one-hot 编码
        传入：需要编码的分类变量
        返回：返回编码后的结果，形式为 dataframe
    '''
    # 通过 LabelEncoder 将分类变量打上数值标签
    lb = LabelEncoder()  # 初始化
    x_pre = lb.fit_transform(x)  # 模型拟合
    x_dict = dict([[i, j] for i, j in zip(x, x_pre)])  # 生成编码字典--> {'收藏': 1, '点赞': 2, '关注': 0}
    x_num = [[x_dict[i]] for i in x]  # 通过 x_dict 将分类变量转为数值型

    # 进行one-hot编码
    enc = OneHotEncoder()  # 初始化
    enc.fit(x_num)  # 模型拟合
    array_data = enc.transform(x_num).toarray()  # one-hot 编码后的结果，二维数组形式

    # 转成 dataframe 形式
    df = pd.DataFrame(array_data)
    inverse_dict = dict([val, key] for key, val in x_dict.items())  # 反转 x_dict 的键、值
    # columns 重命名
    if type(x) == pd.Series:
        firs_name = x.name
    else:
        firs_name = ""
    df.columns = [firs_name + "_" + inverse_dict[i] for i in df.columns]

    return df


def ZscoreNormalization(x):
    '''
        Z-score 标准化
    '''
    return (x - np.mean(x)) / np.std(x)


def DataClean(df, Lable=True):
    '''
        数据预处理函数
    '''
    ########################## 1、Pclass 乘客等级 ##########################
    # 无缺失值，等级变量
    # 数据处理：将Pclass分成两类，Pclass>=3、Pclass<3
    df['PclassType'] = ["Pclass>=3" if i >= 3 else "Pclass<3" for i in df['Pclass']]
    # 再对 PclassType 进行One-Hot编码处理
    df = pd.merge(df, OneHot(df['PclassType']), left_index=True, right_index=True)

    ########################## 2、Name 乘客姓名 ##########################
    # 字符串变量
    # 有无缺失值：无
    # 从乘客姓名中获取头街
    # 姓名中头街字符串与定义头街类别之间的关系
    #     Officer: 政府官员，
    #     RoyaIty: 王室(皇室)，
    #     Mr:      已婚男士，
    #     Mrs:     已婚女士，
    #     Miss:    年轻未婚女子，
    #     Master:  有技能的人/教师
    # 新建字段 Title_Dict
    Title_Dict = {
        'Mr': 'Mr',
        'Mrs': 'Mrs',
        'Miss': 'Miss',
        'Master': 'Master',
        'Don': 'Royalty',
        'Rev': 'Officer',
        'Dr': ')fficer',
        'Mme': 'Mrs',
        'Ms': 'Mrs',
        'Major': 'Officer',
        'Lady': 'Royalty',
        'Sir': 'Royalty',
        'Mlle': 'Miss',
        'Col': 'Officer',
        'Capt': 'Officer',
        'the Countess': 'Royalty',
        'Jonkheer': 'Royalty',
        'Dona': 'Royalty'
    }
    df['NameType'] = [Title_Dict[i.split(".")[0].split(", ")[-1]] for i in df['Name']]  # 对Name进行分类
    # 数据进一步处理：将 NameType 分成三类
    # Mr(已婚男士)
    # Mrs(已婚女士)、Miss(年轻未婚女子)
    # 其他
    df['NameType2'] = ["Mr" if i == "Mr" else ("Mrs and Miss" if i in ['Mrs', 'Miss'] else "Other") \
                       for i in df['NameType']]
    # 再对 NameType2 进行One-Hot编码处理
    df = pd.merge(df, OneHot(df['NameType2']), left_index=True, right_index=True)

    ########################## 3、Sex 性别 ##########################
    # 分类变量
    # 有无缺失值：无
    # 对 Sex 进行One-Hot编码处理
    df = pd.merge(df, OneHot(df['Sex']), left_index=True, right_index=True)

    ########################## 4、Age 年龄 ##########################
    # 连续变量
    # 有无缺失值：有，缺失比例19.9%
    # 缺失值用均值填充
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    # 数据处理：将 Age 分成两类，Age<=5、Age>5
    df['AgeType'] = ["Age<=5" if i <= 5 else "Age>5" for i in df['Age']]
    # 再对 AgeType 进行One-Hot编码处理
    df = pd.merge(df, OneHot(df['AgeType']), left_index=True, right_index=True)

    ########################## 5、SibSp 堂兄弟妹个数 ##########################
    # 无缺失值，等级变量
    # 数据处理：将 SibSp 分成两类，SibSp=0、SibSp>0
    df['SibSpType'] = ["SibSp=0" if i == 0 else "SibSp>0" for i in df['SibSp']]
    # 再对 SibSpType 进行One-Hot编码处理
    df = pd.merge(df, OneHot(df['SibSpType']), left_index=True, right_index=True)

    ########################## 6、Parch 父母与小孩的个数 ##########################
    # 连续变量
    # 有无缺失值：无
    # 数据处理：将 Parch 分成两类，Parch=0、Parch>0
    df['ParchType'] = ["Parch=0" if i == 0 else "Parch>0" for i in df['Parch']]
    # 再对 ParchType 进行One-Hot编码处理
    df = pd.merge(df, OneHot(df['ParchType']), left_index=True, right_index=True)

    ########################## 8、Fare 票价 ##########################
    # 连续变量
    # 有无缺失值：无
    # 对 Fare 分成三类
    # Fare = 0
    # Fare <=50
    # Fare > 50
    df['FareType'] = ["Fare=0" if i == 0 else ("Fare<=50" if i <= 50 else "Fare>50") for i in df['Fare']]
    # 再对 FareType 进行One-Hot编码处理
    df = pd.merge(df, OneHot(df['FareType']), left_index=True, right_index=True)

    ########################## 10、Embarked 登船的港口 ##########################
    # 离散变量
    # 有无缺失值：有，缺失值比例很低
    # 数据处理：缺失值按众数填充，然后再进行One-hot编码处理
    mode = stats.mode(df['Embarked'])[0][0]  # 众数
    df['Embarked'] = df['Embarked'].fillna(mode)

    df = pd.merge(df, OneHot(df['Embarked']), left_index=True, right_index=True)

    ########################## 11、FamilyNumbers 家庭人数 ##########################
    # 计算方式：SibSp(堂兄弟妹个数) + Parch(父母与小孩的个数) + 1(自己)
    df['FamilyNumbers'] = df['SibSp'] + df['Parch'] + 1
    # 新增 FamilyType 字段
    # 1 ： 单身（Single）
    # 2-4：小家庭（Family_Small）
    # >4： 大家庭（Family_Large）
    df['FamilyType'] = ['Single' if i == 1 else ('Family_Small' if i <= 4 else 'Family_Large') for i in
                        df['FamilyNumbers']]
    # 对 FamilyType 进行One-hot编码处理
    df = pd.merge(df, OneHot(df['FamilyType']), left_index=True, right_index=True)

    ########################## 删除冗余变量 ##########################
    drop_columns = ['PassengerId', 'Pclass', 'PclassType', 'Name', 'NameType', 'NameType2', 'Sex', 'Age', 'AgeType', \
                    'SibSp', 'SibSpType', 'Parch', 'ParchType', 'Fare', 'FareType', 'Ticket', 'Cabin', 'Embarked', \
                    'FamilyNumbers', 'FamilyType']
    df.drop(drop_columns, axis=1, inplace=True)

    ########################## 数据标准化 ##########################
    if Lable == True:  # 判断是否是测试集（测试集不含标签）
        data = df.drop("Survived", axis=1).agg(ZscoreNormalization)
        data['Lable'] = df['Survived']
    else:
        data = df.agg(ZscoreNormalization)

    return data


def sklearn_DecisionTreeClassifier(data):
    '''
        决策树二分类
    '''
    # 划分训练集、测试集
    x_train, x_test, y_train, y_test = train_test_split(
        data.drop("Lable", axis=1),
        data['Lable'],
        test_size=0.3,
        random_state=0
    )

    print("\n---------- 模型训练 ----------")
    # 网格寻参
    param_grid = {
        'criterion': ['gini', 'entropy'],  # 划分属性时选用的准则：{“gini”, “entropy”}, default=”gini”
        'splitter': ['best', 'random'],  # 划分方式：{“best”, “random”}, default=”best”
        'max_depth': range(1, 6),  # 最大深度
        'min_samples_split': range(1, 6),  # 拆分内部节点所需的最小样本数
        'min_samples_leaf': range(1, 6),  # 叶节点所需的最小样本数
    }
    clf = DecisionTreeClassifier()  # 初始化
    gs = GridSearchCV(clf, param_grid, cv=5)  # 网格搜索与交叉验证
    gs.fit(x_train, y_train)  # 模型训练
    print("Best Estimator: ", gs.best_estimator_)  # 打印最好的分类器
    print("Best Score: ", gs.best_score_)  # 打印最好分数

    # 模型预测
    print("\n---------- 模型评价 ----------")
    y_pred = gs.predict(x_test)  # 预测
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])  # 混淆矩阵
    df_cm = pd.DataFrame(cm)  # 构建DataFrame
    print('Accuracy score:', accuracy_score(y_test, y_pred))  # 准确率
    print('Recall:', recall_score(y_test, y_pred, average='weighted'))  # 召回率
    print('F1-score:', f1_score(y_test, y_pred, average='weighted'))  # F1分数
    print('Precision score:', precision_score(y_test, y_pred, average='weighted'))  # 精确度

    return gs.best_estimator_  # 返回最好的训练模型


if __name__ == "__main__":
    train = pd.read_csv(r'dataset\titanic\train.csv')
    test = pd.read_csv(r'dataset\titanic\test.csv')

    print("\n---------- 数据预处理 ----------")
    train_data = DataClean(train)
    test_data = DataClean(test, Lable=False)

    # 决策树二分类
    best_estimator = sklearn_DecisionTreeClassifier(train_data)

    # 预测
    y_pred = best_estimator.predict(test_data)
    # 输出预测结果
    result = test[['PassengerId']]
    result['Survived'] = y_pred
    result.to_csv("Titanic Results.csv", index=False)

    print("\n程序运行完成")
