import numpy as np
from sklearn.datasets import load_iris

# 加载鸢尾花数据集
iris = load_iris()

# 特征矩阵 (150x4)
x = iris.data

# 标签数组 (150,)
y = iris.target

from sklearn.model_selection import train_test_split

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


def gini(y):
    sum=len(y)
    p=np.bincount(y)/sum   #bincount返回出现次数
    thegini=1-(np.sum(p**2))
    return thegini

def bestsplit(x,y,sum_selected_feature=None):
    bestweigh_gini=1
    bestfeature=None
    bestthreshold=None
    sum_feature=x.shape[1]

    if sum_selected_feature is None:
        selected_feature=range(sum_feature)
    else:
        selected_feature=np.random.choice(sum_feature,sum_selected_feature,replace=False)

    for feature_y in selected_feature:
        allthreshold=np.unique(x[:,feature_y])  #unique返回不重复出现的数字,表示取了第feature y列
        for threshold in allthreshold:
            left_mask=x[:, feature_y]<=threshold
            all=len(left_mask)
            left_gini=gini(y[left_mask])
            leftall=len(y[left_mask])
            right_gini=gini(y[~left_mask])
            rightall=len(y[~left_mask])
            weigh_gini=left_gini*(leftall/all)+right_gini*(rightall/all)

            if weigh_gini<bestweigh_gini:
                bestweigh_gini=weigh_gini
                bestfeature=feature_y
                bestthreshold=threshold
    return(bestfeature,bestthreshold)

class node:
    def __init__(self,feature=None,threshold=None,left_tree=None,right_tree=None,lable=None):
        self.feature=feature
        self.threshold=threshold
        self.left_tree=left_tree
        self.right_tree=right_tree
        self.lable=lable

def build_decision_tree(x,y,depth,sum_selected_feature=None):
    if gini(y)<1e-6 or depth==3:
        return node(lable=np.argmax(np.bincount(y)))  #返回值最大的索引,叶子

    feature,threshold=bestsplit(x,y,sum_selected_feature)
    left_mask=x[:,feature]<=threshold
    left_treex=x[left_mask]
    left_treey=y[left_mask]
    right_treex=x[~left_mask]
    right_treey=y[~left_mask]
    left_tree=build_decision_tree(left_treex,left_treey,depth+1)
    right_tree=build_decision_tree(right_treex,right_treey,depth+1)

    return node(feature,threshold,left_tree,right_tree) #返回根节点

def predict(tree,x_test):
    if tree.lable is not None:
        return tree.lable
    if x_test[tree.feature]<=tree.threshold:
        return predict(tree.left_tree,x_test)
    else:
        return predict(tree.right_tree,x_test)

def predict_batch(tree,x_test):
    all_predict=[]
    for row in x_test:
        prediction=predict(tree,row)
        all_predict.append(prediction)
    return np.array(all_predict)

# from sklearn.metrics import accuracy_score  #测试准确度
#
# tree=build_decision_tree(x_train,y_train,0)
# print(f"{predict_batch(tree,x_test)}")
# accuracy=accuracy_score(predict_batch(tree,x_test),y_test)
# print(f"{accuracy}")

class random_forest:
    def __init__(self, sum_trees=10, max_depth=3, sum_selected_features=None):
        self.sum_trees = sum_trees
        self.max_depth = max_depth
        self.sum_selected_features = sum_selected_features
        self.trees = []

    def build_random_forest(self,x,y):
        sum_samples,sum_features=x.shape

        if self.sum_selected_features is None:
            self.sum_selected_features = int(np.sqrt(sum_features))
        else:
            self.sum_selected_features = min(self.sum_selected_features, sum_features)

        for num in range(self.sum_trees):
            random_sample_index=np.random.choice(sum_samples,sum_samples,replace=True)
            x_sample=x[random_sample_index]
            y_sample=y[random_sample_index]

            selected_feature=np.random.choice(sum_features,self.sum_selected_features,replace=False)
            tree=build_decision_tree(x_sample[:, selected_feature],y_sample,depth=0,sum_selected_feature=self.sum_selected_features)
            self.trees.append((tree,selected_feature))
    def predict(self,x):
        all_predict=[]
        for tree,selected_feature in self.trees:
            x_to_predict=x[:,selected_feature]
            predict=predict_batch(tree,x_to_predict)
            all_predict.append(predict)
        all_predict=np.array(all_predict)
        all_predict=all_predict.T
        all_result=[]
        for col in all_predict:
            one_result=np.bincount(col).argmax()
            all_result.append(one_result)
        all_result=np.array(all_result)
        return all_result

my_random_forest=random_forest(sum_trees=10,max_depth=3)
my_random_forest.build_random_forest(x_train,y_train)
y_predict=my_random_forest.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_predict)
print(f"随机森林准确率: {accuracy}")



