import numpy as np

from sklearn.datasets import load_iris
iris = load_iris()

x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


def gini(y):
    sum=len(y)
    p=np.bincount(y)/sum
    thegini=1-(np.sum(p**2))
    return thegini

def bestsplit(x,y):
    bestweigh_gini=1
    bestfeature=None
    bestthreshold=None

    for feature_y in range(x.shape[1]):
        allthreshold=np.unique(x[:,feature_y])
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


def build_decision_tree(x,y,depth):
    if gini(y)<1e-6 or depth==2:
        return node(lable=np.argmax(np.bincount(y)))  #返回值最大的索引,叶子
    feature,threshold=bestsplit(x,y)
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

from sklearn.metrics import accuracy_score  #测试准确度

tree=build_decision_tree(x_train,y_train,0)
print(f"{predict_batch(tree,x_test)}")
accuracy=accuracy_score(predict_batch(tree,x_test),y_test)
print(f"{accuracy}")






































