from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


# read in the iris data
iris = load_iris()

# create X (features) and y (response)
X = iris.data
y = iris.target

# define the parameter values that should be searched
k_range = range(1, 31)

# create a parameter grid: map the parameter names to the
# values that should be searched
# 下面是构建parameter grid，其结构是key为参数名称，value是待搜索的数值列表的
# 一个字典结构
param_grid = dict(n_neighbors=k_range)

knn = KNeighborsClassifier(n_neighbors=5)
# instantiate the grid
# 这里GridSearchCV的参数形式和cross_val_score的形式差不多，其中param_grid是
# parameter grid所对应的参数
# GridSearchCV中的n_jobs设置为-1时，可以实现并行计算（如果你的电脑支持的情况下）
#我们可以知道，这里的grid search针对每个参数进行了10次交叉验证，
# 并且一共对30个参数进行相同过程的交叉验证
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')

grid.fit(X, y)


#print(grid.cv_results_)
print(grid.best_score_)

print(grid.best_params_)

print(grid.best_index_)


'''
print(grid.grid_scores_)

# examine the first tuple
print (grid.grid_scores_[0].parameters)
print (grid.grid_scores_[0].cv_validation_scores)
print (grid.grid_scores_[0].mean_validation_score)

# create a list of the mean scores only
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print (grid_mean_scores)


plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# examine the best model
print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)
'''
