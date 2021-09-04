



Columbia 21S ML for Social Sciences 

## Linear Regression

The true linear functions are never linear

RSS: sum of residuals ( difference between y hat and y)

### Confidence Interval

how much confidence to include the true population params, not prob

e.g. βˆ 1 ± 2 · SE(βˆ 1)

### Hypothesis Testing

​	H0: No relationship, βˆ 1  = 0 

​	H1: βˆ 1  != 0 

​	t = (βˆ 1 − 0 ) / SE(βˆ 1)  t test 

### R**-**squared (**R**2) 

is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by  variables in a regression model

​	R square = 1 - RSS/TSS

We interpret βj as the <u>**average effect**</u> on Y of a one unit increase in Xj , <u>**holding all other predictors fixed**</u>. [sloppy interpretation]

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# Controls the shuffling applied to the data before applying the split. shuffle is default 
lr = LinearRegression()
lr.fit(X_train, y_train)

#The “slope” parameters (w), also called weights or coefficients, are stored in the coef_
#..attribute, while the offset or intercept (b) is stored in the intercept_ attribute:

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

# Let’s look at the training set and test set performance using r squared:

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

#cross validation
from sklearn.model_selection import cross_val_score

print(np.mean(cross_val_score(LinearRegression(), X_train, y_train, cv=10, scoring="r2")))

# stats models approach , gives more stats data like R
import statsmodels.api as sm

X_train_new = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_new ).fit()

model.summary() # get a complete summary of the model

```



## Supervised Learning

Assumption: i.i.d of x and y

### KNN

Use distance to represent similarity ; KNN Regression with unscaled data is not accurate.

majority vote of the k nearest neighbors, odd number for k 

the distance need to take square root 

<u>**Scale your IVs**</u> before taking distance calculations 

Minimal training but expensive testing

![](https://i.loli.net/2021/02/22/vePIbk9cxH1OSK3.jpg)

​                                                                                     *sweet point

```python
y = data['InMichelin']  #lower y 
X = data.loc[:, data.columns != 'InMichelin']  # cap XXX

from sklearn.model_selection import train_test_split

# Use train_test_split(X,y) to create four new data sets, defaults to .75/.25 split
X_train, X_test, y_train, y_test = train_test_split(X, y)   # can add randome here

KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

#Print accuracy rounded to two digits to the right of decimal
print("accuracy: {:.2f}".format(knn.score(X_test, y_test)))
y_pred = knn.predict(X_test) 
y_pred # view predictions for test data
```



### Cross Validation 

![](https://i.loli.net/2021/02/22/rfzgEv7LSGn5WaY.jpg)

We separate training data and test data first and then do cv within training data.

1. A model is trained using k−1 of the folds as training data;

2. the resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).

3. The performance measure reported by *k*-fold cross-validation is then the average of the values computed in the loop. 

4. use cross validation first, then go to see test score

• Leave One Out : K Fold(n_folds=n_samples) High variance, takes a long time

• Better: Repeated K Fold. Apply K Fold or Stratified K Fold multiple times with shuffled data. Reduces variance! 

• E.g. – You do 5 fold CV 10 times with data randomly shuffled before each 5 fold CV

```python
#import cross validation functions from sk learn
#*** split first , CV using your training data only 
from sklearn.model_selection import cross_val_score
# we need to import if we want to change the form of cv
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold

# Set up function parameters for diff't cross validation strategies
kfold = KFold(n_splits=5)
skfold = StratifiedKFold(n_splits=5, shuffle=True) # acoid using the default order of data, get rid of systematic issue 
rkf = RepeatedKFold(n_splits=5, n_repeats=10)

print("KFold:\n{}".format(
cross_val_score(KNeighborsClassifier(), X_training, y_training, cv=kfold)))

# result is the average of the five 

print("StratifiedKFold:\n{}".format(
cross_val_score(KNeighborsClassifier(n_neighbors=5), X_training, y_training, cv=skfold)))

print("RepeatedKFold:\n{}".format(
cross_val_score(KNeighborsClassifier(n_neighbors=5), X_training, y_training, cv=rkf)))
```



Tuning Models With Grid Search 

```python
from sklearn.model_selection import GridSearchCV
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

#create dictionary data object with keys equal to parameter name 'n_neighbors' 
#for knn model and values equal to range of k values to create models for

param_grid = {'n_neighbors': np.arange(1, 15, 2)} #np.arange creates sequence of numbers for each k value

grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=10) #instansiation can use cv=rkf from the prev stratify method 

#use meta model methods to fit score and predict model:
grid.fit(X_train, y_train) #auto find the best model 

#extract best score and parameter by calling objects "best_score_" and "best_params_"
print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))
print("test-set score: {:.3f}".format(grid.score(X_test, y_test))) 
#we use the cv model, which uses more data and is more robust

# view data with complete tuning results
results = pd.DataFrame(grid.cv_results_)
results

```



#### Defaults  

Three-fold is default number of folds • For classification cross-validation is stratified • train_test_split has stratify option: train_test_split(X, y, • stratify=y) • No shuffle for repeat sampling by default

Now we can choose different k value and then use cv to choose the best model

### KNN regression model

instead of majority vote, use the average value of y 

## Shrinkage Methods

shrinking the coefficient (towards 0) estimates can significantly reduce their variance

### Ridge Regression 

![](https://i.loli.net/2021/02/22/hDLmg5JNrT62ws1.jpg)

we add a penalty param 

Selecting a good value for λ is critical; cross-validation is used for this.

![](https://i.loli.net/2021/02/22/mGbVrguC7ldKWMQ.jpg)

The Bias-Variance tradeoff:  the property of a model that the variance of the parameter estimates across samples can be reduced by increasing the bias in the estimated parameters.

One obvious disadvantage: unlike subset selection, which will generally select models that involve just a subset of the variables, ridge regression will include all p predictors in the final model, so.... we may consider use lasso.

The Lasso is a relatively recent alternative to ridge regression that overcomes this disadvantage.

```python
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

ridge10 = Ridge(alpha=10).fit(X_train, y_train) #tuning param
print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))

```

### Lasso 

Normalize data first 

![](https://i.loli.net/2021/02/22/ngSKbtfs6V4PZJG.jpg)

### Comparison

However, in the case of the lasso, the L1 penalty has the effect of forcing some of the coefficient estimates to be exactly equal to zero when the tuning parameter λ is sufficiently large. 

Hence, much like best subset selection, the lasso performs variable selection. We say that the lasso yields sparse models — that is, models that involve only a subset of the variables. As in ridge regression, selecting a good value of λ for the lasso is critical; cross-validation is again the method of choice.

Lasso Model has a smaller MSE compared to ridge.

In general, one might expect the lasso to perform better when the response is a function of only a relatively small number of predictors.

A technique such as cross-validation can be used in order to determine which approach is better on a particular data set.

```python
from sklearn.linear_model import Lasso

# Lower alpha to fit a more complex model
# we increase the default setting of "max_iter",
# otherwise the model would warn us that we should increase max_iter.

lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)

print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))


```

### Selecting the Tuning Parameter for Ridge Regression and Lasso 

As for subset selection, for ridge regression and lasso we require a method to determine which of the models under consideration is best. 

That is, we require a method selecting a value for the tuning parameter λ or equivalently, the value of the constraint s. 

 Cross-validation provides a simple way to tackle this problem. We choose a grid of λ values, and compute the cross-validation error rate for each value of λ. 

We then select the tuning parameter value for which the cross-validation error is smallest. 

Finally, the model is re-fit using all of the available observations and the selected value of the tuning parameter.

## Logistic Regression

Logistic regression is a statistical model that in its basic form uses a [logistic function](https://en.wikipedia.org/wiki/Logistic_function) to model a [binary](https://en.wikipedia.org/wiki/Binary_variable) [dependent variable](https://en.wikipedia.org/wiki/Dependent_variable). Either 1 or 0.

p(X) = Pr(Y = 1|X)   the Y value is fixed as one of the two

![](https://i.loli.net/2021/02/22/KlzSkRTaBJE3Hbn.jpg)

the LogisticRegression in sklearn is already a penalized version

so if u want a regular logistic Regression, set c to a very high value 

smaller c means high penalty

For the penalized and non-penalized logistic regression model, the non penalized model should set C=1e90 OR penalty=None; the penalized model should set a small C with either nothing or "penalty=l1". Note that the default is penalty=l2. l1 means lasso, l2 means ridge. They are two different methods to execute the penalty. 

```python
#Set up training and test data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# all logistic models in this model , change c to use different ones
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 

#Note: random_state ensures same data will be generated for example each time

#Note: logistic regression in sklearn is preset to be a regularization model with C=100).
#If you make C really high the model effectively becomes a logistic regression model...
#larger C means larger penalty, e means 10 to the power of 
logreg = LogisticRegression(C=1e90).fit(X_train, y_train)

print("logreg .coef_: {}".format(logreg .coef_))
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

predicted_vals = logreg.predict(X_test) # y_pred includes your predictions
print("logreg.predict: {}".format(predicted_vals))
#sklearn transfer the labels for us
```

```python
# Smaller C will constrain Betas more.  It's a tuning parameter we can find using gridsearch.
#C=100, compare coefs to regular model above. default l2 penalty
#default penalty L2.   need to change l1, if want lasso
# has penalty param
logreg = LogisticRegression(C=100).fit(X_train, y_train)

print("logreg .coef_: {}".format(logreg .coef_))
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

predicted_vals = logreg.predict(X_test) # y_pred includes your predictions
print("logreg.predict: {}".format(predicted_vals))
```



## Support Vector Machine (SVM)

### Motivation

Simply draw a line can lead to large variance in our prediction -> rather than simply drawing a zero-width line between the classes, we can draw around each line a *margin* of some width, up to the nearest point -> In SVM, the line that maximizes this margin is the optimal model.

### USE

Unlike models where we model each class, with SVM we simply find a line or curve (in two dimensions) or manifold (in multiple dimensions) that divides the classes from each other (with the largest gap, maximize the margin). We can move our observations around using math vector transformation.

A key to this classifier's success is that for the fit, only the position of the support vectors matter; any points further from the margin which are on the correct side do not modify the fit! Technically, this is because these points do not contribute to the loss function used to fit the model, so their position and number do not matter so long as they do not cross the margin. SVM's insensitivity to the exact behavior of distant points is one of its strengths. **SVM can work with small data.** 

Because they are affected only by points near the margin, they work well with high-dimensional data—even data with more dimensions than samples, which is a challenging regime for other algorithms.

```python
from sklearn.svm import SVC # "Support vector classifier"
model = SVC(kernel='linear', C=1E10) # kernel and C are the main args to adjust for linear SVC
model.fit(X, y)

```

### Kernel SVM

Kernel Trick:  rising every points in the center, then we can separate the data in 3D dimensions. Many Kernels. with this additional dimension, the data becomes trivially linearly separable,

A potential problem with this strategy—projecting N points into N dimensions—is that it might become very computationally intensive as N grows large. However, because of a neat little procedure known as the [*kernel trick*](https://en.wikipedia.org/wiki/Kernel_trick), a fit on kernel-transformed data can be done implicitly—that is, without ever building the full N-dimensional representation of the kernel projection! This kernel trick is built into the SVM, and is one of the reasons the method is so powerful.

In Scikit-Learn, we can apply kernelized SVM simply by changing our linear kernel to an RBF (radial basis function) kernel, using the `kernel` model hyperparameter:

```python

# normal SVM
from sklearn.datasets.samples_generator import make_circles
X, y = make_circles(100, factor=.1, noise=.1)

clf = SVC(kernel='linear').fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn') #plot
plot_svc_decision_function(clf, plot_support=False);

# kernel,C, and gamma are the main args to adjust for 
# SVC. C and gamma are tuning parameters.
clf = SVC(kernel='rbf', C=1E6, gamma=.1)                                      
clf.fit(X, y)
```



![2021-02-22](https://i.loli.net/2021/02/22/UV96yQjEH7NzgbZ.png)

Using this kernelized support vector machine, we learn a suitable nonlinear decision boundary. This kernel transformation strategy is used often in machine learning to turn fast linear methods into fast nonlinear methods, especially for models in which the kernel trick can be used.

### Tuning Params

#### C

Motivation: if our data has some amount of overlap

The SVM implementation has a bit of a fudge-factor which "softens" the margin: that is, it allows some of the points to creep into the margin if that allows a better fit.

C: for very large C, the margin is hard, and points cannot lie in it. For smaller C, the margin is softer, and can grow to encompass some points. The optimal value of the C parameter will depend on your dataset, and should be tuned using cross-validation or a similar procedure

Soft margin: C in sklearn, move the parallel lines, enable flexibility. If C is large, it is close to the hard margin. If C is small, we will accept more miscalculations.

![2021-02-22 (2)](https://i.loli.net/2021/02/22/Di6hSm3JraZ2znp.png)

#### Gamma

So for smaller gammas two points can be considered similar even if are far from each other and for larger gammas we require a more strict standard for similarity.

##### Case Face Recognition

Each image contains [62×47] or nearly 3,000 pixels. We could proceed by simply using each pixel value as a feature, but often it is more effective to use some sort of preprocessor to extract more meaningful features; here we will use a principal component analysis) to extract 150 fundamental components to feed into our support vector machine classifier. We can do this most straightforwardly by packaging the preprocessor and the classifier into a single pipeline:

```python
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

pca = PCA(svd_solver='randomized',n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

#split
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target,                 random_state=42)
'''
Finally, we can use a grid search cross-validation to explore combinations of parameters. Here we will adjust `C` (which controls the margin hardness) and `gamma` (which controls the size of the radial basis function kernel), and determine the best model:
'''
from sklearn.model_selection import GridSearchCV
param_grid = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)

%time grid.fit(Xtrain, ytrain)
print(grid.best_params_)
print(grid.best_estimator_)
print(grid.best_score_)
```

The optimal values fall toward the middle of our grid; if they fell at the edges, we would want to expand the grid to make sure we have found the true optimum.

Now with this cross-validated model, we can predict the labels for the test data, which the model has not yet seen:

```python
model = grid.best_estimator_
yfit = model.predict(Xtest)
# We can get a better sense of our estimator's performance using the classification report, which lists recovery statistics label by label
from sklearn.metrics import classification_report
print(classification_report(ytest, yfit,
                            target_names=faces.target_names))
```

### Disadvantages

- Transformations in SVMs can result in huge amounts of data being generated. For large numbers of training samples, this <u>**computational cost**</u> can be prohibitive.
- The results are strongly dependent on a suitable choice for the softening parameter C. This must be carefully chosen via cross-validation, which can be expensive as datasets grow in size.
- The results do **<u>NOT have a direct probabilistic interpretation</u>**. This can be estimated via an internal cross-validation (see the `probability` parameter of `SVC`), but this extra estimation is costly. (We tend to stick with models that are built to generate probabilities efficiently as a result)

## Scaling 

Not all models require scaling the data. For ridge and lasso, the constraints the coefs, it is necessary.

### Standard Scaler 

Centering mean on zero and std dev on 1 for each variable; Usually, Z score used for tabular data, minmax for image data.

To calculate the standard scaler subtract mean from observation value and divide result by standard deviation. Scaled data has zero mean and unit variance: we continue to use the scaling method (mean, variance) on training data.

```python
#The function scale provides a quick and easy way to perform this operation on a single array-like dataset:
from sklearn import preprocessing
import numpy as np

#Build dataset with three columns and three rows
#Structure is visually the same as a typical dataframe (i.e.-columns are up and down and rows side to side)

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
X_scaled = preprocessing.scale(X_train)
X_scaled

```

Transformed to Gaussian with zero mean and unit variance. The preprocessing module further provides a utility class StandardScaler that implements the Transformer API to compute the mean and standard deviation on a training set so as to be able to later reapply the same transformation on the testing set. This class is hence suitable for use in the early steps of a sklearn.pipeline.

```python
# set up the the standard scaler to the X_train using fit()
scaler = preprocessing.StandardScaler().fit(X_train)
print(scaler) # show details of scaler object

#apply the fit standard scaler to the X_train data using transform()
scaler.transform(X_train)                         

# print(scaler.mean_) #print the means  per column  

#The scaler instance can then be used on new data to transform it the same way it did on the training set:
#Note that we are scaling new data to the scale built from the training data.  

X_test = [[-1., 1., 0.]]
scaler.transform(X_test) # Transform x_test before running a model, for example        

#It is also possible to disable either centering or scaling by either passing with_mean=False or with_std=False
#to the constructor of StandardScaler.
# Use transformed data with a model:
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = preprocessing.StandardScaler() # scaler instance
scaler.fit(X_train) # set up before using the instance 

X_train_scaled = scaler.transform(X_train) #transform 
ridge = Ridge().fit(X_train_scaled, y_train)

X_test_scaled = scaler.transform(X_test)#transform 
ridge.score(X_test_scaled, y_test)
```



**? scaling before split**

This will bias the model evaluation because information would have leaked from the test set into the training set. Better to use scaler and model in a pipeline.  e.g. u r using the variance and mean of the entire set, some info leaked 

### MinMax Scaler 

Normalization. An alternative standardization is scaling features to lie between a given minimum and maximum value, often between zero and one, or so that the maximum absolute value of each feature is scaled to unit size. This can be achieved using MinMaxScaler.

For each value in a column of X 
$$
Xscaled = (x - min) / (max - min)
$$
min max refers to the value of the column

```python
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train) # fit_transform does both at once.  It's a little faster.

# And again we can then use the scaler to transform new data.
#Note once more that we are scaling new data to the scale built from the training data.  

X_test = np.array([[ -3., -1.,  4.]])
X_test_minmax = min_max_scaler.transform(X_test) #use the same to transform new data 
X_test_minmax
```

### Scikit-learn Pipeline

Sequentially apply a list of transforms and a final estimator.

```python
from sklearn.preprocessing import StandardScaler 
#now we don't need to add preprocessing. before calls to StandardScaler()

from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(), Ridge()) #in one

pipe.fit(X_train, y_train) 
pipe.score(X_test, y_test) #auto transform the new data and make the prediction test
```

### Pipeline and GridSearchCV

```python
knn_pipe = make_pipeline(StandardScaler(), KNeighborsRegressor())
print(knn_pipe.steps) # names of steps in single quotes (i.e.-'standardscaler' and 'kneighborsregressor')

from sklearn.model_selection import GridSearchCV
#refer to step name with two underscores before argument name when...
#you build a parameter grid
param_grid = {'kneighborsregressor__n_neighbors': range(1, 10)} #separate by __ (2_)

grid = GridSearchCV(knn_pipe, param_grid, cv=10)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.score(X_test, y_test))
```



## Decision Tress 

### Definition

**code: tree Visualization and case not required**

A simple decision tree built on this data will iteratively split the data along one or the other axis according to some quantitative criterion, and at each level assign the label of the new region according to a majority vote of points within it. It splits the data based binary criteria, using RSS reduction to decide which criteria to use.

### Params of tree

max_depth: tunes the number of times internal nodes are split. (i.e. - tree size parameter)

criterion:  gini' or 'entropy'

min_samples_leaf: set threshold for minimum number of observations per terminal node. (also a tree size parameter)

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
model = tree.fit(X,y)
print(model)
```



```python
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
model = regressor.fit(X,Y) #note our criterion is mse
print(model) #Tune same parameters as above.

#Extract info
print(model.feature_importances_) 
#in same order as feature names in data
boston.feature_names # RM is most important variable in model.  Variable measures the Number of rooms.
```

Just as using information from two trees improves our results, we might expect that using information from many trees would improve our results even further.

pruning a tree , we decide the size of a tree to prevent overfitting 

### Calculate the error rate 

Gini index (G) : how pure your class , G closer to zero, the purer our prediction

cross entropy is an alternative to gini index 

we find conditional relationship 

In practice, decision trees do not work well, we use ensemble tree

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()

model = tree.fit(X,y)

print(model)


```

### DT Regressor 

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

boston = load_boston()
regressor = DecisionTreeRegressor(random_state=0)
model = regressor.fit(boston.data,boston.target) #note our criterion is mse
print(model) #Tune same parameters as above.


# how each feature reduces RSS
print(model.feature_importances_) 
#in same order as feature names in data
boston.feature_names # RM is most important variable in model.  Variable measures the Number of rooms.
```

### Random Forests

Random forests provide an improvement over bagged trees by way of a small tweak that **<u>decorrelates</u>** the trees and pick up the most meaningful signal. This reduces the variance when we average the trees. 

A random selection of <u>**m**</u> predictors is chosen as split candidates from the full set of <u>**p**</u> predictors. The split is allowed to use only one of those m predictors.  A fresh selection of m predictors is taken at each split, and typically we choose m ≈ sqrt(p) ( the max_features param)

#### Params

- n_estimators: Number of trees(bootstraps) to generate for model
- max_depth: tunes the number of times internal nodes are split. (i.e.-tree size)
- max_features: Number of randomly selected features per split, sqrt(p)
- criterion: 'gini' or 'entropy' for classification and 'mse' for regression
- min_samples_leaf: set threshold for minimum number of observations per terminal node. (also a tree size parameter) 
- oob_score: returns out of bag score in model fit object. Extract result by printing the 'estimators_features_' from model fit object.

```python
#Classification model example first...

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200)
model.fit(X,y)

# we also have random forest regression
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=200)
forest.fit(boston.data,boston.target)
```

#### Advantages:

- Both training and prediction are very fast, because of the simplicity of the underlying decision trees. In addition, both tasks can be straightforwardly parallelized, because the individual trees are entirely independent entities.
- The nonparametric model is extremely flexible, and can thus perform well on tasks that are under-fit by other estimators.

#### Disadvantage:  

- results are not easily interpretable

### Bagging 

bagging and boosting are not just for trees

bootstrap aggregation, to reduce variance of a statistical learning method, can avoid overfitting data.

- randomly select observations with replacement

- we reduce our variance by averaging a set of observations

- fit model to each bagging data set, then we average all the predictions/ majority vote (classifier) to obtain our prediction


Out-of_bag Error, in practice, we use cv .

```python
# Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

tree = DecisionTreeClassifier() #Need to instantiate a model type for bagging first
bag = BaggingClassifier(tree, n_estimators=100, 
                        random_state=1) #input tree

bag.fit(X, y)

# Regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor


tree = DecisionTreeRegressor() #Need to instantiate a model type for bagging first

bag = BaggingRegressor(tree, n_estimators=100, 
                        random_state=1)
bag.fit(boston.data,boston.target)
```

#### Params

- n_estimators: Number of trees to generate for model
- max_depth: tunes the number of times internal nodes are split. (i.e.-tree size)
- max_samples=1.0 : max_samples changes whether you want to draw bootstrap datasets that are the same size as your original dataset or not. Leave this at a default of 1.
- criterion: 'gini' or 'entropy' for classification and 'mse' for regression
- min_samples_leaf: set threshold for minimum number of observations per terminal node. (also a tree size parameter)
- oob_score: returns out of bag score in model fit object. Extract result by printing the 'estimators_features_' from model fit object.

## Boosting

iteratively fit the residuals 

- start from our training data, build decision tree,
- generate prediction from that decision treee
- calculate the residuals, 
- **use the residuals as the new y training data.**
- generate new predictions 
- to use the second prediction to adjust our original predictions

$$
Prediction_{new} = prediction_{ori} + prediction_{res}
$$

we can shrink down the second prediction so slow down the learning process param: 0.01; iterate the process until this process can overfit our training data

### params:

- n_estimators: number of learners(trees)
- learning_rate
- max_depth

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(boston.data, boston.target,random_state=0)

model = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=.01)
modelfit = model.fit(Xtrain,ytrain)
print(modelfit)
print(modelfit.score(Xtest, ytest))
```

Others

Gradianted boost trees, 

ada boost, using a weighting algorithm

xg boost, tune on tree size, we so far are building the same of trees, 

## Variable Importance Measure 

measure how RSS (for regressor) and Gini Index (for classifier) in proportion (so we can compare) reduced by each variable

```python
# Feature importance on a 0 to 1 normalized scale can be extracted from tree models:

print(formodel.feature_importances_) 
# added up to 1 so we can compare
#in same order as feature names in data
boston.feature_names # RM is most important variable in model.  Variable measures the Number of rooms.
```



## Feature Processing 

data preprocessing 

column transform approach

some categorical in numbers, be careful 



submit 

data.drop  clean up the dat

dont change random_state 



column transformer

replace missing data

simple imputer : deal with missing data, calculate and replace the missing data , the median or the average is the most common method, it will influence the coef of our model



pipeline steps, name, process

first deal with missing data

second scaler 



OnehotEncoder: create dummy variables encode the cate datasets

work in a team: how to share your best model



export 

import 

onnx save ml framework in the same format 



http://mlsite5aimodelshare-dev.s3-website.us-east-2.amazonaws.com/login

## Ensemble

The goal of **ensemble methods** is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator.

### voting classifier 

voting = 'hard' (just the majority vote)

or 'soft' (can get a probability, take the average of the majority vote)

 'soft' voting takes the predicted probabilities of each model, then chooses the label for predicted by the average predicted probabilities. If a model does not generated a predicted probability, then the probability is set to 1 for predicted class and zero otherwise.

```python
# 1. Build multiple classification models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#Generally works better if models you choose have diverse methodological approaches...

log_clf = LogisticRegression(random_state=42) # logistic regression w/ C=default
rnd_clf = RandomForestClassifier(random_state=42) # Random Forest
svm_clf = SVC(random_state=42) # support vector machine

# Goal is to predict ytest for each model and then use PREDICTIONS FROM EACH MODEL to select final predictions

# Need to set up a standard for selecting final prediction:
from sklearn.ensemble import VotingClassifier

# Estimator arge is giving each estimator a name for references in functions like GridsearchCV

# voting='hard' takes majority vote of each predicted value to select final prediction for ytest

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard') 

vmodel = voting_clf.fit(X_train, y_train)
print(vmodel.score(X_test, y_test)) #return accuracy of voting classifier  

#Need to ensure that probabilities are generated in each model...

log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft')
voting_clf.fit(X_train, y_train)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft') 

vmodel = voting_clf.fit(X_train, y_train)

print(vmodel.score(X_test, y_test)) #return accuracy of voting classifier
```

Voting Regressor 

taking the average of the results

## Feature Selection

we can use `coef_` or `feature_importances` and `feature_names` to select data

SelectFromModel is a meta-transformer that can be used along with any estimator that has a coef_ or feature_importances_ attribute after fitting.

The features are considered unimportant and removed, if the corresponding coef_ or feature_importances_ values are below the provided threshold parameter.

Apart from specifying the threshold numerically, there are built-in heuristics for finding a threshold using a string argument.

Available heuristics are “mean”, “median” and float multiples of these like “0.1*mean”.

Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.

Sometimes, we delete variable for prediction speed performance.

threshold: below that value of feature importance we can delete

```python
# Set a minimum threshold of 0.25
from sklearn.feature_selection import SelectFromModel
forest = RandomForestRegressor(n_estimators=200)
formodel = forest.fit(Xtrain, ytrain)
sfm = SelectFromModel(formodel, threshold=.25)
sfm.fit(Xtrain, ytrain)

Xtrain_new = sfm.transform(Xtrain) 
# transform data to insert into new model

print(Xtrain_new[0:5,:]) #only two variables in X now
print(Xtrain.shape) #compare to original data with 13 variables



lassomodel = Lasso(alpha=10).fit(Xtrain, ytrain)
model = SelectFromModel(lassomodel, prefit=True) # prefit argument allows non zero features to be chosen from regularized models like lasso    
X_new = model.transform(Xtrain) # transform data to insert into new model
print(lassomodel.coef_)
print(X_new.shape) #down to four variables from 13
```

In `sklearn,`given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of **recursive feature elimination (RFE)** is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a `coef_` attribute or through a feature`_importances_` attribute. Start with full model. Run series of models that evaluate prediction error on y_train after dropping a feature. Repeat for all features. Drop feature that is helps least in predicting y_train. recursive: delete one variable each time. Repeat process with n-1 features...

```python
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

estimator = LinearRegression().fit(Xtrain, ytrain) #model with all X variables
#EXAMPLE:  RFE to find 5 features that help model predict the best
selector = RFE(estimator, 5, step=1) # step tells RFE how many features to remove each time model features are evaluated

selector = selector.fit(Xtrain, ytrain) # fit RFE estimator.
print("Num Features: %d" % selector.n_features_)
print("Selected Features: %s" % selector.support_)# T/F for top five features
print("Feature Ranking: %s" % selector.ranking_)  # ranking for top five + features
# Transform X data for other use in this model or other models:
Xnew = selector.transform( Xtrain) #reduces X to subset identified above
Xnew.shape
```



## Model Evaluation Matrics

so far we use, accuracy for classification model, r square for regression model. now we learn matrices.

### Accuracy 

does not explain why we are wrong

use confusion to know positive negative matrix results

### Precision 

The precision is the ratio `tp / (tp + fp)` where `tp` is the number of true positives and `fp` the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

### Recall

The recall is the ratio `tp / (tp + fn)` where `tp` is the number of true positives and `fn` the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

### f1 score 

The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:
$$
F1 = 2 * (precision * recall) / (precision + recall)
$$
the one you generally want to maximize

<u>**classification use f1 score**</u>

<u>**for regression models, use MSE**</u>

the score we need to use when moving forward, which is more robust

can use accuracy for presentation

mean score error, MSE = rss/ number of observation, smaller is better

sometimes, `scikilearn` put a negative sign in front of it 

## Unsupervised Learning

- Learning from data without y, not interested in predicting Y
- Unsupervised learning used to be used only by visualization
- Can be harder to find the meaningful information
- More subjective 
- usefully to make groups 
- Unsupervised learning is important for understanding the variation and grouping structure of a set of unlabeled data, and can be a useful pre-processor for supervised learning

### Principal Components Analysis PCA

Reduce the dimensionality of large data sets; trade a little accuracy for simplicity 

PCA produces a low-dimensional representation of a dataset. It finds a sequence of linear combinations of the variables that have **maximal variance, and are mutually uncorrelated**.

Need to scale the variables (otherwise, variable with larger variance will be assigned with larger improtance)
$$
Z_i1 = \phi_{11}*X_i1+\phi_{21}*X_i2+...\phi_{p1}*X_ip
$$
It has the largest variance, subject to the constraint that 
$$
SUM_{1_p}\phi^2_{j1} =1
$$
sum of phi (called loadings) equals to 1, normalized, so it would be used to increase the variance 

The loading vector φ1 with elements φ11, φ21, . . . , φp1 defines a direction in feature space along which the data vary the most. If we project the n data points x1, . . . , xn onto this direction, the projected values are the principal component scores z11, . . . , zn1 themselves.

The second principal component is the linear combination of X1, . . . , Xp that has maximal variance among all linear combinations that are uncorrelated with Z1.

- dimensionality reduction ( reduce the number of features)
- from the entire x matrix, we want to extract something that is improtant

- x matrix of continuous variables, fit a Linear regression model,  


![2021-03-22](https://i.loli.net/2021/03/22/v5V8wHSYThz6pgc.png)

​                                                     *in graph, we measure the value along the vector*



#### Decide the number of vectors

we rely on the proportion of variance explained PVE

we can specify the variance we want to capture

![2021-03-22 (1)](https://i.loli.net/2021/03/22/mJ5s6Z1RFInQMqb.png)

​                                                                         *one way is to fin the "elbow”.*



```python
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

X = pd.DataFrame(scale(df), index=df.index, columns=df.columns)
PCA().fit(X).components_.T
# The loading vectors
pca_loadings = pd.DataFrame(PCA().fit(X).components_.T, index=df.columns, columns=['V1', 'V2', 'V3', 'V4'])
pca_loadings
# Fit the PCA model and transform X to get the principal components
pca = PCA()
df_plot = pd.DataFrame(pca.fit_transform(X), columns=['PC1', 'PC2', 'PC3', 'PC4'], index=X.index)
df_plot
# Standard deviation of the four principal components
np.sqrt(pca.explained_variance_)
pca.explained_variance_ratio_
```

### Clustering 

use Euclidean distance, can also use correlation-based distance 

Automatic labelling

e.g.clustering the genes to see subgroups' responses to treatment

at first, we have too much data,

clustering for customer, then marketing, you do not have a specific purpose, for consulting companies

#### K-means clustering

- Assign randomly 1 to k value to observations

- Aompute centroid : the average of each bucket 

- Assign each observation to the closet centroid


we want to minimize the value to a stopping point 

no guarantee we can reach a global minimum

we can look at our data in a new way, generate new ideas

![2021-03-22 (2)](C:\Users\14249\OneDrive\Pictures\Screenshots\2021-03-22 (2).png)

![2021-03-22 (3)](C:\Users\14249\OneDrive\Pictures\Screenshots\2021-03-22 (3).png)

need to specify K, can be an disadvantage 

#### Hierarchical Clustering

we need to scale the data, do not need a K param

we may get the same result as k-mean

bottom-up approach 

1. make every observation as a cluster
2. measure the distance between the clusters
3. find the closet two clusters and merge them

how do calculate the similarity -> linkage

- complete, largest of the dissimilarities
- single, smallest
- average, average 
- centroid (does not work well)

look for mean for clusters 

```python
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
# prep data
X = pd.DataFrame(scale(df), index=df.index, columns=df.columns)
km1 = KMeans(n_clusters=2, n_init=20)
km1.fit(X)

np.random.seed(4)
km2 = KMeans(n_clusters=3, n_init=20)
km2.fit(X)

# count in each cluster
pd.Series(km2.labels_).value_counts()
# Sum of distances of samples to their closest cluster center.
km2.inertia_
km2.cluster_centers 
km2.labels_

# Add new labels to original data and explore what clusters mean by evaluating column means.
X['cluster'] = km1.labels_
 # means for full scaled data
display(X.groupby('cluster').mean()) #cluster means
# Note that original means are zero centered, so you can compare these categories to a mean of zero for all columns in your 
# scaled data
# print groups by index values to see states
X.groupby('cluster').groups
```



## Imbalanced Data

if we have imbalanced data in different categories of dependent variable 

like default data of credit cards

Solutions: add samples to the small group, or remove from the large

change the dataset before fitting into our model

we may have only one category prediction if we do not deal with imbalanced data 

`Imblearn` -> pipeline 

`RandomOverSampler` 

One way to fight this issue is to generate new samples in the classes which are under-represented. The most naive strategy is to generate new samples by randomly sampling with replacement the current available samples. 

```python
conda install -c glemaitre imbalanced-learn

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X, y)

# Count each value of y

print(pd.Series(y_resampled).value_counts())

# The resampled model influences precision and recall for categories with imbalanced data

logreg = LogisticRegression().fit(X_resampled,y_resampled)
y_pred = logreg.predict(X_resampled) # y_pred includes your predictions

print(classification_report(y_resampled, y_pred)) # precision recall and f1
print(logreg.score(X_resampled, y_resampled)) #accuracy 
```

`RandomUnderSampler` 

Of course we lose a lot of data when we undersample. Better to oversample minority class while keeping the signal from the data, if possible.

```python
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)

X_resampled, y_resampled = rus.fit_sample(X, y)

print(pd.Series(y_resampled).value_counts()
```

`smart_sampling` 

### Synthetic Minority Oversampling Technique (SMOTE)

- choose k nearest neighbors from minority class
- take the difference between 
- multiply the result by rand(0,1)
- add back to minority samples

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(ratio='minority')
X_sm, y_sm = smote.fit_sample(X, y)

#Original data fit
logreg = LogisticRegression().fit(X,y)
y_pred = logreg.predict(X) # y_pred includes your predictions

print(classification_report(y, y_pred)) # precision recall and f1
print(logreg.score(X, y)) #accuracy 
```

Combination of over and under sampling 

Tomek links: two instances form a Tomek link then either one of these instances is noise or both are near a border. 

```python
# We can use SMOTE with a second technique that dismisses nearest
# neighbors that have diff't classes before creating synthetic observations (i.e.-delete all Tomek links)


from imblearn.combine import SMOTETomek

smt = SMOTETomek(ratio='minority')
X_smt, y_smt = smt.fit_sample(X, y)

logreg = LogisticRegression().fit(X_smt,y_smt)
y_pred = logreg.predict(X_smt) # y_pred includes your predictions

print(classification_report(y_smt, y_pred)) # precision recall and f1
print(logreg.score(X_smt, y_smt)) #accuracy increases a bit, but we should test on new data
```



## Manifold Learning

While PCA is flexible, fast, and easily interpretable, it does not perform so well when there are **nonlinear** relationships within the data.

Manifold learning is a class of unsupervised estimators that seeks to describe datasets as low-dimensional manifolds embedded in high-dimensional spaces.

reduce to two vectors to visualize data 

should not be used to draw conclusion 

if want to reduce dimensions, should use PCA

manifold, think as a piece of paper, but can be flexible 

### MDS

multidimensional scaling

all pairwise links 

find linear relationship, flat paper



```python
D2 = pairwise_distances(X2)
from sklearn.manifold import MDS
model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)

out = model.fit_transform(D2)

plt.scatter(out[:, 0], out[:, 1], **colorize)
plt.axis('equal');

```



### Nonlinear Manifolds

If you bend, curl, or crumple the paper, it is still a two-dimensional manifold, but the embedding into the three-dimensional space is no longer linear.

Local linear embedding LLE linkage 

nearest neighbor links



## Neural Network 

usually the best model for predication and commonly used in practice

`keras` (is now a part of `tensorflow`) and `tensorflow` 

weights

bias

activation: sigmoid (logit)

cost function MSE



### Multilayer Perceptron Model

can deal with nonlinear relationship 

neuron/node: N weights and 1 bias based on input from the upper layer

hidden layers 

output activation function soft for classification j

hidden layer use sigmoid or relu ....

cost function, use gradient descent to find weights and bias by subtracting gradients



### Set parameters for Network

**epochs:** the number of iterations we run to minimize error and update weights and biases during backpropagation

**inputLayerSize:** The number of features in X

**hiddenLayerSize:** How many hidden neurons will be in our hidden layer

**outputLayerSize:** The length of our output layer. For regression problems it is one. For binary categorization it is one (a predicted probability between zero and one). And for larger categories its the number of categories you will predict.

Other typical parameters-

learning rate: .01 or .001 typically. Used to multiply by gradient adjustments to weights and biases in backpropagation Will slow down the learning process.

number of hidden layers: We can have more than one hidden layer. Typically we experiment with two or three to find the best predictive model.



```python

```



### KERAS

use the old way in the lecture now a part of `tensorflow`

we do not need to specify input layer , just set up the input_shape parameter

sequential model 

dense linear, number of nerons

input_shape 

Activation 

Relu = max(0,x)

output if regression dense 1 output

if classification dense 1 (sigmoid) or more softness

summary() function to see how many parameters that we can train, last two rows about output 

### Optimizer 

- binary for sigmoid
- multi for softness
- mse for regression, output neuron 1 and no activation function 



we need to adjust our input data, two dummy np can help, 

keras cross validation, Wrapper for sklearn 

research CV, we need keras classifier / regressor, we need to write a function in python. 



to evaluate the model in deep learning, cv can take a long time



### Drop-out 

control overfitting

turn off neurons randomly, force the model to learn new path ways



## Convolution Neural Networks

for image data mainly, computer vision 

reducing the parameters is important for image data

### vertical edge detection

use filter to transfer input data to a low-dimensional output data

padding: we can control the output size, add rows of zeros to the input matrix

"valid" for no padding, "same" means the output and input would be the same size

### Strided Convolution 

to make output data smaller, skip the steps u applied the filter 



### Color Image (RGB)

multiple matrix (channels) red green blue 

volumns 

convolution on RGB is to add up the result from each color matrix(each filter for each input matrix)

we can use different filter to get different output 

the more filters we fit in maybe we can learn more from the input

then we have 3D output, we might more patterns 

max pooling take the 4*4 matrix choose the max value in each matrix 



Q
1. stick to the CV score/ f1 score  not test score 
2. use pd.get_dummies() to change the string y vector 