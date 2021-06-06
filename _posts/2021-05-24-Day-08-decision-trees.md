---
layout: post
title: Decision Trees - An in-depth intuition
subtitle: 'Usually mimics the human thinking ability while making a decision, so it is easy to understand.'
description: >-
  The logic behind the decision tree is to show a flow-chart/tree-like structure which is easy to visualize and extract information. With the help of a decision treee, we can predict the output by traversing down the nodes of tree based on the input feature. The idea is to find the pure leaf nodes(same values in the leaf node)
image: >-
  https://www.kepner-tregoe.com/default/assets/Image/Decisions%20Min.jpg
optimized_image: >-
  https://www.kepner-tregoe.com/default/assets/Image/Decisions%20Min.jpg
category: blog
tags:
  - basics
  - decision_trees
author: mowlanica
paginate: False
---

## What is a Decision tree..?
A Decision Tree is a supervised Machine learning technique used for both classification and regression problems. It is a tree like structure with nodes. The spilitting of branches depends up on a number of factors. The splitting of data into branches is done until it achieves a certain threshold value. A decision tree consists of the root nodes, children nodes, and leaf nodes.

Since the start of the lockdown, I have started watching a lot of movies. I follow somewhat similar steps to decide on which movie to watch.
![](https://miro.medium.com/max/897/1*5UGnZAGpI8Gd3wjk68cbQg.png)



#### Features of the Decision
* Easy to understand
* One of the most intuitive classifiers
* Helps to find the most significant attribute

### Parts of a Decision tree
1. **Nodes**: It is The point where the tree splits according to the value of some attribute/feature of the dataset
2. **Edges**: It directs the outcome of a split to the next node .There is an edge for each potential value of each of those attributes/features.
3. **Root**: This is the node where the first split takes place
4. **Leaves**: These are the terminal nodes that predict the outcome of the decision tree


## Learing the Decision Tree
We want to find a Decision Tree that is as short as possible(in depth) and which will correctly predict the output label given in the input features. It takes up too much space for systematic search and computationally expensive. Therefore, we'll search for the best node in a Greedy manner one at a time as follows:
1. Select the best attribute by Attribute Selection measures to split the records.(will be discussed in the next section)
2. Make the attribute a decision node and split dataa into smaller subset
3. Start building the tree by repeating the process recursively for each child until one of the following condition matches.
    1. All the tuples do have the same attribute values.
    2. There are no more of the attributes remaining
    3. No more instances.
![](https://editor.analyticsvidhya.com/uploads/384902_btay8n.jpg)

## Regression trees vs Classification trees

Both the trees work almost similar to each other, let’s look at the primary differences & similarity between classification and regression trees:

1. Regression trees are used when dependent variable is continuous. Classification trees are used when dependent variable is categorical.
2. In case of regression tree, the value obtained by terminal nodes in the training data is the mean response of observation falling in that region. Thus, if an unseen data observation falls in that region, we’ll make its prediction with mean value.
3. In case of classification tree, the value (class) obtained by terminal node in the training data is the mode of observations falling in that region. Thus, if an unseen data observation falls in that region, we’ll make its prediction with mode value.
4. Both the trees divide the predictor space (independent variables) into distinct and non-overlapping regions. For the sake of simplicity, you can think of these regions as high dimensional boxes or boxes.
5. Both the trees follow a top-down greedy approach known as recursive binary splitting. We call it as ‘top-down’ because it begins from the top of tree when all the observations are available in a single region and successively splits the predictor space into two new branches down the tree. It is known as ‘greedy’ because, the algorithm cares (looks for best variable available) about only the current split, and not about future splits which will lead to a better tree.
6. This splitting process is continued until a user defined stopping criteria is reached. For example: we can tell the the algorithm to stop once the number of observations per node becomes less than 50.
7. In both the cases, the splitting process results in fully grown trees until the stopping criteria is reached. But, the fully grown tree is likely to overfit data, leading to poor accuracy on unseen data. This bring ‘pruning’. Pruning is one of the technique used tackle overfitting. We’ll learn more about it in following section.

## Decision the best Decision attribute
1. Entropy and Information Gain
2. Gini Index
3. Gain Ratio
4. Reduction in Variance 
5. Chi-squared test

#### 1. Entropy and Information Gain
    Entropy is the measure of disturbance or degree of randomness in the system. It is a measure impurity.

Let *p*,*n* be the proportion of positive and negative samples in a set *S*. 
> Entropy(s)= -P(p)log2 P(p)- P(n) log2 P(n)

    Information gain is the measurement of changes in entropy value after the splitting/segmenting the dataset based on an attribute.It tells how much information a feature/attribute provides us.Based on the information gain value, splitting of the node and decision tree building is done.Decision tree always tries to maximize the value of the information gain, and a node/attribute having the highest value of the information gain is being split first.

Information gain can be calculated using the below formula:
> Information Gain= Entropy(S)- [(Weighted Avg) *Entropy(each feature)


#### 2. Gini Index
    Gini index is defined as a measure of impurity/ purity used while creating a decision tree in the CART(known as Classification and Regression Tree) algorithm. An attribute having a low Gini index value should be preferred in contrast to the high Gini index value.It only creates binary splits, and the CART algorithm uses the Gini index to create binary splits.

    Gini  says, if we select two items from a population at random then they must be of same class and probability for this is 1 if population is pure.
    1. It works with categorical target variable “Success” or “Failure”.
    2. It performs only Binary splits
    3. Higher the value of Gini higher the homogeneity.
    4. CART (Classification and Regression Tree) uses Gini method to create binary splits.

Gini index can be calculated using the below formula:

> Gini Index= 1- ∑jPj2

Where pj stands for the probability

**Steps**
1. Calculate gini for subnodes using above formula
2. Calculate gini for split  using weighted score of each node of that split
3. Compare all Gini values for split and choose the least value.

**Example**: – Referring to example used above, where we want to segregate the students based on target variable ( playing cricket or not ). In the snapshot below, we split the population using two input variables Gender and Class. Now, I want to identify which split is producing more homogeneous sub-nodes using Gini .

![](https://www.analyticsvidhya.com/wp-content/uploads/2015/01/Decision_Tree_Algorithm1.png)

**Split on Gender**:

1. Calculate, Gini for sub-node Female = (0.2)*(0.2)+(0.8)*(0.8)=0.68
2. Gini for sub-node Male = (0.65)*(0.65)+(0.35)*(0.35)=0.55
3. Calculate weighted Gini for Split Gender = (10/30)*0.68+(20/30)*0.55 = 0.59

**Similar for Split on Class**:

1. Gini for sub-node Class IX = (0.43)*(0.43)+(0.57)*(0.57)=0.51
2. Gini for sub-node Class X = (0.56)*(0.56)+(0.44)*(0.44)=0.51
3. Calculate weighted Gini for Split Class = (14/30)*0.51+(16/30)*0.51 = 0.51

Above, you can see that Gini score for Split on Gender is higher than Split on Class, hence, the node split will take place on Gender.

You might often come across the term `Gini Impurity` which is determined by subtracting the gini value from 1. So mathematically we can say,

> Gini Impurity = 1-Gini


#### 3. Gain Ratio
    A notable problem occurs when information gain is applied to attributes that can take on a large number of distinct values i.e., biased towards attributes with major values. Gain ratio overcomes this problem with Information Gain by taking into account the number of branches that would result before making the split.

For example, if we have a data of users and movie genre preferences based on variables like gender, group of age, rating etc. With the help of information gain, we split at *Gender*(having high information gain), group of age and rating also could be equally important and with the help of gain ratio, it'll penalize a variable with more distinct value which help us decide the split at next level.

![](https://miro.medium.com/max/956/0*0c-qonaHUWmSepYJ.png)


#### 4. Reduction in Variance
    It is used for continuous target variables (Regrssion). It uses variance formula to make split. Split with low variance is selected.

![](https://assets.website-files.com/5e6f9b297ef3941db2593ba1/5f5a28839f1e555eca23ce5b_Screenshot%202020-09-10%20at%2015.21.49.png)

Above X-bar is mean of the values, X is actual and n is number of values.

**Steps to calculate Variance**:

1. Calculate variance for each node.
2. Calculate variance for each split as weighted average of each node variance.

**Example**:- Let’s assign numerical value 1 for play cricket and 0 for not playing cricket. Now follow the steps to identify the right split:

1. Variance for Root node, here mean value is (15*1 + 15*0)/30 = 0.5 and we have 15 one and 15 zero. Now variance would be ((1-0.5)^2+(1-0.5)^2+….15 times+(0-0.5)^2+(0-0.5)^2+…15 times) / 30, this can be written as (15*(1-0.5)^2+15*(0-0.5)^2) / 30 = 0.25
2. Mean of Female node =  (2*1+8*0)/10=0.2 and Variance = (2*(1-0.2)^2+8*(0-0.2)^2) / 10 = 0.16
3. Mean of Male Node = (13*1+7*0)/20=0.65 and Variance = (13*(1-0.65)^2+7*(0-0.65)^2) / 20 = 0.23
4. Variance for Split Gender = Weighted Variance of Sub-nodes = (10/30)*0.16 + (20/30) *0.23 = 0.21
5. Mean of Class IX node =  (6*1+8*0)/14=0.43 and Variance = (6*(1-0.43)^2+8*(0-0.43)^2) / 14= 0.24
6. Mean of Class X node =  (9*1+7*0)/16=0.56 and Variance = (9*(1-0.56)^2+7*(0-0.56)^2) / 16 = 0.25
7. Variance for Split Gender = (14/30)*0.24 + (16/30) *0.25 = 0.25
Above, you can see that Gender split has lower variance compare to parent node, so the split would take place on Gender variable.

**Steps**:
1. Calculate variance of each node.
2. Calculate variance for each split as weighted average of each node variance.


#### 5. Chi-Squared test
    It is an algorithm to find out the statistical significance between the differences between sub-nodes and parent node. We measure it by sum of squares of standardized differences between observed and expected frequencies of target variable.
    1. It works with categorical target variable “Success” or “Failure”.
    2. It can perform two or more splits.
    3. Higher the value of Chi-Square higher the statistical significance of differences between sub-node and Parent node.
    4. Chi-Square of each node is calculated using formula,
    5. Chi-square = ((Actual – Expected)^2 / Expected)^1/2
    6. It generates tree called CHAID (Chi-square Automatic Interaction Detector)
![](http://ai-ml-analytics.com/wp-content/uploads/2020/07/chi-square.png)

**Steps to Calculate Chi-square for a split**:
1. Calculate Chi-square for individual node by calculating the deviation for Success and Failure both
2. Calculated Chi-square of Split using Sum of all Chi-square of success and Failure of each node of the split

**Example**: Let’s work with above example that we have used to calculate Gini.

**Split on Gender**:

1. First we are populating for node Female, Populate the actual value for “Play Cricket” and “Not Play Cricket”, here these are 2 and 8 respectively.
2. Calculate expected value for “Play Cricket” and “Not Play Cricket”, here it would be 5 for both because parent node has probability of 50% and we have applied same probability on Female count(10).
3. Calculate deviations by using formula, Actual – Expected. It is for “Play Cricket” (2 – 5 = -3) and for “Not play cricket” ( 8 – 5 = 3).
4. Calculate Chi-square of node for “Play Cricket” and “Not Play Cricket” using formula with formula, = ((Actual – Expected)^2 / Expected)^1/2. You can refer below table for calculation.
5. Follow similar steps for calculating Chi-square value for Male node.
6. Now add all Chi-square values to calculate Chi-square for split Gender.
![](https://www.analyticsvidhya.com/wp-content/uploads/2015/01/Decision_Tree_Chi_Square1.png)

**Split on Class**:

Perform similar steps of calculation for split on Class and you will come up with below table.
![](https://www.analyticsvidhya.com/wp-content/uploads/2015/01/Decision_Tree_Chi_Square_2.png)
Above, you can see that Chi-square also identify the Gender split is more significant compare to Class.




## Decision Tree Classifier Implementation
#### Import tools


```python
import numpy as np
import pandas as pd
```

#### Get the data


```python
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
data = pd.read_csv("iris.csv", skiprows=1, header=None, names=col_names)
data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.4</td>
      <td>3.9</td>
      <td>1.7</td>
      <td>0.4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.6</td>
      <td>3.4</td>
      <td>1.4</td>
      <td>0.3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5.0</td>
      <td>3.4</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.4</td>
      <td>2.9</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4.9</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### Node class


```python
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value
```

#### Tree class


```python
class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree ''' 
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        ''' function to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
```

#### Train-Test split


```python
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
```

#### Fit the model


```python
classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
classifier.fit(X_train,Y_train)
classifier.print_tree()
```

    X_2 <= 1.9 ? 0.33741385372714494
     left:0.0
     right:X_3 <= 1.5 ? 0.427106638180289
      left:X_2 <= 4.9 ? 0.05124653739612173
        left:1.0
        right:2.0
      right:X_2 <= 5.0 ? 0.019631171921475288
        left:X_1 <= 2.8 ? 0.20833333333333334
            left:2.0
            right:1.0
        right:2.0
    

#### Test the model


```python
Y_pred = classifier.predict(X_test) 
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, Y_pred)
```
    0.9333333333333333





## Decision Tree Regressor
    Regression trees are used when dependent variable is continuous

## Decision Tree Regressor Implementation
#### Import tools


```python
import numpy as np
import pandas as pd
```

#### Get the data


```python
data = pd.read_csv("airfoil_noise_data.csv")
data.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x0</th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>800</td>
      <td>0.0</td>
      <td>0.3048</td>
      <td>71.3</td>
      <td>0.002663</td>
      <td>126.201</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000</td>
      <td>0.0</td>
      <td>0.3048</td>
      <td>71.3</td>
      <td>0.002663</td>
      <td>125.201</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1250</td>
      <td>0.0</td>
      <td>0.3048</td>
      <td>71.3</td>
      <td>0.002663</td>
      <td>125.951</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1600</td>
      <td>0.0</td>
      <td>0.3048</td>
      <td>71.3</td>
      <td>0.002663</td>
      <td>127.591</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000</td>
      <td>0.0</td>
      <td>0.3048</td>
      <td>71.3</td>
      <td>0.002663</td>
      <td>127.461</td>
    </tr>
  </tbody>
</table>
</div>



## Node class


```python
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        
        # for leaf node
        self.value = value
```

#### Tree class


```python
class DecisionTreeRegressor():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree '''
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        best_split = {}
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["var_red"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["var_red"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_var_red = -float("inf")
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    # update the best split if needed
                    if curr_var_red>max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def variance_reduction(self, parent, l_child, r_child):
        ''' function to compute variance reduction '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction
    
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        val = np.mean(Y)
        return val
                
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.var_red)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
        
    def make_prediction(self, x, tree):
        ''' function to predict new dataset '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    
    def predict(self, X):
        ''' function to predict a single data point '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
```

#### Train-Test split


```python
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
```

#### Fit the model


```python
regressor = DecisionTreeRegressor(min_samples_split=3, max_depth=3)
regressor.fit(X_train,Y_train)
regressor.print_tree()
```

    X_0 <= 3150.0 ? 7.132048702017748
     left:X_4 <= 0.033779199999999995 ? 3.5903305690676675
      left:X_3 <= 55.5 ? 1.1789899981318328
        left:X_4 <= 0.00251435 ? 1.614396721819876
            left:128.9919833333333
            right:125.90953579676673
        right:X_1 <= 15.4 ? 2.2342245360792994
            left:129.39160280373832
            right:123.80422222222222
      right:X_0 <= 1250.0 ? 9.970884020498875
        left:X_4 <= 0.0483159 ? 6.355275159824863
            left:124.38024528301887
            right:118.30039999999998
        right:X_3 <= 39.6 ? 5.036286657241022
            left:113.58091666666667
            right:118.07284615384614
     right:X_4 <= 0.00146332 ? 29.082992105065273
      left:X_0 <= 8000.0 ? 11.886497073996967
        left:X_2 <= 0.0508 ? 7.608945827689513
            left:134.04247500000002
            right:127.33581818181818
        right:X_4 <= 0.00076193 ? 10.622919322400815
            left:128.94078571428574
            right:122.4076875
      right:X_4 <= 0.022902799999999997 ? 5.638575922510647
        left:X_0 <= 6300.0 ? 5.985051045988911
            left:120.04740816326529
            right:114.67370491803278
        right:X_4 <= 0.0368233 ? 8.63874479304644
            left:113.83169565217393
            right:107.6395833333333
    

#### Test the model


```python
Y_pred = regressor.predict(X_test) 
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(Y_test, Y_pred))
```




    4.851358097184457




## How to prevent overfitting.?
Overfitting is one of the key challenges faced while using decision tree algorithms. If there is no limit set of a decision tree, it will give you 100% accuracy on training set because in the worse case it will end up making 1 leaf for each observation. Thus, preventing overfitting is pivotal while modeling a decision tree and it can be done in 2 ways:
1. Setting constraints on tree size
2. Tree pruning

Let’s discuss both of these briefly.

### Setting Constraints on tree based algorithms
This can be done by using various parameters which are used to define a tree. First, lets look at the general structure of a decision tree:

![](https://www.analyticsvidhya.com/wp-content/uploads/2016/02/tree-infographic.png)

The parameters used for defining a tree are further explained below. The parameters described below are irrespective of tool. It is important to understand the role of parameters used in tree modeling. These parameters are available in R & Python.

1. Minimum samples for a node split
- Defines the minimum number of samples (or observations) which are required in a node to be considered for splitting.
- Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
- Too high values can lead to under-fitting hence, it should be tuned using CV.

2. Minimum samples for a terminal node (leaf)
- Defines the minimum samples (or observations) required in a terminal node or leaf.
- Used to control over-fitting similar to min_samples_split.
- Generally lower values should be chosen for imbalanced class problems because the regions in which the minority class will be in majority will be very small.

3. Maximum depth of tree (vertical depth)
- The maximum depth of a tree.
- Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
- Should be tuned using CV.
4. Maximum number of terminal nodes
- The maximum number of terminal nodes or leaves in a tree.
- Can be defined in place of max_depth. Since binary trees are created, a depth of ‘n’ would produce a maximum of 2^n leaves.
5. Maximum features to consider for split
- The number of features to consider while searching for a best split. These will be randomly selected.
- As a thumb-rule, square root of the total number of features works great but we should check upto 30-40% of the total number of features.
- Higher values can lead to over-fitting but depends on case to case.
 
### Pruning in tree based algorithms
As discussed earlier, the technique of setting constraint is a greedy-approach. In other words, it will check for the best split instantaneously and move forward until one of the specified stopping condition is reached. Let’s consider the following case when you’re driving:

![](https://www.analyticsvidhya.com/wp-content/uploads/2016/04/graphic-1024x317.png)

There are 2 lanes:
1. A lane with cars moving at 80km/h
2. A lane with trucks moving at 30km/h

At this instant, you are the yellow car and you have 2 choices:
1. Take a left and overtake the other 2 cars quickly
2. Keep moving in the present lane

Let’s analyze these choice. In the former choice, you’ll immediately overtake the car ahead and reach behind the truck and start moving at 30 km/h, looking for an opportunity to move back right. All cars originally behind you move ahead in the meanwhile. This would be the optimum choice if your objective is to maximize the distance covered in next say 10 seconds. In the later choice, you sale through at same speed, cross trucks and then overtake maybe depending on situation ahead. Greedy you!

This is exactly the difference between normal decision tree & pruning. A decision tree with constraints won’t see the truck ahead and adopt a greedy approach by taking a left. On the other hand if we use pruning, we in effect look at a few steps ahead and make a choice.

So we know pruning is better. But how to implement it in decision tree? The idea is simple.

1. We first make the decision tree to a large depth.
2. Then we start at the bottom and start removing leaves which are giving us negative returns when compared from the top.
3. Suppose a split is giving us a gain of say -10 (loss of 10) and then the next split on that gives us a gain of 20. A simple decision tree will stop at step 1 but in pruning, we will see that the overall gain is +10 and keep both leaves.

## Most frequently asked questions in Decision Trees
<style>
.mylist.type1 > span {
  display: list-item;
  /* inside: the marker is placed inside the element box before its content */
  list-style-position: inside;
}
.mylist.type2 > span {
  display: list-item;
  /* outside (default): the marker is placed outside the element box towards left */
  /* by the way you are not restricted to just bullets */
  list-style-type: lower-roman;
}
.mylist.type2 {
  /* add some room on left for bullets positioned outside */
  padding-left: 2em;
}
</style>
<link rel="stylesheet" href="//code.jquery.com/ui/1.10.4/themes/smoothness/jquery-ui.css">
<script src="//code.jquery.com/jquery-1.10.2.js"></script>
<script src="//code.jquery.com/ui/1.10.4/jquery-ui.js"></script>

<script>
$(function() {
$( ".accordion" ).accordion();
$(".accordion").accordion({ header: "h3", collapsible: true, active: false ,heightStyle: "content" });
});
</script>



<div class="accordion">
<h3>1. Advantages and Disadvantages of Decision tree</h3>
   <div>
   <p>
   <b> Advantages </b>
   <div class="mylist type1">
   <span>It is simple to implement and it follows a flow chart type structure that resembles human-like decision making.</span>
   <span>It proves to be very useful for decision-related problems.</span>
   <span>It helps to find all of the possible outcomes for a given problem.</span>
   <span>There is very little need for data cleaning in decision trees compared to other Machine Learning algorithms.</span>
   <span>Handles both numerical as well as categorical values</span>
    <br>
   <b> Disadvantages </b>
   <span>Too many layers of decision tree make it extremely complex sometimes.</span>
    <span>It may result in overfitting ( which can be resolved using the Random Forest algorithm) due to the amount of specificity we look at, leading to smaller sample of events.</span>
    <span>For the more number of the class labels, the computational complexity of the decision tree increases.</span>
    <span>Gives optimal solution but not globally optimal solution.</span>
    <span>Cannot explain the marginal effect</span>
    </p>
   </div>
   </div>
<h3>2. Is Decision tree(tree based algorithms) better than linear models?</h3>
   <div>
     <p><i>“If I can use logistic regression for classification problems and linear regression for regression problems, why is there a need to use trees?”</i> Many of us have this question. And, this is a valid one too.<br>
     Actually, you can use any algorithm. It is dependent on the type of problem you are solving. Let’s look at some key factors which will help you to decide which algorithm to use:
     <div class = "mylist type2">
     <span>If the relationship between dependent & independent variable is well approximated by a linear model, linear regression will outperform tree based model.</span>
     <span>If there is a high non-linearity & complex relationship between dependent & independent variables, a tree model will outperform a classical regression method.</span>
     <span>If you need to build a model which is easy to explain to people, a decision tree model will always do better than a linear model. Decision tree models are even simpler to interpret than linear regression!</span></p>
    </div>
   </div>
<h3>3. Are Decision Trees Robust to Outliers</h3>
   <div>
     <p>Yes. Because decision trees divide items by lines, so it does not difference how far is a point from lines.
     Most likely outliers will have a negligible effect because the nodes are determined based on the sample proportions in each split region (and not on their absolute values).<br>
     However, different implementations to choose split points of continuous variables exist. Some consider all possible split points, others percentiles. But, in some poorly chosen cases (e.g. dividing the range between min and max in equidistant split points), outliers might lead to sub-optimal split points. But you shouldn't encounter these scenarios in popular implementations.<br>
     <img src ="https://i.stack.imgur.com/OBSfq.png"> <img src = "https://i.stack.imgur.com/OBSfq.png">
     </p>
   </div>
<h3>4. Decision Tree Classifier with Majority vote (Is 50% a majority vote)</h3>
   <div>
     <p>The goal of ID3 is to get the purest nodes possible ( ironically that is what contributes to its problem of overfitting), so 50% is not pure at all, the data under that node is equally likely to be in one of the classes which makes peedicition tricky, it would be better to grow the tree further and find nodes which are more pure than atleast 50%.</p>
   </div>
<h3>5. Why do decision trees have low accuracy?</h3>
   <div>
     <p>First a common misconception, Decision trees are deterministic and extremely greedy. A random forest is not a decision tree, it as an ensemble of decision trees selected in a way to avoid the potential pitfall of a decision tree.<br>
     In wikipedia
    `&#09;` They are often relatively inaccurate. Many other predictors perform better with similar data. This can be remedied by replacing a single decision tree with a random forest of decision trees...
    Because they are greedy and deterministic if you add one row more or take one out the result can be different, also that they tend to overfit. That is my understanding of low accuracy in this sentence.<br>
    In elements of statistical learning
    `&#09;` Trees have one aspect that prevents them from being the ideal tool for predictive learning, namely inaccuracy. They seldom provide predictive accuracy comparable to the best that can be achieved with the data at hand. As seen in Section 10.1, boosting decision trees improves their accuracy, often dramatically. <br>
    Because they are greedy and deterministic they don't normally give their best result. That is why random forest and gradient boosting appeared and they are extremely good. They replace this pitfall of decision trees.<br><br>
    In short your question is right, and that problem has been solved historically with random forest and gradient boosting.</p>
   </div>
<h3>6. How do decision tree learning algorithms deal with missing values (under the hood)..?</h3>
   <div>
     <p>There are several methods used by various decision trees. Simply ignoring the missing values (like ID3 and other old algorithms does) or treating the missing values as another category (in case of a nominal feature) are not real handling missing values. However those approaches were used in the early stages of decision tree development.<br>The real handling approaches to missing data does not use data point with missing values in the evaluation of a split. However, when child nodes are created and trained, those instances are distributed somehow.<br><br>Following are the approaches to distribute the missing value instances to child nodes:<br>
     <div class="mylist type1">
     <span>All goes to the node which already has the biggest number of instances (CART, is not the primary rule)</span>
     <span>Distribute to all children, but with diminished weights, proportional with the number of instances from each child node (C45 and others)</span><span>Distribute randomly to only one single child node, eventually according with a categorical distribution (I have seen that in various implementations of C45 and CART for a faster running time)</span>
     <span>Build, sort and use surrogates to distribute instances to a child node, where surrogates are input features which resembles best how the test feature send data instances to left or right child node (CART, if that fails, the majority rule is used)</span>
     </p>
   </div>
</div>
<h3>7. What are the scenarios where Decision Tree works well..?</h3>
   <div>
     <p>Answer goes here</p>
   </div>
<h3>8. Why does Decision Tree have Low Bias And High Variance..?</h3>
   <div>
     <p>Answer goes here</p>
   </div>
</div>
<h3>9. What are the Hyperparameter Techniques for Decision tree..?</h3>
   <div>
     <p>Answer goes here</p>
   </div>
<h3>10. What are the libraries used for constructing decision tree..?</h3>
   <div>
     <p>Answer goes here</p>
   </div>
</div>
<h3>11. Why are we growing decision trees via entropy instead of the classification error?</h3>
   <div>
     <p><a href ="https://github.com/rasbt/python-machine-learning-book/blob/master/faq/decisiontree-error-vs-entropy.md">>Find the answer here<a></p>
   </div>
<h3>12. What are the disadvantages of using classic decision tree algorithm for a large dataset?</h3>
   <div>
     <p><a href ="https://github.com/rasbt/python-machine-learning-book/blob/master/faq/decision-tree-disadvantages.md">>Find the answer here<a></p>
   </div>
<h3>13. Why are implementations of decision tree algorithms usually binary and what are the advantages of the different impurity metrics?</h3>
   <div>
     <p><a href ="https://github.com/rasbt/python-machine-learning-book/blob/master/faq/decision-tree-binary.md">>Find the answer here<a></p>
   </div>
</div>
---

## Next Steps
* Perform Pruning on thr Decision Trees

### References
* https://www.analyticsvidhya.com/blog/2021/04/beginners-guide-to-decision-tree-classification-using-python/
* https://www.analyticsvidhya.com/blog/2016/04/tree-based-algorithms-complete-tutorial-scratch-in-python/
* http://bit.ly/normalizedNERD  - Highly recommend this channel
