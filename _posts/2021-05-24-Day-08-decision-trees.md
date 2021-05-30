---
layout: post
title: Decision Trees - An in-depth intuition
subtitle: 'Usually mimics the human thinking ability while making a decision, so it is easy to understand.'
description: >-
  The logic behind the decision tree is to show a flow-chart/tree-like structure which is easy to visualize and extract information. With the help of a decision treee, we can predict the output by traversing down the nodes of tree based on the input feature. The idea is to find the pure leaf nodes(same values in the leaf node)
image: >-
  https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.kepner-tregoe.com%2Fblog%2Fthe-consequences-of-choice-the-final-step-in-decision-making%2F&psig=AOvVaw2a_6KKQoPy8RyLEYmOlWLL&ust=1622460380772000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCNDYjKCm8fACFQAAAAAdAAAAABAD
optimized_image: >-
  https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.kepner-tregoe.com%2Fblog%2Fthe-consequences-of-choice-the-final-step-in-decision-making%2F&psig=AOvVaw2a_6KKQoPy8RyLEYmOlWLL&ust=1622460380772000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCNDYjKCm8fACFQAAAAAdAAAAABAD
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
|:--:| 
| *![Source](https://miro.medium.com/max/897/1*5UGnZAGpI8Gd3wjk68cbQg.png)* |


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

## Decision the best Decision attribute
1. Entropy and Information Gain
2. Gini Index
3. Gain Ratio
4. Reduction in Variance 
5. Chi-squared test

#### 1. Entropy and Information Gain
`Entropy is the measure of disturbance or degree of randomness in the system. It is a measure impurity.`

Let *p*,*n* be the proportion of positive and negative samples in a set *S*. 
> Entropy(s)= -P(p)log2 P(p)- P(n) log2 P(n)

`Information gain is the measurement of changes in entropy value after the splitting/segmenting the dataset based on an attribute.It tells how much information a feature/attribute provides us.Based on the information gain value, splitting of the node and decision tree building is done.Decision tree always tries to maximize the value of the information gain, and a node/attribute having the highest value of the information gain is being split first.` 

Information gain can be calculated using the below formula:
> Information Gain= Entropy(S)- [(Weighted Avg) *Entropy(each feature)`


#### 2. Gini Index
`Gini index is defined as a measure of impurity/ purity used while creating a decision tree in the CART(known as Classification and Regression Tree) algorithm. An attribute having a low Gini index value should be preferred in contrast to the high Gini index value.It only creates binary splits, and the CART algorithm uses the Gini index to create binary splits.`

Gini index can be calculated using the below formula:

> Gini Index= 1- ∑jPj2

Where pj stands for the probability


#### 3. Gain Ratio


#### 4. Reduction in Variance


#### 5. Chi-Squared test


## Decision Tree Classifier Implementation
## Import tools


```python
import numpy as np
import pandas as pd
```

## Get the data


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



## Node class


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

## Tree class


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

## Train-Test split


```python
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
```

## Fit the model


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
    

## Test the model


```python
Y_pred = classifier.predict(X_test) 
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, Y_pred)
```




    0.9333333333333333





## Decision Tree Regressor



## Decision Tree Regressor Implementation


## Most frequently asked questions in Decision Trees

1. Advantages and Disadvantages of Decision tree
1 It is simple to implement and it follows a flow chart type structure that resembles human-like decision making.

2 It proves to be very useful for decision-related problems.

3 It helps to find all of the possible outcomes for a given problem.

4 There is very little need for data cleaning in decision trees compared to other Machine Learning algorithms.

5 Handles both numerical as well as categorical values

7. Disadvantages of the Decision Tree
1 Too many layers of decision tree make it extremely complex sometimes.

2 It may result in overfitting ( which can be resolved using the Random Forest algorithm)

3 For the more number of the class labels, the computational complexity of the decision tree increases.




---

### References
* https://www.analyticsvidhya.com/blog/2021/04/beginners-guide-to-decision-tree-classification-using-python/
* 
