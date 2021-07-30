##########################################################################################################
# Author: Gene Lee
# CS 540 (Summer 2021)
########################################################################################################## 

import csv
import math
import numpy as np

############################################# HELPER CLASSES #############################################
class Node:
    def __init__(self, feature, threshold, head=None):
        self.feature = str(feature)
        self.threshold = str(threshold)
        self.parent = head
        self.children = []
        self.left = None
        self.right = None

        if head:
            head.children.append(self)

    def get_feature(self):
        if 'label' in self.feature: return self.feature
        else: return int(self.feature)

    def get_threshold(self):
        return int(self.threshold)

    def get_parent(self):
        return self.parent

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def set_left(self, left_node):
        self.left = left_node

    def set_right(self, right_node):
        self.right = right_node

    def print(self):
        print(f"feature = {self.feature}\t threshold = {self.threshold}")

############################################ HELPER FUNCTIONS ############################################
def entropy_calc(data):
    n = len(data)
    benign = sum(b[-1] == 2 for b in data)
    malign = n - benign

    if benign == 0 or malign == 0: return 0 # avoid log2(0) error

    return -(benign/n) * math.log2(benign/n) - (malign/n) * math.log2(malign/n)

def info_gain_calc(data, t, feature_index):
    n = len(data)
    below_t = []
    above_t = []
    for x in data:
        if x[feature_index] <= t: below_t.append(x)
        else: above_t.append(x)

    if len(below_t) == 0 or len(above_t) == 0: return 0

    return entropy_calc(data) - (len(below_t) / n ) * entropy_calc(below_t) - (len(above_t) / n) * entropy_calc(above_t)

def find_split(data, features, parent):
    benign = [row for row in data if row[-1] == 2]
    malign = [row for row in data if row[-1] == 4]

    # stop splitting when the subset's labels are all the same
    if len(benign) == 0: 
        return ('label 4', -1)
    if len(malign) == 0:
        return ('label 2', -1)

    max_gains = [0] * len(features)
    max_t = [0] * len(features)

    for i in range(len(features)): # iterate over all features
        feature_index = features[i] - 1

        for t in range(0, 11): # iterate over all possible thresholds
            info_gain = info_gain_calc(data, t, feature_index)

            if info_gain > max_gains[i]: # check if current information gain is more than the current max gain
                max_gains[i] = info_gain
                max_t[i] = t

    # find max info gain
    max_info_gain = max(max_gains)

    if max_info_gain == 0: # stop splitting when max info gain is 0
       if len(benign) / len(data) >= 0.5: return ('label 2', -1)
       else: return ('label 4', -1)

    max_index = max_gains.index(max_info_gain) # index of feature to split at
    split_feature = features[max_index] # feature number
    split_threshold = max_t[max_index] # threshold

    return (split_feature, split_threshold)

def train_tree(data, features, parent): # code modified from TA's solution
    parent_f = parent.get_feature()
    parent_t = parent.get_threshold()

    below_t, above_t = [], []
    for row in data: # split data set based on parent feature and threshold
        if row[parent_f-1] <= parent_t: below_t.append(row)
        else: above_t.append(row)
    below_t = np.array(below_t)
    above_t = np.array(above_t)

    left_f, left_t = find_split(below_t, features, parent)
    right_f, right_t = find_split(above_t, features, parent)

    parent.set_left(Node(left_f, left_t, parent))
    parent.set_right(Node(right_f, right_t, parent))

    if left_t != -1: train_tree(below_t, features, parent.get_left())
    if right_t != -1: train_tree(above_t, features, parent.get_right())

def print_tree_structure(root, f, prefix=""): # code modified from TA's solution
    fea = root.get_feature()
    t = root.get_threshold()
    l = root.get_left()
    l_fea = l.get_feature()
    r = root.get_right()
    r_fea = r.get_feature()

    if "label" in str(l_fea):
        f.write(prefix+'if (x'+str(fea)+' <= '+str(t)+') return '+l_fea[-1]+'\n')
    else:
        f.write(prefix+'if (x'+str(fea)+' <= '+str(t)+')\n')
        print_tree_structure(l, f, prefix+' ')
    if "label" in str(r_fea):
        f.write(prefix+'else return '+r_fea[-1]+'\n')
    else:
        f.write(prefix+'else\n')
        print_tree_structure(r, f, prefix+' ')

def print_tree_code(root, f, prefix=""): # code modified from TA's solution
    fea = root.get_feature()
    t = root.get_threshold()
    l = root.get_left()
    l_fea = l.get_feature()
    r = root.get_right()
    r_fea = r.get_feature()

    if "label" in str(l_fea):
        f.write(prefix+'if test_data['+str(fea-1)+'] <= '+str(t)+': return '+l_fea[-1]+'\n')
    else:
        f.write(prefix+'if test_data['+str(fea-1)+'] <= '+str(t)+':\n')
        print_tree_code(l, f, prefix+'    ')
    if "label" in str(r_fea):
        f.write(prefix+'else: return '+r_fea[-1]+'\n')
    else:
        f.write(prefix+'else:\n')
        print_tree_code(r, f, prefix+'    ')

def classify_q5(test_data):
    if test_data[2] <= 2:
        if test_data[6] <= 3:
            if test_data[5] <= 4: return 2
            else:
                if test_data[3] <= 1: return 2
                else: return 4
        else:
            if test_data[8] <= 1:
                if test_data[6] <= 5:
                    if test_data[5] <= 1: return 4
                    else: return 2
                else:
                    if test_data[3] <= 1: return 2
                    else: return 4
            else: return 4
    else:
        if test_data[2] <= 4:
            if test_data[6] <= 3:
                if test_data[8] <= 1: return 2
                else:
                    if test_data[8] <= 8:
                        if test_data[8] <= 5:
                            if test_data[5] <= 3:
                                if test_data[6] <= 2:
                                    if test_data[2] <= 3: return 2
                                    else:
                                        if test_data[6] <= 1:
                                            if test_data[6] <= 0:
                                                if test_data[3] <= 3: return 2
                                                else: return 4
                                            else: return 2
                                        else: return 4
                                else:
                                    if test_data[3] <= 2: return 2
                                    else: return 4
                            else: return 4
                        else: return 2
                    else: return 4
            else:
                if test_data[3] <= 4:
                    if test_data[6] <= 7:
                        if test_data[5] <= 3:
                            if test_data[9] <= 1: return 4
                            else: return 2
                        else: return 2
                    else:
                        if test_data[5] <= 6:
                            if test_data[9] <= 2: return 4
                            else:
                                if test_data[8] <= 4: return 4
                                else: return 2
                        else:
                            if test_data[5] <= 7: return 2
                            else: return 4
                else: return 4
        else:
            if test_data[6] <= 0:
                if test_data[5] <= 2: return 4
                else: return 2
            else:
                if test_data[6] <= 8:
                    if test_data[5] <= 5:
                        if test_data[8] <= 7:
                            if test_data[5] <= 4:
                                if test_data[8] <= 6: return 4
                                else:
                                    if test_data[2] <= 8:
                                        if test_data[9] <= 1: return 2
                                        else: return 4
                                    else: return 4
                            else:
                                if test_data[2] <= 9: return 2
                                else: return 4
                        else: return 4
                    else: return 4
                else: return 4

def search_tree(node, data):
    fea = node.get_feature()
    t = node.get_threshold()

    if "label" in str(fea): return fea[-1] # return 2 or 4 when leaf is reached

    # traverse left subtree if less than or equal to threshold, otherwise traverse right subtree
    if data[fea-1] <= t: return search_tree(node.get_left(), data) 
    else: return search_tree(node.get_right(), data) 

def classify_accuracy(root, test_data, test_data_labels):
    labels = [0]*len(test_data)
    correct = 0

    for i in range(len(test_data)):
        labels[i] = search_tree(root, test_data[i]) # find tree classification of test instance
        if labels[i] == str(test_data_labels[i]): # check accuracy of label
            correct = correct + 1
    
    return round(correct/len(test_data), 4) # return accurcy of classification
    
def find_max_label(root, stop_node, training_data):
    fea = root.get_feature()
    t = root.get_threshold()

    if "label" in str(fea): return 0 # if a leaf node is reached, return 0

    if root == stop_node: # if the target node is reached, return its majority label
        label_2 = 0
        label_4 = 0
        for data in training_data:
            if data[-1] == 2: label_2 = label_2 + 1
            else: label_4 = label_4 + 1
        # return majority label at the target node
        if label_2 >= label_2: return 2
        else: return 4

    # split training_data according to current node's feature and threshold
    left_data = [data for data in training_data if data[fea-1] <= t ]
    right_data = [data for data in training_data if data[fea-1] > t]

    left_label = find_max_label(root.get_left(), stop_node, left_data)
    right_label = find_max_label(root.get_right(), stop_node, right_data)

    return max(left_label, right_label)

def find_max_depth_node(node, depth=0):
    fea = node.get_feature()

    if "label" in str(fea): # return when a leaf is reached
        return depth, node.get_parent()

    # call recursively on left and right child of current node
    l_depth, l_leaf = find_max_depth_node(node.get_left(), depth + 1)
    r_depth, r_leaf = find_max_depth_node(node.get_right(), depth + 1)

    # return the max depth between the left and right child
    if l_depth >= r_depth: return l_depth, l_leaf
    else: return r_depth, r_leaf

def prune(root, training_data, test_data, test_data_labels):
    max_depth, max_leaf = find_max_depth_node(root)
    curr_node = max_leaf

    while max_depth > 7:
        root = RE_prune(root, root, training_data, test_data, test_data_labels) # run reduced error pruning
        new_depth, new_max_leaf = find_max_depth_node(root)

        if new_depth == max_depth and new_depth > 7: # force prune if target max depth is not reached
            parent = new_max_leaf.get_parent()
            grandparent = parent.get_parent()
            max_label = find_max_label(root, parent, training_data)
            new_leaf = Node("label " + str(max_label), 0, grandparent)

            if parent == grandparent.get_left():
                grandparent.set_left(new_leaf)
            else:
                grandparent.set_right(new_leaf)
            
            parent.set_left(None)
            parent.set_right(None)
        else:
            max_depth = new_depth

    return root
            
def RE_prune(root, curr_node, training_data, test_data, test_data_labels):
    if curr_node == root: # if the current node is the root, call RE_prune on left and right child
        root = RE_prune(root, root.get_left(), training_data, test_data, test_data_labels)
        return RE_prune(root, root.get_right(), training_data, test_data, test_data_labels)

    if "label" in str(curr_node.get_feature()): # return if a leaf is reached
        return root

    orig_accuracy = classify_accuracy(root, test_data, test_data_labels) # accuracy before pruning
    max_label = find_max_label(root, curr_node, training_data) # majority label at curr_node
    
    # replace subtree at current node
    leaf_parent = curr_node.get_parent()
    new_leaf = Node("label " + str(max_label), 0, leaf_parent)
    leaf_side = ""
    if leaf_parent.get_left() == curr_node: # replacing left child with leaf
        leaf_side = "left"
        leaf_parent.set_left(new_leaf)
    else: # replacing right child with leaf
        leaf_side = "right"
        leaf_parent.set_right(new_leaf)

    new_accuracy = classify_accuracy(root, test_data, test_data_labels) # accuracy after pruning

    if new_accuracy < orig_accuracy: # if accuracy does not improve, add subtree back to tree
        if leaf_side == "left": leaf_parent.set_left(curr_node)
        else: leaf_parent.set_right(curr_node)
        
        # run pruning again on both left and right child
        root = RE_prune(root, curr_node.get_left(), training_data, test_data, test_data_labels)
        return RE_prune(root, curr_node.get_right(), training_data, test_data, test_data_labels)
    else: # subtree was successfully replaced with a new leaf
        if leaf_side == "left": # run pruning on right child
            return RE_prune(root, curr_node.get_right(), training_data, test_data, test_data_labels)
        else: # run pruning on left child
            return RE_prune(root, curr_node.get_left(), training_data, test_data, test_data_labels)

def classify_q8(test_data):
    if test_data[2] <= 2:
        if test_data[6] <= 3:
            if test_data[5] <= 4: return 2
            else:
                if test_data[3] <= 1: return 2
                else: return 4
        else:
            if test_data[8] <= 1:
                if test_data[6] <= 5: return 2
                else:
                    if test_data[3] <= 1: return 2
                    else: return 4
            else: return 4
    else:
        if test_data[2] <= 4:
            if test_data[6] <= 3:
                if test_data[8] <= 1: return 2
                else:
                    if test_data[8] <= 8:
                        if test_data[8] <= 5:
                            if test_data[5] <= 3: return 2
                            else: return 4
                        else: return 2
                    else: return 4
            else:
                if test_data[3] <= 4:
                    if test_data[6] <= 7:
                        if test_data[5] <= 3:
                            if test_data[9] <= 1: return 4
                            else: return 2
                        else: return 2
                    else: return 2
                else: return 4
        else:
            if test_data[6] <= 0: return 2
            else:
                if test_data[6] <= 8:
                    if test_data[5] <= 5:
                        if test_data[8] <= 7: return 2
                        else: return 4
                    else: return 4
                else: return 4

def q7_9(labels, num):
    name = "P2_Q" + str(num) + ".txt"
    q7_9 = ''
    for label in labels: # store test labels to a string
        q7_9 = q7_9 + str(label) + ', '
    q7_9 = q7_9[:-2] # remove last ', ' from string
    with open(name, 'w') as filehandle: # write labels to a text file
        filehandle.writelines("%s\n" % q7_9)

############################################### CODE SETUP ###############################################
training_data = []
n = 0 # number of training data instances
benign_count = 0
malig_count = 0

# parse csv file of training data
with open('breast-cancer-wisconsin.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader: # iterate over all rows in the csv file
        temp_row = ['0' if x=='?' else x for x in row] # replace any '?' with '0'

        # count benign and malignant instances for question 1
        if row[-1] == '2': benign_count = benign_count + 1
        elif row[-1] == '4': malig_count = malig_count + 1

        training_data.append(list(map(int, temp_row))) # add to training_data matrix
        n = n + 1
training_data = np.array(training_data)

# parse csv file of test data
test_data = []
with open('P2_Test_Set.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader: # iterate over all rows in the csv file
        test_data.append(list(map(int, row))) # add to feature matrix
test_data = np.array(test_data)

################################################# PART 1 #################################################
# Question 1
print("Question 1:")
print(f"benign = {benign_count}\t malignant = {malig_count}")

# Question 2
print("\nQuestion 2:")
entropy = -(benign_count/n) * math.log2(benign_count/n) - (malig_count/n) * math.log2(malig_count/n)
print(f"entropy = {round(entropy, 4)}")

var_index = 4 - 1
var4_benign = [row[var_index] for row in training_data if row[-1] == 2]
var4_malig = [row[var_index] for row in training_data if row[-1] == 4]
t = 2

# Question 3
counts = [0]*4 # counts = [n_2minus, n_2plus, n_4minus, n_4plus]
# explicitly calculate counts for question 3
for x in var4_benign:
    if x <= t: counts[0] = counts[0] + 1
    else: counts[1] = counts[1] + 1
for x in var4_malig:
    if x <= t: counts[2] = counts[2] + 1
    else: counts[3] = counts[3] + 1

print("\nQuestion 3 (threshold = 2):")
print(f"n_2minus = {counts[0]}\t n_2plus = {counts[1]}")
print(f"n_4minus = {counts[2]}\t n_4plus = {counts[3]}")

# Question 4
info_gain = info_gain_calc(training_data, t, var_index)

print("\nQuestion 4 (threshold = 2):")
print(f"information gain = {round(info_gain,4)}")

################################################# PART 2 #################################################
features = [7, 6, 9, 10, 4, 3]
#tree_root = train_tree(training_data, features, n, None, entropy)

max_gains = [0]*6
max_t = [0]*6

benign = [row for row in training_data if row[-1] == 2]
malign = [row for row in training_data if row[-1] == 4]

for i in range(len(features)): # iterate over all features
    feature_index = features[i] - 1

    for t in range(0, 11): # iterate over all possible thresholds
        info_gain = info_gain_calc(training_data, t, feature_index)

        if info_gain > max_gains[i]: # check if current information gain is more than the current max gain
            max_gains[i] = info_gain
            max_t[i] = t

# find which feature results in the max info gain and store info about the split
max_info_gain = max(max_gains)
max_index = max_gains.index(max_info_gain) # index of feature to split at
split_feature = features[max_index] # feature number
split_threshold = max_t[max_index] # threshold

root = Node(split_feature, split_threshold, None) # create node for split
train_tree(training_data, features, root) # train tree for root

# Question 5
with open('P2_Q5.txt', 'w') as filehandle: # write tree structure in assignment format
    print_tree_structure(root, filehandle)

with open('P2_Q5_Code.txt', 'w') as filehandle: # write tree structure in python code format
    print_tree_code(root, filehandle)
#pptree.print_tree(root, nameattr='feature')
#pptree.print_tree(root, nameattr='threshold')

labels = [0]*200 # create array to store labels of each test instance
for i in range(len(test_data)):
    labels[i] = classify_q5(test_data[i]) # classify each test instance

# Question 7
q7_9(labels, 7)

# true labels of test_data
test_data_labels = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

pruned_root = prune(root, training_data, test_data, test_data_labels) # prune the tree to max depth 7

# Question 8
with open('P2_Q8.txt', 'w') as filehandle: # write tree structure in assignment format
    print_tree_structure(pruned_root, filehandle)

with open('P2_Q8_Code.txt', 'w') as filehandle: # write tree structure in python code format
    print_tree_code(pruned_root, filehandle)

prune_labels = [0]*200 # create array to store labels of each test instance
for i in range(len(test_data)):
    prune_labels[i] = classify_q8(test_data[i]) # classify each test instance

# Question 9
q7_9(prune_labels, 9)