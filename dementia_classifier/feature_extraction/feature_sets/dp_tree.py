# Author: Jordon Johnson

import re
from collections import Counter
# import numpy as np

#
# Represents a node in a DiscourseParseTree
#


class DPTreeNode:

    # Creates a new node with no parent, no children, and some data about itself
    def __init__(self):
        # node relationship variables
        self.parent = None
        self.left   = None
        self.right  = None
        # node data
        self.span = None
        self.prob = None
        self.text = None
        self.rel2par    = None
        self.nuclearity = None

    # prints the node TODO format this nicely
    def printNode(self):
        output_string = self.nuclearity + ' span ' + ' '.join([str(i) for i in self.span])
        if self.rel2par is not None:
            output_string += ' rel2par ' + self.rel2par
        if self.prob is not None:
            output_string += ' prob ' + str(self.prob)
        if self.text is not None:
            output_string += ' text "' + self.text + '"'
        print(output_string)

    # inserts a node into the tree
    def insert(self, newNode):
        # do nothing if self is a leaf
        if self.isLeaf():
            return
        # if left child is empty, insert new node there
        if self.left is None:
            self.left = newNode
            newNode.parent = self
            return
        # try inserting the new node as a left descendant
        self.left.insert(newNode)
        if newNode.parent is not None:  # inserting on left side was successful
            return
        # if right child is empty, insert new node there
        if self.right is None:
            self.right = newNode
            newNode.parent = self
            return
        # try inserting the new node as a right descendant
        self.right.insert(newNode)
        if newNode.parent is not None:  # inserting on right side was successful
            return
        # insertion on both left and right sides failed. Print error message if this is the case at the root
        if self.parent is None:
            print("ERROR INSERTING NODE")

    # Returns true if node is a leaf, false otherwise
    def isLeaf(self):
        return len(self.span) == 1

    # returns true if the specified node is an ancestor, false otherwise
    def isChildOf(self, node):
        return (self.parent is not None) and (self.parent == node or self.parent.isChildOf(node))

    def depth(self):
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return max(left_depth, right_depth) + 1


#
# Represents a discourse parse tree
#
class DiscourseParseTree:

    # class constants
    NON_NODE_LINE_REGEX = '^\s*\)$'
    SPAN_RELATION = 'span'

    # Creates an empty discourse parse tree (no root yet assigned)
    def __init__(self, lines):
        self.root = None
        self.buildTree(lines)
        self.leaves = list()
        self.buildLeafList(self.root)
#        print("      " + str(len(self.leaves)) + " leaves in extracted discourse parse tree.")  # TODO uncomment

    # creates a node and populates it with data from a line
    def createNode(self, line):
        node = DPTreeNode()
        node.nuclearity = get_nuclearity(line)
        node.span    = get_span(line)
        node.rel2par = get_rel2par(line)
        node.text    = get_text(line)
        return node

    # builds the tree given a list of strings acquired from a discourse parse tree
    def buildTree(self, lines):
        for line in lines:
            if not re.match(self.NON_NODE_LINE_REGEX, line):  # there is node data in this line
                node = self.createNode(line)
                if self.root is None:
                    self.root = node
                else:
                    self.root.insert(node)

    # recursively builds a list of the leaves in the tree
    def buildLeafList(self, node):
        if node.isLeaf():
            self.leaves.append(node)
        else:
            self.buildLeafList(node.left)
            if node.right is not None:
                self.buildLeafList(node.right)

    # gets the relation and its probability at the root of the discourse parse tree
    def getRootRelation(self):
        if self.root.right is None:  # only one EDU
            return self.root.left.rel2par, 0  # using zero because there is no relation
        else:
            relation = self.root.left.rel2par
            prob = self.root.left.prob
            right_child_relation = self.root.right.rel2par
            right_child_prob = self.root.right.prob
            if prob != right_child_prob:
                print('*** ERROR: relation probability mismatch')
                return None
            if relation == self.SPAN_RELATION:
                relation = right_child_relation
            return relation, prob

    # returns the text of all leaves (one line per leaf) that are descendants of the specified node
    def getSpanText(self, node):
        span = list()
        for leaf in self.leaves:
            if leaf.isChildOf(node):
                span.append(leaf.text)
        return span

    # prints the tree
    def printTree(self):
        self.printTreeHelper(self.root)

    # recursive helper function for printTree
    def printTreeHelper(self, node):
        node.printNode()
        if node.left is not None:
            self.printTreeHelper(node.left)
        if node.right is not None:
            self.printTreeHelper(node.right)

    def getAllEdges(self):
        return self.getAllEdgesHelper(self.root)

    # recursive helper function for printTree
    def getAllEdgesHelper(self, node):
        edges = []
        if node.rel2par is not None:
            edges.append(node.rel2par)
        if node.left is not None:
            edges += self.getAllEdgesHelper(node.left)
        if node.right is not None:
            edges += self.getAllEdgesHelper(node.right)
        return edges


# --------------------------
# -- Line parsing methods --
# --------------------------

def get_nuclearity(line):
    words = re.findall(r"[\w\-']+", line)
    return words[0]


def get_span(line):
    return [int(s) - 1 for s in re.findall(r"[\w\-']+", line) if s.isdigit()]


def get_rel2par(line):
    # Split into words and return word after 'rel2par'
    if 'rel2par' in line:
        words = re.findall(r"[\w\-']+", line)
        return words[words.index('rel2par') + 1]
    else:
        return None


def get_text(line):
    if 'text' in line:
        return line.split('_!')[1]
    else:
        return None


def mean(arr):
    return sum(arr) / float(len(arr))

# --------------
# For testing
# --------------
if __name__ == '__main__':
    BREAK = 'EDU_BREAK'
    with open("../data/dbank/discourse_trees/control/doc/280-2_doc.dis") as f:
        lines = f.readlines()
    dpt = DiscourseParseTree(lines)
    import pdb
    pdb.set_trace()

    # with open('seg_001-0.txt') as f:
    #     lines = f.readlines()

    # edus = []
    # for l in lines:
    #     edus.append(1 + l.count(BREAK))

    # print mean(edus)
