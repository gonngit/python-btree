# Thanks to SK lee who provided the skeleton code

import math, sys
import pandas as pd
import numpy as np
from tqdm import tqdm


class Node:
    def __init__(self, leaf=False):
        self.keys = []
        self.parent = []
        self.children = []
        self.leaf = leaf


class BTree:
    def __init__(self, t):
        '''
        # create a instance of the Class of a B-Tree
        # t : the minimum degree t
        # (the max num of keys is 2*t -1, the min num of keys is t-1)
        '''
        self.root = Node(True)
        self.t = t

    # B-Tree-Split-Child
    def split_child(self, x, i): 
        '''
        # split the node x's i-th child that is full
        # x: the current node
        # i: the index of the node x's child to be split
        # return: None
        '''
        # (TODO)
        # Write your own code here
        t = self.t
        y = x.children[i]
        z = Node(y.leaf)

        x.children.insert(i + 1, z)
        x.keys.insert(i, y.keys[t - 1])

        z.keys = y.keys[t:]
        y.keys = y.keys[:t - 1]

        if not y.leaf:
            z.children = y.children[t:]
            y.children = y.children[:t]
        
        # ## 슬라이싱 사용, t부터 끝까지, t-1까지(t-1 미포함)
        # z.keys = y.keys[t:]
        # y.keys = y.keys[:t-1]

        # if not y.leaf:
        #     z.children = y.children[t:]
        #     y.children = y.children[:t]

        # ## 가운데 값을 부모로 올리기
        # x.keys.insert(i, y.keys[t - 1])
        # x.children.insert(i + 1, z)

        # ## 부모 업데이트
        # z.parent = x
        # for child in z.children:
        #     child.parent = z


        



    # B-Tree-Insert
    def insert(self, k):
        '''
        # insert the key k into the B-Tree
        # return: None
        '''
        # (TODO)
        # Write your own code here 

        root = self.root

        # Case 1: if the root is full
        if len(root.keys) == 2*self.t - 1:
            new_root = Node()
            self.root = new_root
            new_root.children.append(root)
            self.split_child(new_root, 0)
            self.insert_key(new_root, k)
        # Case 2: if the root is not full
        else:
            self.insert_key(root, k)



    # B-Tree-Insert-Nonfull
    def insert_key(self, x, k):
        '''
        # insert the key k into node x
        # return: None
        '''
        # (TODO)
        # Write your own code here 
        # Case 1: if the node x is leaf
        i = len(x.keys) - 1  ## i = x.n
        if x.leaf:
            x.keys.append(None)
            while i >= 0 and k < x.keys[i]:
                x.keys[i + 1] = x.keys[i]
                i -= 1
            x.keys[i + 1] = k
            ## Disk-Write(x)

        # Case 2: if the node x is an internal node
        else:
            while i >= 0 and k < x.keys[i]:
                i -= 1
            i += 1
            ## Disk-Read(x.c[i])
            if len(x.children[i].keys) == (2 * self.t) - 1:
                self.split_child(x, i)
                if k > x.keys[i]:
                    i += 1
            self.insert_key(x.children[i], k)


    # B-Tree-Search
    def search_key(self, x, key):
        '''
        # search for the key in node x
        # return: the node x that contains the key, the index of the key if the key is in the B-tree
        '''
        # (TODO)
        # Write your own code here 
        i = 0
        while i < len(x.keys) and key > x.keys[i]:
            i += 1

        if i < len(x.keys) and key == x.keys[i]:
            return x, i
        elif x.leaf:
            return None
        else:
            return self.search_key(x.children[i], key)



    def delete(self, k):
        '''
        # delete the key k from the B-tree
        # return: None
        '''
        # (TODO)
        # Write your own code here 
        result = self.search_key(self.root, k[0])
        if result:
            node, i = result

            ## Case 1: key가 leaf 노드에 있을 때
            if node.leaf:
                self.delete_leaf_node(node, i)
                
            ## Case 2: key가 internal 노드에 있을 때
            else:
                self.delete_internal_node(node, i)

            ## root 처리
            if len(self.root.keys) == 0 and not self.root.leaf:
                self.root = self.root.children[0]
                self.root.parent = None
        else:
            pass

        

    def delete_leaf_node(self, x, i):
        '''
        # delete the key in a leaf node
        '''
        # (TODO)
        # Write your own code here 
        ## Case 1-1: 노드에 key가 t-1개보다 많을 때
        if len(x.keys) > self.t - 1:
            x.keys.pop(i)
        
        ## Case 1-2: 노드에 key가 t-1개일 때
        else:
            ## root가 leaf고 여기에 있는 값 제거할 때
            parent = x.parent 
            if parent is None:
                if len(x.keys) == 1:
                    x.keys.pop(i)
                    return
            parent_index = parent.children.index(x)
            is_left = False

        ## Case 1-2a: sibling이 key가 t개 이상일 때
            ## 왼쪽 sibling이 존재하고, 왼쪽 sibling이 key가 t개 이상일 때
            if parent_index > 0 and len(parent.children[parent_index - 1].keys) > self.t - 1:
                is_left = True
                self.borrow_sibling(x, i, is_left)
            ## 오른쪽 sibling이 존재하고, 오른쪽 sibling이 key가 t개 이상일 때
            elif parent_index < len(parent.children) - 1 and len(parent.children[parent_index + 1].keys) > self.t - 1:
                is_left = False
                self.borrow_sibling(x, i, is_left)
            
        ## Case 1-2b: 양쪽 sibling 모두 key가 t-1개일 때
            else:
                ## 왼쪽 sibling이랑 부모에 빌려온 key를 합칠 때 
                if parent_index > 0:
                    ## key 제거하고 부모에 있는 key를 빌려오기
                    x.keys.insert(0, parent.keys[parent_index - 1])
                    x.keys.pop(i + 1)
                    ## 왼쪽 sibling과 합치기
                    left_sibling = parent.children[parent_index - 1]
                    x.keys = left_sibling.keys + x.keys
                    x.children = left_sibling.children + x.children
                    parent.children.pop(parent_index - 1)

                    ## 제거 후 root가 비어있을 때, 높이 줄이기
                    if self.root == parent and len(parent.children) == 1:
                        parent.keys.pop(parent_index - 1)
                        self.root = x
                        self.root.parent = None
                        return

                    ## 부모에 남은 key 제거, 재귀적으로 다시 부모에서 1-2a, 1-2b 확인
                    self.delete_leaf_node(parent, parent_index - 1)
                
                ## 왼쪽 sibling이 없을 때
                else: 
                    ## key 제거하고 부모에 있는 key를 빌려오기
                    x.keys.append(parent.keys[parent_index + 1])
                    x.keys.pop(i)
                    ## 오른쪽 sibling과 합치기
                    right_sibling = parent.children[parent_index + 1]
                    right_sibling.keys = x.keys + right_sibling.keys
                    right_sibling.children = x.children + right_sibling.children
                    parent.children.pop(parent_index)

                    ## 제거 후 root가 비어있을 때, 높이 줄이기
                    if self.root == parent and len(parent.children) == 1:
                        parent.keys.pop(parent_index)
                        self.root = right_sibling
                        self.root.parent = None
                        return

                    ## 부모에 남은 key 제거, 재귀적으로 다시 부모에서 1-2a, 1-2b 확인
                    self.delete_leaf_node(parent, parent_index)


            


    def delete_internal_node(self, x, i):
        '''
        # delete the key in an internal node
        '''
        # (TODO)
        # Write your own code here 
        predecessor = self.find_predecessor(x.children[i])
        x.keys[i], predecessor.keys[-1] = predecessor.keys[-1], x.keys[i]
        self.delete_leaf_node(predecessor, len(predecessor.keys) - 1)



    # implement whatever you need 
    def borrow_merge(self, x, j):
        pass

    def check_smaller_than_t(self, x):
        pass

    def find_predecessor(self, x):
        ## 여기서 x는 제거할 노트의 왼쪽 자식 노드
        current = x
        while not current.leaf:
            current = current.children[-1]
        return current.children[-1]
        
    def merge_sibling(self, x, i, j):
        pass

    def borrow_sibling(self, x, i, j):
        ## i: key index, j: is_left
        parent = x.parent
        parent_index = parent.children.index(x)

        ## 왼쪽 sibling이 존재하고, 왼쪽 sibling이 key가 t개 이상일 때
        if j:
            left_sibling = parent.children[parent_index - 1]  
            x.keys.insert(0, parent.keys.pop(parent_index - 1))   # 부모의 key를 노드의 앞에 넣기(insert, 뒤로 밀림)
            parent.keys[parent_index - 1] = left_sibling.keys.pop(-1)   # 왼쪽 sibling의 마지막 key를 부모로 올리기
            x.pop(i - 1)   # key 삭제

        ## 오른쪽 sibling이 존재하고, 오른쪽 sibling이 key가 t개 이상일 때
        else:
            right_sibling = parent.children[parent_index + 1]
            x.keys.append(parent.keys.pop(parent_index))   # 부모의 key를 노드의 뒤에 넣기(append)
            parent.keys[parent_index] = right_sibling.keys.pop(0)   # 오른쪽 sibling의 첫번째 key를 부모로 올리기
            x.pop(i)   # key 삭제
        


    # for printing the statistic of the resulting B-tree
    def traverse_key(self, x, level=0, level_counts=None):
        '''
        # run BFS on the B-tree to count the number of keys at every level
        # return: level_counts
        '''
        if level_counts is None:
            level_counts = {}

        if x:
            # counting the number of keys at the current level
            if level in level_counts:
                level_counts[level] += len(x.keys)
            else:
                level_counts[level] = len(x.keys)

            # recursively call the traverse_key() for further traverse
            for child in x.children:
                self.traverse_key(child, level + 1, level_counts)

        return level_counts

# Btree Class done


def get_file():
    '''
    # read an input file (.csv) with its name
    '''
    file_name = (input("Enter the file name you want to insert or delete ▷ (e.g., insert1 or delete1_50 or delete1_90 or ...) "))

    while True:
        try:
            file = pd.read_csv('inputs/'+file_name+'.csv',
                               delimiter='\t', names=['key', 'value'])
            return file
        except FileNotFoundError:
            print("File does not exist.")
            file_name = (input("Enter the file name again. ▷ "))


def insertion_test(B, file):
    '''
    #   read all keys and values from the file and insert them into the B-tree
    #   B   : an empty B-tree
    #   file: a csv file that contains keys to be inserted
    #   return: the resulting B-tree
    '''

    file_key = file['key']
    file_value = file['value']

    print('===============================')
    print('[ Insertion start ]')

    for i in tqdm(range(len(file_key))): # tqdm shows the insertion progress and the elapsed time
        B.insert([file_key[i], file_value[i]])

    print('[ Insertion complete ]')
    print('===============================')
    print()

    return B


def deletion_test(B, root, delete_file):
    '''
    #   read all keys and values from the file and delete them from the B-tree
    #   B   : the current B-tree
    #   file: a csv file that contains keys to be deleted
    #   return: the resulting B-tree
    '''

    delete_key = delete_file['key']

    print('===============================')
    print('[ Deletion start ]')

    for i in tqdm(range(len(delete_key))):
        B.delete(delete_key[i])

    print('[ Deletion complete ]')
    print('===============================')
    print()

    return B


def print_statistic(B):
    '''
    # print the information about the current B-tree
    # the number of keys at each level
    # the total number of keys in the B-tree
    '''
    print('===============================')
    print('[ Print statistic of tree ]')

    level_counts = B.traverse_key(B.root)

    for level, counts in level_counts.items():
        if level == 0:
            print(f'Level {level} (root): Key Count = {counts}')
        else:
            print(f'Level {level}: Key Count = {counts}')
    print('-------------------------------')
    total_keys = sum(counts for counts in level_counts.values())
    print(f'Total number of keys across all levels: {total_keys}')
    print('[ Print complete ]')
    print('===============================')
    print()

def main():
    while True:
        try:
            num = int(input("1.insertion 2.deletion. 3.statistic 4.end ▶  "))

            # 1. Insertion
            if num == 1: 
                t = 3 # minimum degree
                B = BTree(t) # make an empty b-tree with the minimum degree t

                insert_file = get_file()
                B = insertion_test(B, insert_file)

            # 2. Deletion
            elif num == 2:
                delete_file = get_file()
                B = deletion_test(B, delete_file)

            # 3. Statistic
            elif num == 3:
                print_statistic(B)

            # 4. End program
            elif num == 4:
                sys.exit(1)

            else:
                print("Invalid input. Please enter 1, 2, 3, or 4.")

        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == '__main__':
    main()

