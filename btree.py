# Thanks to SK lee who provided the skeleton code

import math, sys
import pandas as pd
import numpy as np
from tqdm import tqdm


class Node:
    def __init__(self, leaf=False):
        self.keys = []
        self.parent = None
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

        ## 가운데 값을 부모로 올리기
        x.children.insert(i + 1, z)   # 리스트 사용으로 뒤에 넣어야(i+1)
        x.keys.insert(i, y.keys[t - 1])

        ## 슬라이싱 사용, t부터 끝까지, t-1까지(t-1 미포함)
        z.keys = y.keys[t:]
        y.keys = y.keys[:t - 1]

        ## leaf가 아닐 때, leaf일 때는 children이 없음
        if not y.leaf:
            z.children = y.children[t:]
            y.children = y.children[:t]

        ## parent 업데이트
            for child in z.children:
                child.parent = z
        z.parent = x
        y.parent = x
        
        


    # B-Tree-Insert
    def insert(self, k):
        '''
        # insert the key k into the B-Tree
        # return: None
        '''
        # (TODO)
        # Write your own code here 
        root = self.root
        # if isinstance(k[0], str):   ## test code에 작은 수를 넣었더니 key  value통으로 str로 들어와서
        #     data = k[0].split()     ## 왜인지 모르겠지만 혼자만든 테스트 파일에 문제가 있는 것 같음  
        #     k = [int(data[0]), int(data[1])]          

        # Case 1: if the root is full
        if len(root.keys) == 2 * self.t - 1:
            new_root = Node()
            self.root = new_root
            new_root.children.append(root)
            root.parent = new_root
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
        # Case 1: x가 leaf 노드일 때
        i = len(x.keys) - 1  ## i = x.n
        if x.leaf:
            x.keys.append(None)
            while i >= 0 and k[0] < x.keys[i][0]:
                x.keys[i + 1] = x.keys[i]
                i -= 1
            x.keys[i + 1] = k
            # Disk-Write(x)

        # Case 2: x가 internal 노드일 때
        else:
            while i >= 0 and k[0] < x.keys[i][0]:
                i -= 1
            i += 1
            ## Disk-Read(x.c[i])
            if len(x.children[i].keys) == 2 * self.t - 1:
                self.split_child(x, i)
                if k[0] > x.keys[i][0]:
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
        if x is None:
            return None, None
        
        i = 0
        while i < len(x.keys) and key > x.keys[i][0]:   ## key값이 더 클 때
               i += 1
        
        if i < len(x.keys) and key == x.keys[i][0]:   ## 노드에 key가 있을 때
            return x, i
        elif x.leaf:   ## key가 없고 leaf 노드일 때
            return None, None
        else:   ## key가 없고 internal 노드일 때, 다음 자식 노드로 이동
            if i < len(x.children):
                return self.search_key(x.children[i], key)



    def delete(self, k):
        '''
        # delete the key k from the B-tree
        # return: None
        '''
        # (TODO)
        # Write your own code here 
        
        node, i = self.search_key(self.root, k)
        if node is not None:
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

        

    def delete_leaf_node(self, x, i):
        '''
        # delete the key in a leaf node
        '''
        # (TODO)
        # Write your own code here 
        ## x는 leaf 노드, i는 leaf 노드에 있는 key index
        ## Case 1-1: 노드에 key가 t-1개보다 많을 때
        if len(x.keys) > self.t - 1:
            x.keys.pop(i)
        
        ## Case 1-2: 노드에 key가 t-1개일 때
        else:
            ## root가 leaf고 여기에 있는 값 제거할 때
            parent = x.parent
            if parent is None:
                x.keys.pop(i)
                return
            
            parent_index = parent.children.index(x)

        ## Case 1-2a: sibling이 key가 t개 이상일 때
            ## 왼쪽 sibling이 존재하고, 왼쪽 sibling이 key가 t개 이상일 때
            if parent_index > 0 and len(parent.children[parent_index - 1].keys) > self.t - 1:
                self.borrow_sibling(x, i, True)
            ## 오른쪽 sibling이 존재하고, 오른쪽 sibling이 key가 t개 이상일 때
            elif parent_index < len(parent.children) - 1 and len(parent.children[parent_index + 1].keys) > self.t - 1:
                self.borrow_sibling(x, i, False)
            
        ## Case 1-2b: 양쪽 sibling 모두 key가 t-1개일 때, 모두 leaf
            else:
                ## 왼쪽 sibling이랑 부모에 빌려온 key를 합칠 때
                if parent_index > 0:
                    left_sibling = parent.children[parent_index - 1]
                    x.keys.pop(i)   # key 제거
                    left_sibling.keys.append(parent.keys[parent_index - 1])   # 부모의 key를 왼쪽 형제에 추가
                    left_sibling.keys.extend(x.keys)   # 노드의 key를 왼쪽 형제에 추가

                    parent.children.pop(parent_index)   # 부모에서 노드 제거
                    parent.keys.pop(parent_index - 1)   # 부모에서 key 제거
                    x.parent = None

                ## 왼쪽 sibling이 없을 때
                else: 
                    right_sibling = parent.children[parent_index + 1]
                    x.keys.pop(i)
                    x.keys.append(parent.keys[parent_index])
                    x.keys.extend(right_sibling.keys)

                    parent.children.pop(parent_index + 1)
                    parent.keys.pop(parent_index)
                    right_sibling.parent = None

                ## 1-2b 제거 후 root가 비어있을 때, 높이 줄이기
                if parent == self.root and len(parent.keys) == 0:
                    if parent_index > 0:
                        self.root = left_sibling
                    else:
                        self.root = x
                    self.root.parent = None
                ## 여기서는 p20처럼 1-2b처럼 부모에 빌려와서 옆 노드랑 병합만
                elif len(parent.keys) < self.t - 1:
                    self.check_smaller_than_t(parent)
                    


    def delete_internal_node(self, x, i):
        '''
        # delete the key in an internal node
        '''
        # (TODO)
        # Write your own code here
        predecessor = self.find_predecessor(x.children[i])   ## leaf가 아니면 무조건 있다
        x.keys[i], predecessor.keys[-1] = predecessor.keys[-1], x.keys[i]
        self.delete_leaf_node(predecessor, -1)



    # implement whatever you need 
    def borrow_merge(self, x, j):
        pass
        
    def check_smaller_than_t(self, x):

        parent = x.parent
        if parent is None:
            return
        
        parent_index = parent.children.index(x)

        if parent_index > 0:
            left_sibling = parent.children[parent_index - 1]
            left_sibling.keys.append(parent.keys[parent_index - 1])
            left_sibling.keys.extend(x.keys) 
            parent.children.pop(parent_index)
            parent.keys.pop(parent_index - 1)
            x.parent = None
            
            ## 여기는 leaf가 아니라서 children 병합해줘야 함
            left_sibling.children.extend(x.children)
            for child in x.children:
                child.parent = left_sibling
            x.children.clear()

        else:
            right_sibling = parent.children[parent_index + 1]
            x.keys.append(parent.keys[parent_index])
            x.keys.extend(right_sibling.keys)
            parent.children.pop(parent_index + 1)
            parent.keys.pop(parent_index)
            right_sibling.parent = None

            ## 여기는 leaf가 아니라서 children 병합해줘야 함
            x.children.extend(right_sibling.children)
            for child in right_sibling.children:
                child.parent = x
            right_sibling.children.clear()

        if parent == self.root and len(parent.keys) == 0:
            if parent_index > 0:
                self.root = left_sibling
            else:
                self.root = x
            self.root.parent = None
        elif len(parent.keys) < self.t - 1:
            self.check_smaller_than_t(parent)
        
        

    def find_predecessor(self, x):
        ## 여기서 x는 제거할 노트의 왼쪽 자식 노드
        current = x
        while not current.leaf:
            current = current.children[-1]
        return current
        
    def merge_sibling(self, x, i, j):
        pass

    def borrow_sibling(self, x, i, is_left):
        ## i: key index, j: is_left
        parent = x.parent
        parent_index = parent.children.index(x)

        ## 왼쪽 sibling이 존재하고, 왼쪽 sibling이 key가 t개 이상일 때
        if is_left:
            left_sibling = parent.children[parent_index - 1]
            x.keys.pop(i)   # key 제거
            x.keys.insert(0, parent.keys[parent_index - 1])  # 부모의 키를 노드의 앞에 삽입
            parent.keys[parent_index - 1] = left_sibling.keys.pop()  # 왼쪽 형제의 마지막 키를 부모로 이동

        ## 오른쪽 sibling이 존재하고, 오른쪽 sibling이 key가 t개 이상일 때
        else:
            right_sibling = parent.children[parent_index + 1]
            x.keys.pop(i)   # key 제거
            x.keys.append(parent.keys[parent_index])  # 부모의 키를 노드의 뒤에 추가
            parent.keys[parent_index] = right_sibling.keys.pop(0)  # 오른쪽 형제의 첫 번째 키를 부모로 이동


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


def deletion_test(B, delete_file):
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

