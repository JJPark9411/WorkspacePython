class Node:
    """링크드 리스트의 노드 클래스"""

    def __init__(self, data):
        self.data = data  # 노드가 저장하는 데이터
        self.next = None  # 다음 노드에 대한 레퍼런스


# 데이터 2, 3, 5, 7, 11을 담는 노드들 생성
head_node = Node(2)
node_1 = Node(3)
node_2 = Node(5)
node_3 = Node(7)
tail_node = Node(11)

# 노드들을 연결
head_node.next = node_1
node_1.next = node_2
node_2.next = node_3
node_3.next = tail_node

# 노드 순서대로 출력
iterator = head_node

while iterator is not None:
    print(iterator.data)
    iterator = iterator.next


class LinkedList:
    """링크드 리스트 클래스"""

    def __init__(self):
        self.head = None
        self.tail = None

    def insert_after(self, previous_node, data):
        """링크드 리스트 주어진 노드 뒤 삽입 연산 메소드"""
        new_node = Node(data)

        # 가장 마지막 순서 삽입
        if previous_node is self.tail:
            self.tail.next = new_node
            self.tail = new_node
        else: # 두 노드 사이 삽입
            new_node.next = previous_node.next
            previous_node.next = new_node

    def find_node_at(self, index):
        """링크드 리스트 접근 연산 메소드. 파라미터 인덱스는 항상 있다고 가정"""
        iterator = self.head

        for _ in range(index):
            iterator = iterator.next

        return iterator

    def append(self, data):
        """링크드 리스트 추가 연산 메소드"""
        new_node = Node(data)

        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node

    def __str__(self):
        """링크드 리스트를 문자열로 표현해서 리턴하는 메소드"""
        res_str = "|"

        # 링크드  리스트 안에 모든 노드를 돌기 위한 변수. 일단 가장 앞 노드로 정의한다.
        iterator = self.head

        # 링크드  리스트 끝까지 돈다
        while iterator is not None:
            # 각 노드의 데이터를 리턴하는 문자열에 더해준다
            res_str += f" {iterator.data} |"
            iterator = iterator.next  # 다음 노드로 넘어간다

        return res_str


# 새로운 링크드 리스트 생성
my_list = LinkedList()

# 링크드 리스트에 데이터 추가
my_list.append(2)
my_list.append(3)
my_list.append(5)
my_list.append(7)
my_list.append(11)

# insert_after
node_2 = my_list.find_node_at(2) # 인덱스 2에 있는 노드 접근
my_list.insert_after(node_2, 6)  # 인덱스 2 노드 뒤에 6 삽입

print("insert_after")
print(my_list)


# find_node_at
# 링크드 리스트 노드에 접근 (데이터 가져오기)
print("my_list[3] = ", end='')
print(my_list.find_node_at(3).data)

# 링크드 리스트 노드에 접근 (데이터 바꾸기)
my_list.find_node_at(2).data = 13


# 노드 순서대로 출력
iterator = my_list.head

while iterator is not None:
    print(iterator.data)
    iterator = iterator.next

# __str__로 링크드 리스트 출력
print("__str__")
print(my_list)