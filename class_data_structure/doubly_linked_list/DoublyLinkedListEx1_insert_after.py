class Node:
    """링크드 리스트의 노드 클래스"""

    def __init__(self, data):
        self.data = data  # 실제 노드가 저장하는 데이터
        self.next = None  # 다음 노드에 대한 레퍼런스
        self.prev = None  # 전 노드에 대한 레퍼런스


class LinkedList:
    """링크드 리스트 클래스"""

    def __init__(self):
        self.head = None  # 링크드 리스트의 가장 앞 노드
        self.tail = None  # 링크드 리스트의 가장 뒤 노드

    def insert_after(self, previous_node, data):
        """더블리 링크드 리스트 삽입 연산"""
        new_node = Node(data)

        if previous_node is self.tail:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
        else:
            new_node.prev = previous_node
            new_node.next = previous_node.next
            previous_node.next = new_node
            previous_node.next.prev = new_node

    def find_node_at(self, index):
        """링크드 리스트 접근 연산 메소드. 파라미터 인덱스는 항상 있다고 가정한다"""

        iterator = self.head  # 링크드 리스트를 돌기 위해 필요한 노드 변수

        # index 번째 있는 노드로 간다
        for _ in range(index):
            iterator = iterator.next

        return iterator

    def append(self, data):
        """링크드 리스트 추가 연산 메소드"""
        new_node = Node(data)  # 새로운 노드 생성

        # 빈 링크드 리스트라면 head와 tail을 새로 만든 노드로 지정
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        # 이미 노드가 있으면
        else:
            self.tail.next = new_node  # 마지막 노드의 다음 노드로 추가
            new_node.prev = self.tail
            self.tail = new_node  # 마지막 노드 업데이

    def __str__(self):
        """링크드 리스트를 문자열로 표현해서 리턴하는 메소드"""
        res_str = "|"

        # 링크드 리스트 안에 모든 노드를 돌기 위한 변수. 일단 가장 앞 노드로 정의한다.
        iterator = self.head

        # 링크드 리스트 끝까지 돈다
        while iterator is not None:
            # 각 노드의 데이터를 리턴하는 문자열에 더해준다
            res_str += " {} |".format(iterator.data)
            iterator = iterator.next  # 다음 노드로 넘어간다

        return res_str


# 새로운 링크드 리스트 생성
my_list = LinkedList()

# 새로운 노드 5개 추가
my_list.append(2)
my_list.append(3)
my_list.append(5)
my_list.append(7)
my_list.append(11)

print(my_list)

# tail 노드 뒤에 노드 삽입
tail_node = my_list.tail  # 4 번째(마지막)노드를 찾는다
my_list.insert_after(tail_node, 5)  # 4 번째(마지막)노드 뒤에 노드 추가
print(my_list)
print(my_list.tail.data)  # 새로운 tail 노드 데이터 출력

# 링크드 리스트 중간에 데이터 삽입
node_at_index_3 = my_list.find_node_at(3)  # 노드 접근
my_list.insert_after(node_at_index_3, 3)
print(my_list)

# 링크드 리스트 중간에 데이터 삽입
node_at_index_2 = my_list.find_node_at(2)  # 노드 접근
my_list.insert_after(node_at_index_2, 2)
print(my_list)

