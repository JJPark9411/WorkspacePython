class Node:
    """링크드 리스트의 노드 클래스"""
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None # 다음 노드에 대한 레퍼런스
        self.prev = None # 이전 노드에 대한 레퍼런스

class LinkedList:
    """링크드 리스트 클래스"""
    def __init__(self):
        self.head = None # 링크드 리스트의 가장 앞 노드
        self.tail = None # 링크드 리스트의 가장 뒤 노드

    def find_node_with_key(self, key):
        """링크드 리스트에서 주어진 데이터를 갖고 있는 노드를 리턴한다. 단, 해당 노드가 없으면 None을 리턴한다."""
        iterator = self.head # 링크드 리스트를 돌기 위해 필요한 노드 변수

        while iterator is not None:
            if iterator.key == key:
                return iterator

            iterator = iterator.next

        return None
    