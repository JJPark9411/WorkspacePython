class User:
    pass # 아무 내용이 없다는 의미


user1 = User()
user2 = User()
user3 = User()

user1.name = "Kim"
user1.email = "kim@codeit.kr"
user1.password = "12345"

user2.name = "Park"
user2.email = "park@codeit.kr"
user2.password = "67890"

user3.name = "Choi"
user3.email = "choi@codeit.kr"
user3.password = "13579"

print(user1.email)
print(user2.password)