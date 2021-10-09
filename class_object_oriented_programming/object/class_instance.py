class User:
    def say_hello(some_user):
        # 인사 메시지 출력 메소드
        print("Hello! I'm {}.".format(some_user.name))

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

User.say_hello(user1)
User.say_hello(user2)
User.say_hello(user3)