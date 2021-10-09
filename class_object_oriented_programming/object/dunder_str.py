class User:
    count = 0

    def __init__(self, name, email, pw):
        # 유저 인스턴스의 모든 변수를 지정해주는 메소드
        self.name = name
        self.email = email
        self.pw = pw

        User.count += 1

    def say_hello(self):
        print("Hello! I'm {}.".format(self.name))

    def __str__(self): # dunder str # print 함수를 호출할 때 자동으로 호출된다.
        return "User: {}, Email: {}, Password: *****".format(self.name, self.email)


user1 = User("Kim", "kim@codeit.kr", "12345")
user2 = User("Lee", "lee@codeit.kr", "67890")
user3 = User("Park", "park@codeit.kr", "67890")

print(user1)
print(user2)

print(User.count)