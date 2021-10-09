class User:
    def say_hello(self):
        # 인사 메시지 출력 메소드
        print("Hello! I'm {}.".format(self.name))

    def sign_in(self, my_email, my_password):
        # 로그인 메소드
        if self.email == my_email and self.password == my_password:
            print("Hello! {}.".format(self.name))
        else:
            print("Failed to sign in.")

    def check_name(self, name):
        # 파라미터로 받는 name이 유저의 이름과 같은지 boolean으로 리턴하는 메소드
        return self.name == name


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
user1.say_hello() # instance의 method는 instance를 첫 번째 인자로 받는다.

# user1.sign_in(user1, "kim@codeit.kr", "12345") # 에러 발생
user1.sign_in("kim@codeit.kr", "12345")

print(user1.check_name("Kim"))
print(user1.check_name("Lee"))