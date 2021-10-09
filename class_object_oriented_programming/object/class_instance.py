class User:
    def say_hello(some_user):
        # 인사 메시지 출력 메소드
        print("Hello! I'm {}.".format(some_user.name))

    def sign_in(some_user, my_email, my_password):
        # 로그인 메소드
        if some_user.email == my_email and some_user.password == my_password:
            print("Hello! {}.".format(some_user.name))
        else:
            print("Failed to sign in.")


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