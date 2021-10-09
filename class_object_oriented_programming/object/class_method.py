class User:
    count = 0

    def __init__(self, name, email, password):
        self.name = name
        self.email = email
        self.password = password

        User.count += 1

    def say_hello(self):
        print("Hello! I'm {}.".format(self.name))

    def __str__(self):
        return "User: {}, Email: {}, Password: *****".format(self.name, self.email)

    @classmethod
    def number_of_users(cls): # class method는 첫 번째 인자로 class를 전달받는다. 이름은 일반적으로 cls로 정한다.
        print("Number of Users: {}".format(cls.count))

    @staticmethod
    def is_valid_email(email_address):
        return "@" in email_address


user1 = User("Kim", "kim@codeit.kr", "12345")
user2 = User("Lee", "lee@codeit.kr", "12346")
user3 = User("Park", "park@codeit.kr", "12347")

User.number_of_users() # class method는 class로 호출하든 instance로 호출하든 첫 번째 인자로 class를 전달한다.
user1.number_of_users() # class method는 class로 호출하든 instance로 호출하든 첫 번째 인자로 class를 전달한다.

print(User.is_valid_email("test"))
print(User.is_valid_email("test@codeit.kr"))