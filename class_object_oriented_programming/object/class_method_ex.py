class User:
    def __init__(self, name, email, password):
        self.name = name
        self.email = email
        self.password = password

    @classmethod
    def from_string(cls, string_params):
        # 코드를 쓰세요
        separated_params =  string_params.split(sep=',')
        return cls(separated_params[0], separated_params[1], separated_params[2])

    @classmethod
    def from_list(cls, list_params):
        # 코드를 쓰세요
        return cls(list_params[0], list_params[1], list_params[2])


# 유저 생성 및 초기값 설정
younghoon = User.from_string("강영훈,younghoon@codeit.kr,123456")
yoonsoo = User.from_list(["이윤수", "yoonsoo@codeit.kr", "abcdef"])

print(younghoon.name, younghoon.email, younghoon.password)
print(yoonsoo.name, yoonsoo.email, yoonsoo.password)