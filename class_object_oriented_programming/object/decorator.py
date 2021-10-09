def add_print_to(original): # decorator function
    def wrapper():
        print("Start method")
        original()
        print("End method")
    return wrapper


@add_print_to # print_hello 함수를 add_print_to로 꾸며주라는 의미
def print_hello():
    print("Hello!")


print_hello()

