def print_hello():
    print("Hello!")


def add_print_to(original):
    def wrapper():
        print("Start method")
        original()
        print("End method")
    return wrapper


print_hello = add_print_to(print_hello)

print_hello()