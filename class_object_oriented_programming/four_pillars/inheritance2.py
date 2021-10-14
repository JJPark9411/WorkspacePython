class BankAccount:
    # 코드를 쓰세요
    def __init__(self, name, balance):
        self.name = name
        self.balance = balance

    def withdraw(self, amount):
        self.balance -= amount

    def deposit(self, amount):
        self.balance += amount

    def __str__(self):
        return "{}님의 계좌 예치금은 {}원입니다".format(self.name, self.balance)


class CheckingAccount(BankAccount):
    # 코드를 쓰세요
    def __init__(self, name, balance, max_spending):
        super().__init__(name, balance)
        self.max_spending = max_spending

    def use_check_card(self, amount):
        if amount <= self.max_spending:
            self.balance -= amount
        else:
            print("{}님의 체크 카드는 한 회 {} 초과 사용 불가능합니다".format(self.name, self.max_spending))


class SavingsAccount(BankAccount):
    # 코드를 쓰세요
    def __init__(self, name, balance, interest_rate):
        super().__init__(name, balance)
        self.interest_rate = interest_rate

    def add_interest(self):
        self.balance *= (1+self.interest_rate)

bank_account_1 = CheckingAccount("성태호", 100000, 10000)
bank_account_2 = SavingsAccount("강영훈", 20000, 0.05)

bank_account_1.withdraw(1000)
bank_account_1.deposit(1000)
bank_account_1.use_check_card(2000)

bank_account_2.withdraw(1000)
bank_account_2.deposit(1000)
bank_account_2.add_interest()

print(bank_account_1)
print(bank_account_2)

print(CheckingAccount.mro())
print(SavingsAccount.mro())