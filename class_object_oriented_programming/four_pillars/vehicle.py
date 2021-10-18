from abc import ABC, abstractmethod

class Vehicle(ABC):
    # 코드를 쓰세요
    @abstractmethod
    def start(self):
        pass

    @property
    @abstractmethod
    def speed(self):
        pass

    def stop(self):
        self.speed = 0


print(Vehicle.mro())
print(dir(Vehicle))