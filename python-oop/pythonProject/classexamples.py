from abc import ABC, abstractmethod

'''
Python does not support method overloading
'''

class Book:
    def __init__(self, title, quantity, author, price):
        self.title = title
        self.quantity = quantity
        self.author = author
        # private fields need two underscores
        self.__price = price
        self.__discount = None

    '''
    The annotation @abstactmethod
    allows for abastractions in Python
    foricing every subclass to implements the abstract method
    '''
    @abstractmethod
    def __repr__(self):
        return f"Book: {self.title}, Quantity: {self.quantity}, Author: {self.author}, Price: {self.get_price()}"

    '''
    Setters and getters are needed to access private variables
    '''

    def set_discount(self, discount):
        self.__discount = discount

    def get_discount(self, discount):
        return self.__discount

    def get_price(self):
        if self.__discount:
            return self.__price * (1 - self.__discount)
        return self.__price


'''
Inheritance
'''


class Novel(Book):
    def __init__(self, title, quantity, author, price, pages):
        super().__init__(title, quantity, author, price)
        self.pages = pages

    def __repr__(self):
        return f"Book: {self.title}, Quantity: {self.quantity}, Author: {self.author}, Price: {self.get_price()}"



class Academic(Book):
    def __init__(self, title, quantity, author, price, branch):
        super().__init__(title, quantity, author, price)
        self.branch = branch

    def __repr__(self):
        return f"Book: {self.title}, Branch: {self.branch},  Quantity: {self.quantity}, Author: {self.author}, Price: {self.get_price()}"