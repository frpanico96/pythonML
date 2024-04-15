from classexamples import Book, Novel, Academic

book1 = Book('Book 1', 12, 'Author 1', 120)
book2 = Book('Book 2', 18, 'Author 2', 220)
book3 = Book('Book 3', 28, 'Author 3', 320)

book1.set_discount(0.2)
book2.set_discount(0.3)

print(book1)
print(book2)
print(book3)

novel1 = Novel('Two states', 20, 'Chetan Bhagat', 200, 187)
novel1.set_discount(0.20)

academic1 = Academic('Python Foundations', 12, 'PSF', 655, 'IT')

print(novel1)
print(academic1)