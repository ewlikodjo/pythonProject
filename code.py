from string import digits
from itertools import product

n = int(input("enter the number: "))

for passcode in product(digits, repeat=n):
    print(*passcode)