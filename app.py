count = 0
for i in range(1, 10):
    if (i % 2) == 0:
        print(i)
        count += 1
print(f"We have {count} even numbers")


def add(a, b):
    return a+b


print(add(2, 3))
