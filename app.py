count = 0
for i in range(1, 10):
    if (i % 2) == 0:
        print(i)
        count += 1
print(f"We have {count} even numbers")


def add(a, b):
    return a+b


print(add(2, 3))


def xargs(*numbers):
    print(numbers)


xargs(2, 3, 4, 5, 6)


capitals = {"USA": "Washingotn",
            "Pakistan": "Islamabad",
            "China": "Beijing",
            "Russia": "Moscow"}

for key in capitals.keys():
    print(key)
for key, value in capitals.items():
    print(f"{key}: {value}")
