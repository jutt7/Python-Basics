import csv

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


# file = open('test_file.txt', 'w')
# file.write('hello world')
# file.close()

# file_csv = open('people.csv', 'a', newline='')
# tup1 = ('awais', 28)
# writer = csv.writer(file_csv)
# writer.writerow(tup1)
# file_csv.close

file = open('people.csv', 'r')

for lines in csv.reader(file):
    print(lines)
