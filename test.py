x = 0.1
y = x
z = x
a = x

for i in range(300):
    x = x * 0.96 ** ((i + 1) / 300)

for i in range(300):
    y = y * 0.97 ** ((i + 1) / 300)

for i in range(300):
    z = z * 0.98 ** ((i + 1) / 300)

for i in range(300):
    a = a * 0.99 ** ((i + 1) / 300)

print(x)
print(y)
print(z)
print(a)
