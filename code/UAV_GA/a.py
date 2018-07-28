a = '11'
b = '1'
a = a[::-1]
b = b[::-1]
L1 = len(a)
L2 = len(b)
if L1 > L2:
    b = b + '0' * (L1 - L2)
else:
    a = a + '0' * (L2 - L1)
L_max = max(L1, L2)
a = list(a)
b = list(b)
out = ''
sign = 0
for i in range(L_max):
    c = int(a[i]) + int(b[i]) + sign
    if c > 1:
        sign = 1
        c = c % 2
    else:
        sign = 0
    out = out + str(c)
if sign == 1:
    out = out + '1'
out = out[::-1]
print(out)