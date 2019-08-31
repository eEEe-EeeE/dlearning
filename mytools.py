def gcd(a, b):
    mod = a % b
    while mod != 0:
        a = b
        b = mod
        mod = a % b
    return b


def mcm(a, b):
    return a * b / gcd(a, b)


def func(n, fg):
    f = fg[0]
    g = fg[1]
    f.reverse()
    g.reverse()
    index = -1
    for e in f:
        index += 1
        if e != 0:
            break
    fm = index
    index = -1
    for e in g:
        index += 1
        if e != 0:
            break
    gm = index
    if fm < gm:
        return '1/0'
    elif fm > gm:
        return '0'
    else:
        d = gcd(f[fm], g[gm])
        if f[fm] > g[gm]:
            return str(f[fm] // d)
        else:
            return str(g[gm] // d)