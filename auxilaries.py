

def glance_list(li, Nhead=2, Ntail=1):
    head = str(li[:Nhead])[:-1]
    body = ', ... ,'
    tail = str(li[-Ntail:])[1:]
    length = len(li)
    return f'{head + body + tail}, len = {length}'


def glance_dict(di, Nhead=5):
    return {k: v for i, (k, v) in enumerate(di.items()) if i < Nhead}
