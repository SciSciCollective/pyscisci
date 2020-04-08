def load_int(v):
    try:
        return int(v)
    except ValueError:
        return None

def load_long(v):
    try:
        return long(v)
    except ValueError:
        return None

def load_float(v):
    try:
        return float(v)
    except ValueError:
        return None

def load_str(v):
    try:
        return str(v)
    except ValueError:
        return None