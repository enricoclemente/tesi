
def strip_empty_str(strings):
    while strings and strings[-1] == "":
        del strings[-1]
    return strings