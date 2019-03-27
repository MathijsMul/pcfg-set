"""
Add brackets to PCFG SET data.

"""

def place_brackets(seq):
    if type(seq) is str:
        seq = seq.split()
    seq.append("END")
    queue = []
    new_seq = []
    for token in seq:
        if token == "append" or token == "prepend":
            new_seq.append(token)
            new_seq.append("(")
            queue.append(["two-place", 0])
        elif token in ["copy", "reverse", "shift", "echo"]:
            new_seq.append(token)
            new_seq.append("(")
            queue.append(["one-place", 0])
        elif token == "," or token == "END":
            while len(queue) > 0:
                if queue[-1][0] == "one-place":
                    _ = queue.pop()
                    new_seq.append(")")
                elif queue[-1][0] == "two-place" and queue[-1][1] == 0:
                    queue[-1][1] = 1
                    break
                elif queue[-1][0] == "two-place" and queue[-1][1] == 1:
                    new_seq.append(")")
                    _ = queue.pop()
            if token == "," : new_seq.append(token)
        else:
            new_seq.append(token)
    assert new_seq.count("(") == new_seq.count(")"), "Number of opening and closing brackets do not match."
    return " ".join(new_seq)

if __name__ == "__main__":
    with open("pcfg_data_transformed.txt") as f1, open("pcfg_data_transformed_brackets.txt") as f2:
        for line1, line2 in zip(f1, f2):
            [source1, _] = line1.split("\t")
            [source2, _] = line2.split("\t")
            assert source2 == place_brackets(source1), "{} does not equal {}".format(source1, source2)