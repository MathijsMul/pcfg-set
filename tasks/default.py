alphabet = ['A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L',
            'M', 'N', 'O', 'P',
            'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X',
            'Y', 'Z']

# UNARY

def copy(sequence):
    return (sequence)

def reverse(sequence):
    return (sequence[::-1])

def shift(sequence):
    return (sequence[1:] + [sequence[0]])

def echo(sequence):
    return (sequence + [sequence[-1]])

# def echo_first(sequence):
#     return ([sequence[0]] + sequence)

def swap_first_last(sequence):
    return([sequence[-1]] + sequence[1:-1] + [sequence[0]])

def repeat(sequence):
    return(sequence + sequence)

# BINARY

def append(sequence1, sequence2):
    return (sequence1 + sequence2)

def prepend(sequence1, sequence2):
    return (sequence2 + sequence1)

def remove_first(sequence1, sequence2):
    return (sequence2)

def remove_second(sequence1, sequence2):
    return(sequence1)

# def interleave(sequence1, sequence2):
#     len_1, len_2 = len(sequence1), len(sequence2)
#
#     if len_1 > len_2:
#         sequence2 += ['' for i in range(len_1 - len_2)]
#     elif len_2 > len_1:
#         sequence1 += ['' for i in range(len_2 - len_1)]
#
#     seqs = [sequence1, sequence2]
#     interleave = [x for t in zip(*seqs) for x in t]
#     result = [x for x in interleave if x!= '']
#     return(result)

unary_functions = [copy, reverse, shift, echo, swap_first_last, repeat]
binary_functions = [append, prepend, remove_first, remove_second]
