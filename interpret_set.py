from collections import defaultdict
import tasks

task_name = 'default'

task = getattr(tasks, task_name)
UNARY = task.unary_functions
BINARY = task.binary_functions
FUNC = UNARY + BINARY

UNARY_DICT = {func.__name__ : func for func in UNARY}
BINARY_DICT = {func.__name__ : func for func in BINARY}

FUNC_NAMES = set([func.__name__ for func in FUNC])

def get_inputs(file):
    inputs = []
    with open(file, 'r') as f:
        for idx, line in enumerate(f):
            input = line.split('\t')[0]
            inputs += [input.split()]
    return(inputs)

def is_basestring(input):
    return(list(set(input) & FUNC_NAMES) == [])

def interpret(input):
    #print(input)
    func_name = input[0]

    if func_name in UNARY_DICT.keys():
        func = UNARY_DICT[func_name]
        arg = get_arguments(input, nr_arguments=1)
        if is_basestring(arg):
            # base case
            return(func(arg))
        else:
            return(func(interpret(arg)))

    elif func_name in BINARY_DICT.keys():
        func = BINARY_DICT[func_name]
        arg1, arg2 = get_arguments(input, nr_arguments=2)
        if is_basestring(arg1):
            if is_basestring(arg2):
                return(func(arg1, arg2))
            else:
                return(func(arg1, interpret(arg2)))
        else:
            if is_basestring(arg2):
                return(func(interpret(arg1), arg2))
            else:
                return(func(interpret(arg1), interpret(arg2)))

def get_arguments(input, nr_arguments):

    if nr_arguments == 1:
        arg = []
        idx = 2
        open_bracket_count = 1
        while open_bracket_count != 0:
            unit = input[idx]
            arg += [unit]
            if unit == '(':
                open_bracket_count += 1
            if unit == ')':
                open_bracket_count -= 1
            idx += 1
        arg = arg[:-1]
        return(arg)

    elif nr_arguments == 2:
        arg1, arg2 = [], []
        idx = 2
        open_bracket_count = 1

        while not (input[idx] == ',' and open_bracket_count == 1):
            unit = input[idx]
            if unit == '(':
                open_bracket_count += 1
            if unit == ')':
                open_bracket_count -= 1
            arg1 += [unit]
            idx += 1
        else:
            while open_bracket_count != 0:
                unit = input[idx]
                if unit == '(':
                    open_bracket_count += 1
                if unit == ')':
                    open_bracket_count -= 1
                arg2 += [input[idx]]
                idx += 1
        arg2 = arg2[1:-1]

        return(arg1, arg2)

def is_basecall(sample):
    func = sample[0]
    return((func in UNARY_DICT.keys() and is_basestring(sample[1])) or (func in BINARY_DICT.keys() and is_basestring(sample[1]) and is_basestring(sample[2])))

def get_substructures(sample):
    sub_dict = defaultdict(list)
    global_bracket_count = 1
    for idx, item in enumerate(sample):
        if item == '(':
            global_bracket_count += 1
        if item == ')':
            global_bracket_count -= 1
        if item in FUNC_NAMES:
            if item in UNARY_DICT.keys():
                subs = []
                local_bracket_count = 1
                next_idx = idx + 2
                open_idx = next_idx
                while local_bracket_count != 0:
                    if sample[next_idx] == '(':
                        local_bracket_count += 1
                    if sample[next_idx] == ')':
                        local_bracket_count -= 1
                    next_idx += 1
                subs += [sample[open_idx : next_idx - 1]]
            elif item in BINARY_DICT.keys():
                subs = []
                local_bracket_count = 1
                next_idx = idx + 2
                open_idx = next_idx
                while local_bracket_count != 0:
                    if sample[next_idx] == '(':
                        local_bracket_count += 1
                    if sample[next_idx] == ')':
                        local_bracket_count -= 1
                    if local_bracket_count == 1 and sample[next_idx] == ',':
                        comma_idx = next_idx
                    next_idx += 1
                subs += [sample[open_idx : comma_idx], sample[comma_idx + 1 : next_idx - 1]]
            sub_dict[global_bracket_count] += [sub for sub in subs if not is_basestring(sub)]
    return(dict(sub_dict))

def get_substructures_by_level(filein, fileout, level):
    inputs = get_inputs(filein)
    with open(fileout, 'w') as fout:
        for input in inputs:
            sub_dict = get_substructures(input)
            if level in sub_dict.keys():
                subs = sub_dict[level]
                for sub in subs:
                    output = interpret(sub)
                    fout.write(' '.join(sub) + '\t' + ' '.join(output))
                    fout.write('\n')

# for i in [1,2,3]:
#     for d in ['train', 'test']:
#         get_substructures_by_level('markov_sentences_' + d + '.txt', 'data/markov_sub/' + d + '/markov_' + d + '_sub' + str(i) + '.txt', level=i)