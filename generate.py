import argparse
import random

# Unary functions
def copy(string):
    return(string)

def reverse(string):
    return(string[::-1])

def shift(string):
    return(string[1:] + [string[0]])

def echo(string):
    return(string + [string[-1]])

# Binary functions
def append(string1, string2):
    return(string1 + string2)

def prepend(string1, string2):
    return(string2 + string1)

class MarkovTree():
    """
    PCFG
    """
    def __init__(self, alphabet, prob_unary, prob_func, lengths):
        self.alphabet = alphabet

        self.prob_unary = prob_unary
        self.prob_binary = 1 - self.prob_unary

        self.prob_func = prob_func
        self.prob_str = 1 - self.prob_func

        self.lengths = lengths

        self.string_arguments = []

    def function_next(self):
        # Determine item following arbitrary function call
        if random.random() < self.prob_unary:
            return([random.choice(UNARY), self.unary_next()])
        else:
            [arg1, arg2] = self.binary_next()
            return([random.choice(BINARY), arg1, arg2])

    def unary_next(self):
        # Determine item following unary function call
        if random.random() < self.prob_func:
            return(self.function_next())
        else:
            return (self.string_argument())

    def binary_next(self):
        # Determine items following binary function call
        next = []

        for i in range(2):
            if random.random() < self.prob_str:
                next += [self.string_argument()]
            else:
                next += [self.function_next()]
        return(next)

    def string_argument(self):
        candidate = [random.choice(self.alphabet) for i in range(random.choice(self.lengths))]
        if not candidate in self.string_arguments:
            # make sure that the same string arguments do not occur
            self.string_arguments += [candidate]
            return(candidate)
        else:
            return(self.string_argument())

    def build(self):
        # Always start with function call
        tree = self.function_next()
        return(tree)

    def generate_data(self, nr_samples):
        data = [self.build() for i in range(nr_samples)]
        return(data)

    def evaluate_tree(self, tree):
        # Evaluate output
        if all(isinstance(item, str) for item in tree):
            return(tree)
        if tree[0] in UNARY:
            return(tree[0](self.evaluate_tree(tree[1])))
        elif tree[0] in BINARY:
            return(tree[0](self.evaluate_tree(tree[1]), self.evaluate_tree(tree[2])))

    def write(self, tree):
        # Convert tree to string for data file
        if all(isinstance(item, str) for item in tree):
            return(' '.join(tree))
        if tree[0] in UNARY:
            return(tree[0].__name__ + ' ( ' + self.write(tree[1])) + ' )'
        elif tree[0] in BINARY:
            return(tree[0].__name__ + ' ( ' + self.write(tree[1]) + ' , ' + self.write(tree[2]) + ' )')

def generate_data(alphabet, prob_unary, prob_func, lengths, total_samples, train_ratio, data_root):
    t = MarkovTree(alphabet=alphabet, prob_unary=prob_unary, prob_func=prob_func, lengths=lengths)

    train_file = data_root + '_train.txt'
    test_file = data_root + '_test.txt'

    with open(train_file, 'w') as f_train:
        with open(test_file, 'w') as f_test:
            for i in range(total_samples):
                tree = t.build()
                if random.random() < train_ratio:
                    f_train.write(t.write(tree) + '\t' + ' '.join(t.evaluate_tree(tree)) + '\n')
                else:
                    f_test.write(t.write(tree) + '\t' + ' '.join(t.evaluate_tree(tree)) + '\n')

if __name__ == '__main__':
    ALPHABET = ['A', 'B', 'C', 'D',
                'E', 'F', 'G', 'H',
                'I', 'J', 'K', 'L',
                'M', 'N', 'O', 'P',
                'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X',
                'Y', 'Z']

    PROB_UNARY = 0.75
    #PROB_BINARY = 1 - PROB_UNARY
    PROB_FUNC = 0.25
    #PROB_STR = 1 - PROB_FUNC

    LENGTHS = [2, 3, 4]  # TODO: make this adaptive, also dependent on distribution

    # so far: same weight to functions within classes
    UNARY = [copy, reverse, shift, echo]
    BINARY = [append, prepend]
    FUNC = UNARY + BINARY

    parser = argparse.ArgumentParser()
    parser.add_argument('--alphabet', type=list, help='Alphabet', default=ALPHABET)
    parser.add_argument('--prob_unary', type=float, help='P(unary|function)', default=PROB_UNARY)
    parser.add_argument('--prob_func', type=float, help='P(function|argument)', default=PROB_FUNC)
    parser.add_argument('--lengths', type=int, help='Lengths of string arguments', default=LENGTHS)
    parser.add_argument('--nr_samples', type=int, help='Number of samples to generate', default=2500)
    parser.add_argument('--train_ratio', type=float, help='Fraction of generated data to use for training', default=0.8)
    parser.add_argument('--data_root', type=str, help='Data path root')
    opt = parser.parse_args()

    if not opt.data_root:
        parser.error('Data path root required.')

    generate_data(alphabet=opt.alphabet,
                  prob_unary=opt.prob_unary,
                  prob_func=opt.prob_func,
                  lengths=opt.lengths,
                  total_samples=opt.nr_samples,
                  train_ratio=opt.train_ratio,
                  data_root=opt.data_root)