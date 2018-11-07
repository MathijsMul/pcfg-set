import argparse
import random
import tasks
import sys

sys.setrecursionlimit(50)

class MarkovTree():
    """
    PCFG
    """
    def __init__(self, unary_functions, binary_functions, alphabet, prob_unary, prob_func, lengths, placeholders, omit_brackets):
        self.unary_functions = unary_functions
        self.binary_functions = binary_functions
        self.all_functions = self.unary_functions + self.binary_functions

        self.alphabet = alphabet

        self.set_probabilities(prob_unary, prob_func)

        self.lengths = lengths
        self.string_arguments = []
        self.arg_length_counts = {i : len(self.alphabet) ** i for i in self.lengths}

        self.placeholders = placeholders
        self.omit_brackets = omit_brackets

    def set_probabilities(self, prob_unary, prob_func):
        self.prob_unary = prob_unary
        self.prob_binary = 1 - self.prob_unary

        self.prob_func = prob_func
        self.prob_str = 1 - self.prob_func

    def function_next(self):
        # Determine item following arbitrary function call
        if random.random() < self.prob_unary:
            return([random.choice(self.unary_functions), self.unary_next()])
        else:
            [arg1, arg2] = self.binary_next()
            return([random.choice(self.binary_functions), arg1, arg2])

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
        if not self.placeholders:
            candidate = [random.choice(self.alphabet) for i in range(random.choice(self.lengths))]
            if not candidate in self.string_arguments:
                # make sure that the same string arguments do not occur
                self.string_arguments += [candidate]
                return(candidate)
            else:
                return(self.string_argument())
        else:
            candidate_len = random.choice(self.lengths)
            if not self.arg_length_counts[candidate_len] == 0:
                self.arg_length_counts[candidate_len] -= 1
                return(['X' for i in range(candidate_len)])
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
        if tree[0] in self.unary_functions:
            return(tree[0](self.evaluate_tree(tree[1])))
        elif tree[0] in self.binary_functions:
            return(tree[0](self.evaluate_tree(tree[1]), self.evaluate_tree(tree[2])))

    def write(self, tree):
        # Convert tree to string for data file
        if all(isinstance(item, str) for item in tree):
            return(' '.join(tree))
        if tree[0] in self.unary_functions:
            if self.omit_brackets:
                return (tree[0].__name__ + ' ' + self.write(tree[1]))
            else:
                return(tree[0].__name__ + ' ( ' + self.write(tree[1])) + ' )'
        elif tree[0] in self.binary_functions:
            if self.omit_brackets:
                return (tree[0].__name__ + ' ' + self.write(tree[1]) + ' , ' + self.write(tree[2]))
            else:
                return(tree[0].__name__ + ' ( ' + self.write(tree[1]) + ' , ' + self.write(tree[2]) + ' )')

def generate_data(task, random_probs, prob_unary, prob_func, lengths, total_samples, no_split, train_ratio, data_root, placeholders=False, omit_brackets=False):
    # so far: same weight to functions within classes
    unary_functions = task.unary_functions
    binary_functions = task.binary_functions
    alphabet = task.alphabet

    t = MarkovTree(unary_functions=unary_functions,
                   binary_functions=binary_functions,
                   alphabet=alphabet,
                   prob_unary=prob_unary,
                   prob_func=prob_func,
                   lengths=lengths,
                   placeholders=placeholders,
                   omit_brackets=omit_brackets)

    if no_split:
        file = open(data_root + '.txt', 'w')
    else:
        train_file = open(data_root + '_train.txt', 'w')
        test_file = open(data_root + '_test.txt', 'w')

    for i in range(total_samples):
        if random_probs:
            t.set_probabilities(prob_unary=random.random(),
                                prob_func = random.random())
                                #prob_func=random.uniform(0.0, 0.9)) # keep prob_func a bit low to prevent recursion errors
        try:
            tree = t.build()
            written_tree = t.write(tree)
            # Control maximum tree size
            if len(written_tree) < 300:
                if no_split:
                    file.write(written_tree+ '\t' + ' '.join(t.evaluate_tree(tree)) + '\n')
                else:
                    if random.random() < train_ratio:
                        train_file.write(written_tree + '\t' + ' '.join(t.evaluate_tree(tree)) + '\n')
                    else:
                        test_file.write(written_tree + '\t' + ' '.join(t.evaluate_tree(tree)) + '\n')
        except RecursionError:
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='The PCFG SET task to use', default='default')
    parser.add_argument('--random_probs', action='store_true', help='Use different random probabilities for each sample')
    # TODO: distribution input lengths
    parser.add_argument('--prob_unary', type=float, help='P(unary|function)', default=0.75)
    parser.add_argument('--prob_func', type=float, help='P(function|argument)', default=0.25)
    parser.add_argument('--lengths', type=int, help='Lengths of string arguments', default=[2, 3, 4, 5]) # TODO: make this adaptive, also dependent on distribution
    parser.add_argument('--nr_samples', type=int, help='Number of samples to generate', default=2500)
    parser.add_argument('--no_split', action='store_true', help='Do not split into train and test yet')
    parser.add_argument('--train_ratio', type=float, help='Fraction of generated data to use for training', default=0.8)
    parser.add_argument('--data_root', type=str, help='Data path root')
    parser.add_argument('--placeholder_args', action='store_true', help='Generate data with placeholder arguments, containing only X characters')
    parser.add_argument('--omit_brackets', action='store_true', help='Do not use brackets')
    opt = parser.parse_args()

    if not opt.data_root:
        parser.error('Data path root required.')

    if not opt.task:
        parser.error('Specify PCFG SET task.')

    task = getattr(tasks, opt.task)

    generate_data(task=task,
                  random_probs=opt.random_probs,
                  prob_unary=opt.prob_unary,
                  prob_func=opt.prob_func,
                  lengths=opt.lengths,
                  total_samples=opt.nr_samples,
                  no_split=opt.no_split,
                  train_ratio=opt.train_ratio,
                  data_root=opt.data_root,
                  placeholders=opt.placeholder_args,
                  omit_brackets=opt.omit_brackets)