import argparse
import random
import tasks
import sys

from naturalize import DataNaturalization

class MarkovTree():
    """
    PCFG
    """
    def __init__(self, unary_functions, binary_functions, alphabet, prob_unary, prob_func, lengths, placeholders, omit_brackets):
        sys.setrecursionlimit(50)

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

def generate_data(pcfg_tree, total_samples, data_root, random_probs):
    t = pcfg_tree
    output_file_name = data_root + '.txt'
    output_file = open(output_file_name, 'w')

    for i in range(total_samples):
        if random_probs:
            t.set_probabilities(prob_unary=random.random(),
                                prob_func = random.random())
        try:
            tree = t.build()
            written_tree = t.write(tree)

            # Control maximum tree size
            if len(written_tree) < 500:
                output_file.write(written_tree+ '\t' + ' '.join(t.evaluate_tree(tree)) + '\n')
        except RecursionError:
            pass

    return(output_file_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='The PCFG SET task to use', default='default')
    parser.add_argument('--alphabet_ratio', type=int, help='How many times to increase alphabet size', default=1)
    parser.add_argument('--random_probs', action='store_true', help='Use different random probabilities for each sample')
    parser.add_argument('--prob_unary', type=float, help='P(unary|function)', default=0.75)
    parser.add_argument('--prob_func', type=float, help='P(function|argument)', default=0.25)
    parser.add_argument('--lengths', type=int, help='Lengths of string arguments', default=[2, 3, 4, 5])
    parser.add_argument('--nr_samples', type=int, help='Number of samples to generate', default=2500)
    parser.add_argument('--no_split', action='store_true', help='Do not split into train and test yet')
    parser.add_argument('--train_ratio', type=float, help='Fraction of generated data to use for training', default=0.8)
    parser.add_argument('--data_root', type=str, help='Data path root')
    parser.add_argument('--placeholder_args', action='store_true', help='Generate data with placeholder arguments, containing only X characters')
    parser.add_argument('--omit_brackets', action='store_true', help='Do not use brackets')
    parser.add_argument('--naturalize', action='store_true', help='Impose natural language distribution on data')
    parser.add_argument('--nl_file', type=str, help='Natural language file to mimic distribution from')
    opt = parser.parse_args()

    if not opt.data_root:
        parser.error('Data path root required.')

    if opt.naturalize:
        opt.random_probs = True
        opt.placeholder_args = True
        opt.no_split = True

    task = getattr(tasks, opt.task)

    unary_functions = task.unary_functions
    binary_functions = task.binary_functions
    alphabet = [letter + str(i) for letter in task.alphabet for i in range(1, opt.alphabet_ratio + 1)]

    pcfg_tree_generator = MarkovTree(unary_functions=unary_functions,
                   binary_functions=binary_functions,
                   alphabet=alphabet,
                   prob_unary=opt.prob_unary,
                   prob_func=opt.prob_func,
                   lengths=opt.lengths,
                   placeholders=opt.placeholder_args,
                   omit_brackets=opt.omit_brackets)


    output_file = generate_data(pcfg_tree=pcfg_tree_generator,
                  total_samples=opt.nr_samples,
                  data_root=opt.data_root,
                  random_probs=opt.random_probs)

    if opt.naturalize:
        naturalizer = DataNaturalization(alphabet=alphabet,
                                         unary_functions=unary_functions,
                                         binary_functions=binary_functions)

        depth_intervals = range(1, 2)
        length_intervals = range(1, 6)

        opt_kl_div = 999
        for depth_interval in depth_intervals:
            for length_interval in length_intervals:
                try:
                    # Default: mimic English WMT test file
                    kl_div, new_output_file = naturalizer.force_dist_on_data(
                        data_gold_dist=opt.nl_file,
                        data_to_be_transformed=output_file,
                        depth_interval=depth_interval,
                        length_interval=length_interval)
                    if kl_div < opt_kl_div:
                        opt_kl_div = kl_div
                        opt_dep_len = (depth_interval, length_interval)
                        opt_file = new_output_file
                except:
                    pass

        print('Best results for depth_interval={0}, length_interval={1}'.format(opt_dep_len[0], opt_dep_len[1]))

        naturalizer.finalize(file=opt_file)

# python3 generate.py --alphabet_ratio 20 --random_probs --nr_samples 100000 --no_split --data_root 'pcfg_10funcs_520letters_100K' --placeholder_args --naturalize