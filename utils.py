import string

class DataLoader():
    def __init__(self, data_file):
        self.file = data_file

    def decode_line(self, line):
        # Remove linebreak
        line = line[:-1]
        # Remove interpunction
        line = line.translate(str.maketrans('', '', string.punctuation))
        # Split by words
        list_words = line.split('▁')[1:]
        # Reconstruct words & sentence
        sentence = ' '.join([word.replace(" ", "") for word in list_words])
        return(sentence)

    def load_data(self, stop_idx=99999999):
        data = []
        with open(self.file, 'r') as f:
            for idx, line in enumerate(f):
                if idx < stop_idx:
                    decoded_line = self.decode_line(line)
                    data += [decoded_line]
        self.data = data