from pickle import load, dump, HIGHEST_PROTOCOL
from numpy.random import shuffle
from numpy import savetxt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64, shape

class PrepareDataset:
    def __init__(self, n_sentences, train_split, val_split, **kwargs):
        super().__init__(**kwargs)
        self.n_sentences = n_sentences
        self.train_split = train_split
        self.val_split = val_split

    def create_tokenizer(self, dataset):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(dataset)
        return tokenizer
    
    def find_seq_length(self, dataset):
        return max(len(seq.split()) for seq in dataset)

    def find_vocab_size(self, tokenizer, dataset):
        tokenizer.fit_on_texts(dataset)
        return len(tokenizer.word_index) + 1
    
    def encode_pad(self, dataset, tokenizer, seq_length):
        x = tokenizer.texts_to_sequences(dataset)
        x = pad_sequences(x, maxlen=seq_length, padding='post')
        x = convert_to_tensor(x, dtype=int64)
        return x

    def save_tokenizer(self, tokenizer, name):
        with open(name + '_tokenizer.pkl', 'wb') as handle:
            dump(tokenizer, handle, protocol=HIGHEST_PROTOCOL)

    def __call__(self, filename, **kwargs):
        clean_dataset = load(open(filename, 'rb'))

        dataset = clean_dataset[:self.n_sentences, :]
        for i in range(dataset[:, 0].size):
            dataset[i, 0] = '<START> ' + dataset[i, 0] + ' <EOS>'
            dataset[i, 1] = '<START> ' + dataset[i, 1] + ' <EOS>'
        
        shuffle(dataset)

        train = dataset[:int(self.n_sentences * self.train_split)]
        val = dataset[int(self.n_sentences * self.train_split): int(self.n_sentences * (1 - self.val_split))]
        test = dataset[int(self.n_sentences * (1 - self.val_split)):]

        enc_tokenizer = self.create_tokenizer(dataset[:, 0])
        enc_seq_length = self.find_seq_length(dataset[:, 0])
        enc_vocab_size = self.find_vocab_size(enc_tokenizer, train[:, 0])

        dec_tokenizer = self.create_tokenizer(dataset[:, 1])
        dec_seq_length = self.find_seq_length(dataset[:, 1])
        dec_vocab_size = self.find_vocab_size(dec_tokenizer, train[:, 1])

        trainX = self.encode_pad(train[:, 0], enc_tokenizer, enc_seq_length)
        trainY = self.encode_pad(train[:, 1], dec_tokenizer, dec_seq_length)

        valX = self.encode_pad(val[:, 0], enc_tokenizer, enc_seq_length)
        valY = self.encode_pad(val[:, 1], dec_tokenizer, dec_seq_length)

        self.save_tokenizer(enc_tokenizer, 'enc')
        self.save_tokenizer(dec_tokenizer, 'dec')

        savetxt('test_dataset.txt', test, fmt='%s')

        return (trainX, trainY, valX, valY, train, val, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size)
        