from pickle import load
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import Module, convert_to_tensor, int64, TensorArray, argmax, newaxis, transpose
from transformer import Transformer

h = 8
d_k = 64
d_v = 64
d_model = 512
d_ff = 2048
n = 6

enc_seq_length = 7
dec_seq_length = 12
enc_vocab_size = 2404
dec_vocab_size = 3864

inferencing_model = Transformer(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, 0)

class Translate(Module):
    def __init__(self, inferencing_model, **kwargs):
        super().__init__(**kwargs)
        self.transformer = inferencing_model

    def load_tokenizer(self, name):
        with open(name, 'rb') as handle:
            return load(handle)
        
    def __call__(self, sentence):
        if len(sentence) != 1:
            print('Works only for one sentence for now')
            return
        sentence[0] = "<START> " + sentence[0] + " <EOS>"

        enc_tokenizer = self.load_tokenizer('enc_tokenizer.pkl')
        dec_tokenizer = self.load_tokenizer('dec_tokenizer.pkl')

        encoder_input = enc_tokenizer.texts_to_sequences(sentence)
        encoder_input = pad_sequences(encoder_input, maxlen=enc_seq_length, padding='post')
        encoder_input = convert_to_tensor(encoder_input, dtype=int64)

        output_start = dec_tokenizer.texts_to_sequences(['<START>'])
        output_start = convert_to_tensor(output_start[0], dtype=int64)

        output_end = dec_tokenizer.texts_to_sequences(['<EOS>'])
        output_end = convert_to_tensor(output_end[0], dtype=int64)

        decoder_output = TensorArray(dtype=int64, size=0, dynamic_size=True)
        decoder_output = decoder_output.write(0, output_start)

        for i in range(dec_seq_length):
            prediction = self.transformer(encoder_input, transpose(decoder_output.stack()), training=False)
            prediction = prediction[:, -1, :]

            predicted_id = argmax(prediction, axis=-1)

            decoder_output = decoder_output.write(i + 1, predicted_id)

            if predicted_id == output_end:
                break
        output = transpose(decoder_output.stack())[0]
        output = output.numpy()

        output_str = []

        for i in range(output.shape[0]):
            key = output[i]
            output_str.append(dec_tokenizer.index_word[key])

        return output_str

sentence = ['im thirsty']

inferencing_model.load_weights('weights/wghts16.ckpt')

translator = Translate(inferencing_model)

print(translator(sentence))