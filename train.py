from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.metrics import Mean
from tensorflow import float32, math, cast, data, equal, reduce_sum, train, function, GradientTape, argmax
from tensorflow.keras.losses import sparse_categorical_crossentropy
from prepare_dataset import PrepareDataset
from transformer import Transformer
from time import time

h = 8
d_k = 64
d_v = 64
d_model = 512
d_ff = 2048
n = 6

epochs = 20
batch_size = 64
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
dropout_rate = 0.1

filename = 'english-german-both.pkl'
n_sentences = 10000
train_split = 0.8
val_split = 0.1

class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = cast(d_model, float32)
        self.warmup_steps = cast(warmup_steps, float32)

    def __call__(self, step_num):
        step_num = cast(step_num, float32)
        arg1 = step_num ** -0.5
        arg2 = step_num * (self.warmup_steps ** -1.5)
        return (self.d_model ** -0.5) * math.minimum(arg1, arg2)
    
optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)

dataset = PrepareDataset(n_sentences, train_split, val_split)
trainX, trainY, valX, valY, train_total, val_total, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = dataset(filename)

train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.batch(batch_size)

val_dataset = data.Dataset.from_tensor_slices((valX, valY))
val_dataset = val_dataset.batch(batch_size)

training_model = Transformer(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)

def loss_fcn(target, prediction):
    mask = math.logical_not(equal(target, 0))
    mask = cast(mask, float32)
    loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * mask
    return reduce_sum(loss)/reduce_sum(mask)

def accuracy_fcn(target, prediction):
    mask = math.logical_not(equal(target, 0))
    mask = cast(mask, float32)

    accuracy = equal(target, argmax(prediction, axis=2))
    accuracy = math.logical_and(mask, accuracy)
    accuracy = cast(accuracy, float32)

    return reduce_sum(accuracy)/reduce_sum(mask)

train_loss = Mean(name='train_loss')
train_accuracy = Mean(name='train_accuracy')

val_loss = Mean(name='val_loss')

ckpt = train.Checkpoint(model=training_model, optimizer=optimizer)
ckpt_manager = train.CheckpointManager(ckpt, './checkpoints', max_to_keep=3)

train_loss_dict = {}
val_loss_dict = {}

@function
def train_step(encoder_input, decoder_input, decoder_output):
    with GradientTape() as tape:
        prediction = training_model(encoder_input, decoder_input, training=True)
        loss = loss_fcn(decoder_output, prediction)
        accuracy = accuracy_fcn(decoder_output, prediction)
    gradients = tape.gradient(loss, training_model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))

    train_loss(loss)
    train_accuracy(accuracy)

start_time = time()
for epoch in range(epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()
    print('\nStart of the epoch', (epoch + 1))

    for step, (train_batchX, train_batchY) in enumerate(train_dataset):
        encoder_input = train_batchX[:, 1:]
        decoder_input = train_batchY[:, :-1]
        decoder_output = train_batchY[:, 1:]
        
        train_step(encoder_input, decoder_input, decoder_output)
        if step % 50 == 0:
            print("Epoch ", (epoch + 1), " Step ", step, ": Loss ", train_loss.result(), " Accuracy ", train_accuracy.result())
    
    print("Epoch ", (epoch + 1), ": Training loss ", train_loss.result(), " Training accuracy ", train_accuracy.result())

    if (epoch + 1) % 5 == 0:
        save_path = ckpt_manager.save()
        print("Saved checkpoint at epoch: ", (epoch + 1))
print("Time taken: ", (time() - start_time))
