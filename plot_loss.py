from pickle import load
from matplotlib.pylab import plt
from numpy import arange

train_loss = load(open('train_loss.pkl', 'rb'))
val_loss = load(open('val_loss.pkl', 'rb'))

train_values = train_loss.values()
val_values = val_loss.values()

print(len(train_values))

epochs = range(1, 21)

plt.plot(epochs, train_values, label='Training Loss')
plt.plot(epochs, val_values, label='Validation Loss')

plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.xticks(arange(0, 21, 2))

plt.legend(loc='best')
plt.show()