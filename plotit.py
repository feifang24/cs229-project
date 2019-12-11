import matplotlib.pyplot as plt

xs = ['OG','SD12800','SD6400','SD3200','SD1600','SD800','NWD00','NWD01','WD00','WD01']
overall_acc = [0.8678,0.8497,0.8085,0.7958,0.7120,0.7037,0.7011,0.7494,0.6505,0.7092]
pos_acc = [0.8572,0.8324,0.8208,0.8011,0.7522,0.8531,0.7777,0.8253,0.4938,0.6379]
neg_acc = [0.8785,0.8617,0.7962,0.7906,0.6719,0.5542,0.6245,0.6735,0.8071,0.7805]

plt.plot(xs, overall_acc, label='Overall Accuracy')
plt.plot(xs, pos_acc, label='Accuracy over positive samples')
plt.plot(xs, neg_acc, label='Accuracy over negative samples')

plt.legend()
plt.title('LSTM')
plt.ylabel('Accuracy')
plt.show()