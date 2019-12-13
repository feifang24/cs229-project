import matplotlib.pyplot as plt

def plot_lstm():
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

def plot_bert():
  xs = ['OG','SD12800','SD6400','SD3200','SD1600','SD800','NWD00','WD00','WD01']
  test_acc = [0.883,0.880,0.863,0.865,0.854,0.823,0.732,0.762,0.795]
  dev_acc = [0.901,0.893,0.876,0.869,0.854,0.841,0.754,0.775,0.816]

  plt.plot(xs, dev_acc, label='Dev Set Accuracy')
  plt.plot(xs, test_acc, label='Test Set Accuracy')

  plt.legend()
  plt.title('Fine-tuned Bert')
  plt.ylabel('Accuracy')
  plt.show()

plot_bert()