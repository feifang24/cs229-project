from dataprocessor import ImdbProcessor

'''
This file is not needed. It demonstrates how to use ImdbProcessor.
'''

def inspect(listOfInputExamples, k=2):
  '''
  For demo only.
  '''
  print("##########################")
  print("There are %d examples in total, inspecting top %d..." % (len(listOfInputExamples), k))
  for inputExample in listOfInputExamples[:k]:
    print("#####")
    print(inputExample.text_a, inputExample.label)


processor = ImdbProcessor("data")
inspect(processor.get_train_examples("sd100"))
inspect(processor.get_test_examples())