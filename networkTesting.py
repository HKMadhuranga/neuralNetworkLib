import neuNet

cls = neuNet.NeuNet()

training_input = [[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]]  # input training data set
training_output = [0, 0, 1, 1]  # output data set
layers = [neuNet.Layer(2)]
# hidden layers
cls.connect_layers(training_input, layers, training_output)
tr_in, tr_out, test_in, test_out = cls.split_training_testing(training_input, training_output)
print(cls.predict([[0, 0, 1],
                   [0, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]]))
# cls.save_data('myname')
# cls.load_data('myname')
print(cls.accuracy(training_input, training_output))
