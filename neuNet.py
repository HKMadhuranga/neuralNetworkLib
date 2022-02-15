import numpy as np
import metrics as mt


def layer_neuron_count(lay):
    return len(lay.neurons)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivation(value):
    return sigmoid(value) * (1 - sigmoid(value))


def linear_derivation(para_co_effs, der):
    return 0


class Neuron:
    inputs = None  # outputs from previous layer neurons
    outputs = None
    weighted_sum = None  # weighted sums (exact float value)

    def __init__(self, inputs=None, output_type='sigmoid'):
        self.inputs = inputs
        self.output_type = output_type

    def output(self, wts, bs, inputs=None, output_type='sigmoid'):
        self.inputs = inputs
        self.weighted_sum = mt.mat_to_num(mt.mat_mul_mat(wts, mt.mat_transpose(self.inputs))) + bs
        if output_type == 'sigmoid':
            self.outputs = sigmoid(self.weighted_sum)
            return self.outputs


class Layer:

    # layer makes, [neu(1), neu(2), neu(3),..., neu(n)]
    def __init__(self, neuron_count):
        self.neurons = [Neuron() for i in range(neuron_count)]


class Weights:
    def __init__(self, neurons_from, neurons_to):
        self.weights = [[float(1) for i in range(neurons_from)]
                        for j in range(neurons_to)]


class Bias:
    def __init__(self, neuron_count):
        self.bias_list = [float(1) for i in range(neuron_count)]


class NeuNet:
    g_wt = None
    g_lays = None
    g_bs = None
    t_inputs = None

    def connect_layers(self, inputs, lays, correct_outputs, rs=1000, rate=0.5):
        self.g_lays = lays
        self.t_inputs = inputs
        self.g_wt = self.make_weights()
        wts = self.g_wt
        self.g_bs = self.make_bias()
        bs_list = self.g_bs

        temp_input_list = inputs
        layer_out = []
        for r in range(rs):
            for i in range(len(temp_input_list)):
                temp_inputs = temp_input_list[i]
                for j in range(len(lays)):
                    lay = lays[j]
                    bs = bs_list[j]
                    for k in range(len(lay.neurons)):
                        neu = lay.neurons[k]
                        wt = wts[j].weights[k]
                        neu_bs = bs.bias_list[k]
                        neu_out = neu.output(wt, neu_bs, temp_inputs)
                        layer_out.append(neu_out)
                    temp_inputs = layer_out
                    layer_out = []

                error_list = mt.mat_subtraction(correct_outputs[i], temp_inputs)  # correct output - predict output
                cost_list = self.cost_function(error_list)  # second power list of the error_list each elements
                cost_func_derivation = mt.mat_mul_num(error_list, (-2))  # cost function is derived w.r.t. temp_inputs
                wts, bias_list = self.back_prop(temp_input_list[i],cost_func_derivation, rate)

    def predict(self, inputs):
        wts = self.g_wt
        bs_list = self.g_bs
        lays = self.g_lays

        returns = []
        if np.array(inputs).ndim == 1:
            inputs = [inputs]
        for i in range(len(inputs)):
            temp_inputs = inputs[i]
            layer_out = []
            for j in range(len(lays)):
                lay = lays[j]
                bs = bs_list[j]
                for k in range(len(lay.neurons)):
                    neu = lay.neurons[k]
                    wt = wts[j].weights[k]
                    neu_bs = bs.bias_list[k]
                    neu_out = neu.output(wt, neu_bs, temp_inputs)
                    layer_out.append(neu_out)
                temp_inputs = layer_out
                layer_out = []
            returns.append(temp_inputs)
        return returns

    def back_prop(self, inputs, cost_der_list, rate):
        lays = self.g_lays
        wts = self.g_wt
        dw = self.make_weight_copy()
        for i in range(len(wts)):
            for j in range(len(wts[i].weights)):
                for k in range(len(wts[i].weights[j])):
                    if i == 0:
                        dw[i][j][k] = self.make_dw(wts, lays, cost_der_list, i, j) * sigmoid_derivation(
                            mt.mat_to_num(lays[i].neurons[j].weighted_sum)) * inputs[k]
                    else:
                        dw[i][j][k] = self.make_dw(wts, lays, cost_der_list, i, j) * sigmoid_derivation(
                            mt.mat_to_num(lays[i].neurons[j].weighted_sum)) * lays[i - 1].neurons[k].outputs

        dbs = self.make_bias_copy()
        for i in range(len(dbs)):
            for j in range(len(dbs[i])):
                dbs[i][j] = self.make_dw(wts, lays, cost_der_list, i, j) * sigmoid_derivation(
                    mt.mat_to_num(lays[i].neurons[j].weighted_sum)) * 1

        wts = self.update_weights(dw, rate)
        bias_list = self.update_bias(dbs, rate)

        return wts, bias_list

    def update_weights(self, dwt, rate):
        wts = self.g_wt
        for i in range(len(dwt)):
            for j in range(len(dwt[i])):
                for k in range(len(dwt[i][j])):
                    wts[i].weights[j][k] = wts[i].weights[j][k] - dwt[i][j][k] * rate
        return wts

    def update_bias(self, dbs, rate):
        bs = self.g_bs
        for i in range(len(dbs)):
            for j in range(len(dbs[i])):
                bs[i].bias_list[j] = bs[i].bias_list[j] - dbs[i][j] * rate
        return bs

    def cost_function(self, input_list):
        returns = []
        if np.array(input_list).ndim == 0:
            returns.append(input_list * input_list)
        elif np.array(input_list).ndim == 1:
            for i in range(len(input_list)):
                returns.append(input_list[i] * input_list[i])
        return returns

    def make_weights(self):
        inputs = self.t_inputs
        lays = self.g_lays
        returns = [Weights(len(mt.mat_transpose(inputs)), layer_neuron_count(lays[0]))]
        for i in range(1, len(lays)):
            returns.append(Weights(layer_neuron_count(lays[i - 1]), layer_neuron_count(lays[i])))
        return returns

    def make_bias(self):
        lays = self.g_lays
        returns = []
        for i in range(len(lays)):
            returns.append(Bias(layer_neuron_count(lays[i])))
        return returns

    def make_dw(self, wts, lays, cost_der, i=0, j=0):
        pre_neu_pos = [i, j]
        if i < len(wts) - 1:
            internal_value = self.loop_generate(i + 1, lays, wts, pre_neu_pos, cost_der)
        else:
            internal_value = cost_der[j]
        return internal_value

    def loop_generate(self, cur_lay_pos, neu_lays, wt_lays, pre_neu_pos, cost_der):
        returns = 0
        if cur_lay_pos == len(neu_lays) - 1:
            for a in range(len(neu_lays[cur_lay_pos].neurons)):
                returns += sigmoid_derivation(neu_lays[cur_lay_pos].neurons[a].weighted_sum
                                              ) * wt_lays[cur_lay_pos].weights[a][pre_neu_pos[1]] * cost_der[a]
            return returns
        else:
            for a in range(len(neu_lays[cur_lay_pos].neurons)):
                pre_n_pos = [cur_lay_pos, a]
                val = self.loop_generate(cur_lay_pos + 1, neu_lays, wt_lays, pre_n_pos, cost_der)
                returns += sigmoid_derivation(neu_lays[cur_lay_pos].neurons[a].weighted_sum
                                              ) * wt_lays[cur_lay_pos].weights[a][pre_neu_pos[1]] * val
            return returns

    def make_weight_copy(self):
        wts = self.g_wt
        returns = []
        for i in range(len(wts)):
            ret1 = []
            for j in range(len(wts[i].weights)):
                ret2 = []
                for k in range(len(wts[i].weights[j])):
                    ret2.append(wts[i].weights[j][k])
                ret1.append(ret2)
            returns.append(ret1)
        return returns

    def make_bias_copy(self):
        bs = self.g_bs
        returns = []
        for i in range(len(bs)):
            ret1 = []
            for j in range(len(bs[i].bias_list)):
                ret1.append(bs[i].bias_list[j])
            returns.append(ret1)
        return returns

    def print_wts(self, wts):
        wts = self.g_wt
        for i in range(len(wts)):
            for j in range(len(wts[i].weights)):
                for k in range(len(wts[i].weights[j])):
                    print(wts[i].weights[j][k])

    def print_bs(self, bs):
        bs = self.g_bs
        for i in range(len(bs)):
            for j in range(len(bs[i].bias_list)):
                print(bs[i].bias_list[j])


training_input = [[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]]  # input training data set
training_output = [0, 0, 1, 1]  # output training data set
hidden_layers = [Layer(2), Layer(1)]
cls = NeuNet()
cls.connect_layers(training_input, hidden_layers, training_output)
print(cls.predict([[0, 0, 1],
                   [0, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]]), 'predicted value')
