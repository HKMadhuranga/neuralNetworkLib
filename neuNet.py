import numpy as np
import metrics as mt
import random as rn


def layer_neuron_count(lay):
    return len(lay.neurons)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivation(value):
    return sigmoid(value) * (1 - sigmoid(value))


class Neuron:
    inputs = None
    outputs = None
    weighted_sum = None

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
    def __init__(self, neuron_count=0):
        self.neurons = [Neuron() for i in range(neuron_count)]


class Weights:
    def __init__(self, neurons_from=0, neurons_to=0):
        self.weights = [[float(1) for i in range(neurons_from)]
                        for j in range(neurons_to)]


class Bias:
    def __init__(self, neuron_count=0):
        self.bias_list = [float(1) for i in range(neuron_count)]


def cost_function(input_list):
    returns = []
    if np.array(input_list).ndim == 0:
        returns.append(input_list * input_list)
    elif np.array(input_list).ndim == 1:
        for i in range(len(input_list)):
            returns.append(input_list[i] * input_list[i])
    return returns


class NeuNet:
    __g_wt = []
    __g_lays = []
    __g_bs = []
    __t_inputs = None

    def connect_layers(self, inputs, lays, correct_outputs, rs=1000, rate=0.5):
        if np.array(correct_outputs).ndim == 1:
            lays.append(Layer(1))
        elif np.array(correct_outputs).ndim == 2:
            lays.append(Layer(len(correct_outputs[0])))
        self.__g_lays = lays
        self.__t_inputs = inputs
        self.__g_wt = self.__make_weights()
        wts = self.__g_wt
        self.__g_bs = self.__make_bias()
        bs_list = self.__g_bs

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
                error_list = mt.mat_subtraction(correct_outputs[i], temp_inputs)
                # cost_list = cost_function(error_list)
                cost_func_derivation = mt.mat_mul_num(error_list, (-2))
                wts, bias_list = self.__back_prop(temp_input_list[i], cost_func_derivation, rate)

    def predict(self, inputs):
        wts = self.__g_wt
        bs_list = self.__g_bs
        lays = self.__g_lays

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

    def accuracy(self, inputs, outputs):
        predict_count = 0
        total_count = 0
        predict_val = self.predict(inputs)
        temp_list = []
        if np.array(outputs).ndim == 1:
            for i in range(len(outputs)):
                temp_list.append([outputs[i]])
            outputs = temp_list

        for i in range(len(predict_val)):
            for j in range(len(predict_val[i])):
                if round(predict_val[i][j]) == round(outputs[i][j]):
                    predict_count += 1
                total_count += 1
        return predict_count / total_count

    def split_training_testing(self, inputs, outputs, fraction=0.75, shuffle_data=False):
        return_training_inputs = []
        return_training_outputs = []
        return_testing_inputs = []
        return_testing_outputs = []
        for i in range(int(len(inputs) * fraction)):
            return_training_inputs.append(inputs[i])
            return_training_outputs.append(outputs[i])

        for i in range(int(len(inputs) * fraction), len(inputs)):
            return_testing_inputs.append(inputs[i])
            return_testing_outputs.append(outputs[i])
        if shuffle_data:
            return self.__shuffle_list(return_training_inputs, return_training_outputs), self.__shuffle_list(
                return_testing_inputs, return_testing_outputs)
        else:
            return return_training_inputs, return_training_outputs, return_testing_inputs, return_testing_outputs

    def save_data(self, file_name='', path=''):
        if file_name.split('.')[len(file_name.split('.')) - 1] == 'txt':
            file_path = path + file_name
        else:
            file_path = path + file_name + '.txt'
        try:
            fl = open(str(file_path), 'w')
            fl.write(str(self.__prepare_wt_data_save()))
            fl.write('$')
            fl.write(str(self.__prepare_bs_data_save()))
            fl.close()
        except FileNotFoundError:
            print('file is not found')
        except FileExistsError:
            print('file is exist')
        return 0

    def load_data(self, file_name='', path=''):
        if file_name.split('.')[len(file_name.split('.')) - 1] == 'txt':
            print('wade hari')
            file_path = path + file_name
        else:
            file_path = path + file_name + '.txt'
        try:
            fl = open(str(file_path), 'r')
            self.__prepare_network(fl.read())
            fl.close()
        except FileNotFoundError:
            print('file is not found')

        return 0

    def __prepare_network(self, load_data):
        data_set = load_data.split('$')
        wt_data = data_set[0]
        bs_data = data_set[1]
        self.__load_weights(wt_data)
        self.__load_bs(bs_data)
        return 0

    def __load_weights(self, wts):
        wts_lays = wts.split('%')
        returns = []
        for i in range(len(wts_lays)):
            wt_list = wts_lays[i].split('&')
            return1 = []
            for j in range(len(wt_list)):
                wt = wt_list[j].split(',')
                return2 = []
                for k in range(len(wt)):
                    return2.append(float(wt[k]))
                return1.append(return2)
            returns.append(return1)
        for i in range(len(returns)):
            self.__g_wt.append(Weights(len(returns[i][0]), len(returns[i])))
            self.__g_lays.append(Layer(len(returns[i])))

        for i in range(len(returns)):
            for j in range(len(returns[i])):
                for k in range(len(returns[i][j])):
                    self.__g_wt[i].weights[j][k] = returns[i][j][k]

    def __load_bs(self, bs):
        bs_lays = bs.split('@')
        returns = []
        for i in range(len(bs_lays)):
            return1 = []
            bsl = bs_lays[i].split(',')
            for j in range(len(bsl)):
                return1.append(float(bsl[j]))
            returns.append(return1)
        for i in range(len(returns)):
            self.__g_bs.append(Bias(layer_neuron_count(self.__g_lays[i])))

        for i in range(len(returns)):
            for j in range(len(returns[i])):
                self.__g_bs[i].bias_list[j] = returns[i][j]

    def __prepare_wt_data_save(self):
        wts = self.__g_wt
        returns = ""
        for i in range(len(wts)):
            return1 = ""
            for j in range(len(wts[i].weights)):
                return2 = ""
                for k in range(len(wts[i].weights[j])):
                    return2 = return2 + str(wts[i].weights[j][k])
                    if k < len(wts[i].weights[j]) - 1:
                        return2 = return2 + ','
                return1 = return1 + return2
                if j < len(wts[i].weights[j]) - 2:
                    return1 = return1 + '&'
            returns = returns + return1
            if i < len(wts) - 1:
                returns = returns + '%'
        return returns

    def __prepare_bs_data_save(self):
        bs = self.__g_bs
        returns = ""
        for i in range(len(bs)):
            return1 = ""
            for j in range(len(bs[i].bias_list)):
                return1 = return1 + str(bs[i].bias_list[j])
                if j < len(bs[i].bias_list) - 1:
                    return1 = return1 + ','
            returns = returns + return1
            if i < len(bs) - 1:
                returns = returns + '@'
        return returns

    def __back_prop(self, inputs, cost_der_list, rate):
        lays = self.__g_lays
        wts = self.__g_wt
        dw = self.__make_weight_copy()
        for i in range(len(wts)):
            for j in range(len(wts[i].weights)):
                for k in range(len(wts[i].weights[j])):
                    if i == 0:
                        dw[i][j][k] = self.__make_dw(wts, lays, cost_der_list, i, j) * sigmoid_derivation(
                            mt.mat_to_num(lays[i].neurons[j].weighted_sum)) * inputs[k]
                    else:
                        dw[i][j][k] = self.__make_dw(wts, lays, cost_der_list, i, j) * sigmoid_derivation(
                            mt.mat_to_num(lays[i].neurons[j].weighted_sum)) * lays[i - 1].neurons[k].outputs

        dbs = self.__make_bias_copy()
        for i in range(len(dbs)):
            for j in range(len(dbs[i])):
                dbs[i][j] = self.__make_dw(wts, lays, cost_der_list, i, j) * sigmoid_derivation(
                    mt.mat_to_num(lays[i].neurons[j].weighted_sum)) * 1

        wts = self.__update_weights(dw, rate)
        bias_list = self.__update_bias(dbs, rate)

        return wts, bias_list

    def __update_weights(self, dwt, rate):
        wts = self.__g_wt
        for i in range(len(dwt)):
            for j in range(len(dwt[i])):
                for k in range(len(dwt[i][j])):
                    wts[i].weights[j][k] = wts[i].weights[j][k] - dwt[i][j][k] * rate
        return wts

    def __update_bias(self, dbs, rate):
        bs = self.__g_bs
        for i in range(len(dbs)):
            for j in range(len(dbs[i])):
                bs[i].bias_list[j] = bs[i].bias_list[j] - dbs[i][j] * rate
        return bs

    def __make_weights(self):
        inputs = self.__t_inputs
        lays = self.__g_lays
        returns = [Weights(len(mt.mat_transpose(inputs)), layer_neuron_count(lays[0]))]
        for i in range(1, len(lays)):
            returns.append(Weights(layer_neuron_count(lays[i - 1]), layer_neuron_count(lays[i])))
        return returns

    def __make_bias(self):
        lays = self.__g_lays
        returns = []
        for i in range(len(lays)):
            returns.append(Bias(layer_neuron_count(lays[i])))
        return returns

    def __make_dw(self, wts, lays, cost_der, i=0, j=0):
        pre_neu_pos = [i, j]
        if i < len(wts) - 1:
            internal_value = self.__loop_generate(i + 1, lays, wts, pre_neu_pos, cost_der)
        else:
            internal_value = cost_der[j]
        return internal_value

    def __loop_generate(self, cur_lay_pos, neu_lays, wt_lays, pre_neu_pos, cost_der):
        returns = 0
        if cur_lay_pos == len(neu_lays) - 1:
            for a in range(len(neu_lays[cur_lay_pos].neurons)):
                returns += sigmoid_derivation(neu_lays[cur_lay_pos].neurons[a].weighted_sum
                                              ) * wt_lays[cur_lay_pos].weights[a][pre_neu_pos[1]] * cost_der[a]
            return returns
        else:
            for a in range(len(neu_lays[cur_lay_pos].neurons)):
                pre_n_pos = [cur_lay_pos, a]
                val = self.__loop_generate(cur_lay_pos + 1, neu_lays, wt_lays, pre_n_pos, cost_der)
                returns += sigmoid_derivation(neu_lays[cur_lay_pos].neurons[a].weighted_sum
                                              ) * wt_lays[cur_lay_pos].weights[a][pre_neu_pos[1]] * val
            return returns

    def __make_weight_copy(self):
        wts = self.__g_wt
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

    def __make_bias_copy(self):
        bs = self.__g_bs
        returns = []
        for i in range(len(bs)):
            ret1 = []
            for j in range(len(bs[i].bias_list)):
                ret1.append(bs[i].bias_list[j])
            returns.append(ret1)
        return returns

    def __print_wts(self):
        wts = self.__g_wt
        for i in range(len(wts)):
            for j in range(len(wts[i].weights)):
                for k in range(len(wts[i].weights[j])):
                    print(wts[i].weights[j][k])

    def __print_bs(self):
        bs = self.__g_bs
        for i in range(len(bs)):
            for j in range(len(bs[i].bias_list)):
                print(bs[i].bias_list[j])

    def __shuffle_list(self, input_list, output_list):
        for i in range(len(input_list)):
            rand_pos = rn.randrange(0, len(input_list), 1)
            val1 = input_list[i]
            val2 = output_list[i]
            input_list[i] = input_list[rand_pos]
            output_list[i] = output_list[rand_pos]
            input_list[rand_pos] = val1
            output_list[rand_pos] = val2
        return input_list, output_list
