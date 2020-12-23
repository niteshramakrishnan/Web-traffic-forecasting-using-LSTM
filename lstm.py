import random

import numpy as np
import math

def sigmoid(x): 
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(x): 
    return x*(1-x)

def tanh_derivative(x): 
    return 1. - x ** 2

def create_random_array(x, y, *args): 
    np.random.seed(0)
    return np.random.rand(*args) * (y - x) + x

class LstmParam:
    def __init__(self, num_memory_cells, x_vector_size):
        self.num_memory_cells = num_memory_cells
        self.x_vector_size = x_vector_size
        c_len = x_vector_size + num_memory_cells

        self.weight_gate = create_random_array(-0.1, 0.1, num_memory_cells, c_len)
        self.weight_input = create_random_array(-0.1, 0.1, num_memory_cells, c_len) 
        self.weight_forget = create_random_array(-0.1, 0.1, num_memory_cells, c_len)
        self.weight_output = create_random_array(-0.1, 0.1, num_memory_cells, c_len)

        self.bias_inputas_gate = create_random_array(-0.1, 0.1, num_memory_cells) 
        self.bias_input = create_random_array(-0.1, 0.1, num_memory_cells) 
        self.bias_forget = create_random_array(-0.1, 0.1, num_memory_cells) 
        self.bias_output = create_random_array(-0.1, 0.1, num_memory_cells) 

        self.weight_gate_diff = np.zeros((num_memory_cells, c_len)) 
        self.weight_input_diff = np.zeros((num_memory_cells, c_len)) 
        self.weight_forget_diff = np.zeros((num_memory_cells, c_len)) 
        self.weight_output_diff = np.zeros((num_memory_cells, c_len)) 
        self.bias_inputas_gate_diff = np.zeros(num_memory_cells) 
        self.bias_input_diff = np.zeros(num_memory_cells) 
        self.bias_forget_diff = np.zeros(num_memory_cells) 
        self.bias_output_diff = np.zeros(num_memory_cells) 

    def apply_diff(self, l = 1):
        self.weight_gate -= l * self.weight_gate_diff
        self.weight_input -= l * self.weight_input_diff
        self.weight_forget -= l * self.weight_forget_diff
        self.weight_output -= l * self.weight_output_diff
        self.bias_inputas_gate -= l * self.bias_inputas_gate_diff
        self.bias_input -= l * self.bias_input_diff
        self.bias_forget -= l * self.bias_forget_diff
        self.bias_output -= l * self.bias_output_diff
        
        self.weight_gate_diff = np.zeros_like(self.weight_gate)
        self.weight_input_diff = np.zeros_like(self.weight_input) 
        self.weight_forget_diff = np.zeros_like(self.weight_forget) 
        self.weight_output_diff = np.zeros_like(self.weight_output) 
        self.bias_inputas_gate_diff = np.zeros_like(self.bias_inputas_gate)
        self.bias_input_diff = np.zeros_like(self.bias_input) 
        self.bias_forget_diff = np.zeros_like(self.bias_forget) 
        self.bias_output_diff = np.zeros_like(self.bias_output) 

class LstmState:
    def __init__(self, num_memory_cells, x_vector_size):
        self.gate = np.zeros(num_memory_cells)
        self.input = np.zeros(num_memory_cells)
        self.forget = np.zeros(num_memory_cells)
        self.output = np.zeros(num_memory_cells)
        self.sstate = np.zeros(num_memory_cells)
        self.hstate = np.zeros(num_memory_cells)
        self.bottom_output_diff_hstate = np.zeros_like(self.hstate)
        self.bottom_output_diff_sstate = np.zeros_like(self.sstate)
    
class LstmNode:
    def __init__(self, lstm_param, lstm_state):

        self.lstm_state = lstm_state
        self.lstm_param = lstm_param

        self.x_c = None

    def bias_outputttom_data_is(self, x, sstate_prev = None, hstate_prev = None):
        # if this is the first lstm node in the netweight_outputrk
        if sstate_prev is None: sstate_prev = np.zeros_like(self.lstm_state.sstate)
        if hstate_prev is None: hstate_prev = np.zeros_like(self.lstm_state.hstate)
        # save data for use in backprop
        self.sstate_prev = sstate_prev
        self.hstate_prev = hstate_prev

        # concatenate x(t) and h(t-1)
        x_c = np.hstack((x,  hstate_prev))
        self.lstm_state.gate = np.tanh(np.dot(self.lstm_param.weight_gate, x_c) + self.lstm_param.bias_inputas_gate)
        self.lstm_state.input = sigmoid(np.dot(self.lstm_param.weight_input, x_c) + self.lstm_param.bias_input)
        self.lstm_state.forget = sigmoid(np.dot(self.lstm_param.weight_forget, x_c) + self.lstm_param.bias_forget)
        self.lstm_state.output = sigmoid(np.dot(self.lstm_param.weight_output, x_c) + self.lstm_param.bias_output)
        self.lstm_state.sstate = self.lstm_state.gate * self.lstm_state.input + sstate_prev * self.lstm_state.forget
        self.lstm_state.hstate = self.lstm_state.sstate * self.lstm_state.output

        self.x_c = x_c
    
    def top_diff_is(self, top_diff_h, top_diff_s):
        d_state = self.lstm_state.output * top_diff_h + top_diff_s
        d_output = self.lstm_state.sstate * top_diff_h
        d_input = self.lstm_state.gate * d_state
        d_gate = self.lstm_state.input * d_state
        d_forget = self.sstate_prev * d_state

        d_input_input = sigmoid_derivative(self.lstm_state.input) * d_input 
        d_forget_input = sigmoid_derivative(self.lstm_state.forget) * d_forget 
        d_output_input = sigmoid_derivative(self.lstm_state.output) * d_output 
        d_gate_input = tanh_derivative(self.lstm_state.gate) * d_gate

        self.lstm_param.weight_input_diff += np.outer(d_input_input, self.x_c)
        self.lstm_param.weight_forget_diff += np.outer(d_forget_input, self.x_c)
        self.lstm_param.weight_output_diff += np.outer(d_output_input, self.x_c)
        self.lstm_param.weight_gate_diff += np.outer(d_gate_input, self.x_c)
        self.lstm_param.bias_input_diff += d_input_input
        self.lstm_param.bias_forget_diff += d_forget_input       
        self.lstm_param.bias_output_diff += d_output_input
        self.lstm_param.bias_inputas_gate_diff += d_gate_input       

        dx_c = np.zeros_like(self.x_c)
        dx_c += np.dot(self.lstm_param.weight_input.T, d_input_input)
        dx_c += np.dot(self.lstm_param.weight_forget.T, d_forget_input)
        dx_c += np.dot(self.lstm_param.weight_output.T, d_output_input)
        dx_c += np.dot(self.lstm_param.weight_gate.T, d_gate_input)

        self.lstm_state.bottom_output_diff_sstate = d_state * self.lstm_state.forget
        self.lstm_state.bottom_output_diff_hstate = dx_c[self.lstm_param.x_vector_size:]

class LstmNetwork():
    def __init__(self, lstm_param):
        self.lstm_param = lstm_param
        self.lstm_node_list = []

        self.x_list = []

    def y_list_is(self, y_list, loss_layer):
        assert len(y_list) == len(self.x_list)
        index = len(self.x_list) - 1

        loss = loss_layer.loss(self.lstm_node_list[index].lstm_state.hstate, y_list[index])
        diff_h = loss_layer.bottom_diff(self.lstm_node_list[index].lstm_state.hstate, y_list[index])

        diff_s = np.zeros(self.lstm_param.num_memory_cells)
        self.lstm_node_list[index].top_diff_is(diff_h, diff_s)
        index -= 1

        while index >= 0:
            loss += loss_layer.loss(self.lstm_node_list[index].lstm_state.hstate, y_list[index])
            diff_h = loss_layer.bottom_diff(self.lstm_node_list[index].lstm_state.hstate, y_list[index])
            diff_h += self.lstm_node_list[index + 1].lstm_state.bottom_output_diff_hstate
            diff_s = self.lstm_node_list[index + 1].lstm_state.bottom_output_diff_sstate
            self.lstm_node_list[index].top_diff_is(diff_h, diff_s)
            index -= 1 

        return loss

    def x_list_clear(self):
        self.x_list = []

    def x_list_add(self, x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_node_list):

            lstm_state = LstmState(self.lstm_param.num_memory_cells, self.lstm_param.x_vector_size)
            self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))


        index = len(self.x_list) - 1
        if index == 0:

            self.lstm_node_list[index].bias_outputttom_data_is(x)
        else:
            sstate_prev = self.lstm_node_list[index - 1].lstm_state.sstate
            hstate_prev = self.lstm_node_list[index - 1].lstm_state.hstate
            self.lstm_node_list[index].bias_outputttom_data_is(x, sstate_prev, hstate_prev)

