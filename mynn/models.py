from .op import *
import pickle
import numpy as np

class Model_MLP(Layer):
    """
    A model with linear layers. We provided you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i+2]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplementedError
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.input_shape = None
    def __call__(self, X):
        return self.forward(X)
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)
    def backward(self, grads):
        return grads.reshape(self.input_shape)

class Model_CNN(Layer):
    """
    A simple CNN for MNIST using the self-implemented conv2D operator.
    Structure: Conv(1->8, 3x3, pad=1) -> ReLU -> Flatten -> Linear -> ReLU -> Linear.
    """
    def __init__(self, num_classes=10, lambda_list=None):
        self.num_classes = num_classes
        self.layers = [
            conv2D(1, 8, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Flatten(),
            Linear(8 * 28 * 28, 128),
            ReLU(),
            Linear(128, num_classes),
        ]
        if lambda_list is not None:
            opt_layers = [layer for layer in self.layers if layer.optimizable]
            for layer, lam in zip(opt_layers, lambda_list):
                layer.weight_decay = True
                layer.weight_decay_lambda = lam

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, 28, 28)
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            saved = pickle.load(f)
        self.__init__(num_classes=saved[0]['num_classes'])
        opt_layers = [layer for layer in self.layers if layer.optimizable]
        for layer, state in zip(opt_layers, saved[1:]):
            layer.W = state['W']
            layer.b = state['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = state['weight_decay']
            layer.weight_decay_lambda = state['lambda']
        
    def save_model(self, save_path):
        param_list = [{'model': 'Model_CNN', 'num_classes': self.num_classes}]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W': layer.params['W'], 'b': layer.params['b'], 'weight_decay': layer.weight_decay, 'lambda': layer.weight_decay_lambda})
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
