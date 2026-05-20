from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim)) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros((1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size = self.input.shape[0]
        self.grads['W'] = self.input.T @ grad / batch_size
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / batch_size
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride
        self.padding = padding
        # Small initialization is important for stable training.
        scale = np.sqrt(2.0 / (in_channels * self.kernel_size * self.kernel_size))
        self.W = initialize_method(size=(out_channels, in_channels, self.kernel_size, self.kernel_size)) * scale
        self.b = np.zeros((1, out_channels, 1, 1))
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.input_padded = None
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        output: [batch, out_channels, new_H, new_W]
        """
        self.input = X
        if self.padding > 0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            X_pad = X
        self.input_padded = X_pad

        batch, channels, H, W = X_pad.shape
        k, s = self.kernel_size, self.stride
        out_H = (H - k) // s + 1
        out_W = (W - k) // s + 1
        out = np.zeros((batch, self.out_channels, out_H, out_W))

        for i in range(out_H):
            for j in range(out_W):
                h0, w0 = i * s, j * s
                window = X_pad[:, :, h0:h0+k, w0:w0+k]
                # window: [B,C,k,k], W: [O,C,k,k] -> [B,O]
                out[:, :, i, j] = np.tensordot(window, self.W, axes=([1, 2, 3], [1, 2, 3]))
        out += self.b
        return out

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        X_pad = self.input_padded
        batch, channels, H, W = X_pad.shape
        k, s = self.kernel_size, self.stride
        _, _, out_H, out_W = grads.shape

        dX_pad = np.zeros_like(X_pad)
        dW = np.zeros_like(self.W)
        db = np.sum(grads, axis=(0, 2, 3), keepdims=True) / batch

        for i in range(out_H):
            for j in range(out_W):
                h0, w0 = i * s, j * s
                window = X_pad[:, :, h0:h0+k, w0:w0+k]
                # dW[o] += sum_b grad[b,o,i,j] * window[b]
                dW += np.tensordot(grads[:, :, i, j], window, axes=([0], [0])) / batch
                # dX window += sum_o grad[b,o,i,j] * W[o]
                dX_pad[:, :, h0:h0+k, w0:w0+k] += np.tensordot(grads[:, :, i, j], self.W, axes=([1], [0]))

        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W
        self.grads['W'] = dW
        self.grads['b'] = db

        if self.padding > 0:
            return dX_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dX_pad
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.predicts = None
        self.labels = None
        self.probs = None
        self.grads = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts
        self.labels = labels.astype(int)
        batch_size = predicts.shape[0]

        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts

        eps = 1e-12
        loss = -np.mean(np.log(self.probs[np.arange(batch_size), self.labels] + eps))
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.predicts.shape[0]
        self.grads = self.probs.copy()
        self.grads[np.arange(batch_size), self.labels] -= 1
        # Do not divide by batch_size here, because Linear/conv layers already average parameter gradients.
        # This keeps gradient magnitudes compatible with the starter optimizer settings.
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition