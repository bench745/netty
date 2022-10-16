# https://towardsdatascience.com/how-to-build-a-deep-neural-network-without-a-framework-5d46067754d5
# https://neptune.ai/blog/backpropagation-algorithm-in-neural-networks-guide
# https://github.com/ahmedfgad/IntroDLPython
# https://mathspp.com/blog/neural-networks-fundamentals-with-python-mnist

# TODO add new loss functions and their derivatives
# TODO add a generic train function
# TODO add a gerneric test function
# TODO add some more activation functions
# TODO add model pickling
# TODO add other training methods
# TODO make faster - try threading? perhaps
# TODO Try adding convolutional layers?

from abc import abstractmethod
from typing import Callable
from pprint import pprint

import numpy as np
from numpy.typing import NDArray
from numpy import float64



## Activation functions

class ActivationFunction:
    @staticmethod
    @abstractmethod
    def f(x):
        pass

    @staticmethod
    @abstractmethod
    def df_dx(x):
        pass

    
class SigmoidAF(ActivationFunction):
    @staticmethod
    def f(x):
        '''The sigmoid activation function.'''
        return 1/(1 + np.exp(-x))

    @staticmethod
    def df_dx(x):
        sigx = SigmoidAF.f(x)
        return sigx * (1 - sigx)

    
class ReluAF(ActivationFunction):
    @staticmethod
    def f(x: NDArray[float64]) -> NDArray[float64]:
        '''The ReLU activation function.'''
        return np.greater_equal(x, 0) * x

    @staticmethod
    def df_dx(x: NDArray[float64]) -> NDArray[float64]:
        return np.greater_equal(x, 0) * 1.0


class LeakyReluAF(ActivationFunction):
    LEAK: float = 0.1
    
    @staticmethod
    def f(x: NDArray[float64]) -> NDArray[float64]:
        return np.maximum(LeakyReluAF.LEAK * x, x)

    @staticmethod
    def df_dx(x: NDArray[float64]) -> NDArray[float64]:
        return np.maximum(np.greater_equal(x, 0)*1.0, LeakyReluAF.LEAK)

    
class IdentityAF(ActivationFunction):
    @staticmethod
    def f(x: NDArray[float64]) -> NDArray[float64]:
        '''The identity acitvation function.'''
        return x

    @staticmethod
    def df_dx(_: NDArray[float64]) -> NDArray[float64]:
        return np.array([[1.0]])

## Loss function


class LossFunction:
    @staticmethod
    @abstractmethod
    def f(hyp: NDArray[float64], ans: NDArray[float64]) -> float:
        return 0.0

    @staticmethod
    @abstractmethod
    def df(hyp: NDArray[float64], ans: NDArray[float64]) -> NDArray[float64]:
        return np.array([])

class MeanSquaresLF(LossFunction):
    @staticmethod
    def f(hyp: NDArray[float64], ans: NDArray[float64])  -> float:
        return float(np.mean(np.square(np.subtract(hyp, ans))))

    @staticmethod
    def df(hyp: NDArray[float64], ans: NDArray[float64]) -> NDArray[float64]:
        return 2/hyp.size * np.subtract(hyp, ans)
    

## The actual network classes

class Layer:

    w: NDArray[float64]  # weights
    b: NDArray[float64]  # biases
    
    af: Callable
    daf_dx: Callable
    
    node_cnt: int 
    input_size: int

    lr: float  # learning rate

    cache_used: bool
    last_input: NDArray[float64]
    last_wsum: NDArray[float64]
    last_output: NDArray[float64]
    
    
    def __init__(self, input_size: int, node_cnt: int,
                 activation_funct: ActivationFunction, learning_rate: float):
        self.w = np.random.randn(node_cnt, input_size) / (input_size * node_cnt)
        #self.w = np.zeros((node_cnt, input_size))
        self.b = np.random.randn(node_cnt, 1) / (node_cnt)
        self.af = activation_funct.f
        self.daf_dx = activation_funct.df_dx
        self.node_cnt = node_cnt
        self.input_size = input_size
        self.lr = learning_rate
        self.cache_used = False

    def forward(self, x: NDArray[float64]) -> NDArray[float64]:
        '''
        Feed an input through a network
        '''

        assert(x.shape == (self.input_size, 1))
        assert(not np.isnan(x).any())
        
        wsum = np.dot(self.w, x) + self.b
        
        output = self.af(wsum)

        self.last_input = x
        self.last_wsum = wsum
        self.last_output = output

        self.cache_used = True
        
        return output

    def back_propergate(self, dcost_dout: NDArray[float64]) -> NDArray[float64]:
        '''Carry out back propergation on this layer, based on the effect this
        layer had on the last output.

        Parametres:
        - dout_dcost (NDArray[float64]) derivative of cost with respect to the
          last outputut.

        returns:
        - din_dcost (NDArray[float64]) the effect the previous layer had on the
          error.
        '''
        if not self.cache_used:
            raise RuntimeError('Cannot back properagte without forward pass.')

        ### Compute derivatives
        ## See: https://blog.yani.ai/backpropagation/
        
        dout_dwsum = self.daf_dx(self.last_wsum)  # the effect the wsum has on the output
        dwsum_dw = self.last_input.T
        #assert(dwsum_dw.shape == self.w.shape)
        
        # dwsum_db = 1  # effect on the wsum that the bias has
            
        # we use dcost_dw = dcost_dout * dout_dwsum * dwsum_dw 
        # we use dcost_db = dcost_dout * dout_dwsum * dwsum_db = dcost_dout * dout_dwsum

        dcost_db = dcost_dout * dout_dwsum # * dwsum_db
        dcost_dw = np.dot(dcost_db, dwsum_dw)

        # By summing (the effects of a weights applied to a neurons output * the
        # bias that sees), we see the effect the neuron outputs from the layer
        # that preceedes this one have on cost.
        #col_sum_dcost_dw = np.sum(dcost_dw, 0).reshape((self.input_size, 1))
        this_dcost_dout = np.dot(self.w.T, dcost_db)
        
        ### Make changes
        self.w = self.w - self.lr * dcost_dw  
        self.b = self.b - self.lr * dcost_db

        assert(not np.isnan(self.w).any())
        assert(not np.isnan(self.b).any())
        
        return this_dcost_dout

        
class Network:
    '''
    A neural network composed from layers, of various kinds.
    '''

    layers: list[Layer]
    input_size: int
    lf: LossFunction

    training_age: int

    
    def __init__(self, input_size: int, layer_sizes: list[int],
                 activation_functions: list[ActivationFunction],
                 loss_function: LossFunction, learning_rate: float):
        '''
        Produce a neural network.

        Parametres:
        - input_size (int) The number of items an input vector will be composed
          of.
        - layer_sizes (list[int]) The number of nodes in each layer.
        - activation_functions (list[ActivationFunctions]) The activation
          functions to be used by each layer.
        - learning_rate (float) how fast should the network try and learn.
        '''
        self.input_size = input_size
        self.lf = loss_function
        self.training_age = 0
        
        layers = list()
        for size, af in zip(layer_sizes, activation_functions):
            layers.append(Layer(input_size, size, af, learning_rate))
            input_size = size
        self.layers = layers

        #for layer in layers:
        #    print(layer.w.shape, layer.b.shape)
            
    def infer(self, nw_input: list[float]) -> list[float]:
        '''
        Make an inference about the input
        '''
        assert(len(nw_input) == self.input_size)

        #print()
        x: NDArray[float64] = np.array(nw_input, float64).reshape((self.input_size, 1))
        for l, layer in enumerate(self.layers):
            #print(f'x shape: {x.shape}, w shape {layer.w.shape}, b shape {layer.b.shape}')
            try:
                x = layer.forward(x)
            except AssertionError as e:
                pos_suffix = lambda n: ["th", "st", "nd", "rd"][n if n <= 3 else 0]
                print(f'Error in \
{l+1}{pos_suffix(l+1)} layer, on {self.training_age}{self.training_age+1} \
training epoch.')
                raise e
                
        #print(f'x shape: {x.shape}')
        return x.flatten().tolist()

    def learn(self, actual_answer: list[float]) -> float:
        '''
        Teach the network with the correct answer.

        Give the network the real answer to the last inference, it can then
        learn from it's mistakes.

        Parametres:
         - actual_answer (list[float]) the correct answer to the last
           query given to the network.

        Returns:
        - cost (float) a measure of the error in the last output.

        '''
        assert(len(actual_answer) == self.layers[-1].node_cnt)
        if not self.layers[-1].cache_used:
            raise RuntimeError('Make sure you\'ve completed a forward pass \
            before trying back propergation.')
        
        # prep data
        ans: NDArray[float64] = np.array(actual_answer).reshape((len(actual_answer), 1))
        out: NDArray[float64] = self.layers[-1].last_output

        # compute cost
        sqdiff = np.square(np.subtract(ans, out))
        np.nan_to_num(sqdiff, copy=False, nan=2**16-1)
        cost = self.lf.f(out, ans)
        
        # compute cost derivative with respect to the output, at ans
        dcost_dout = self.lf.df(out, ans)
        
        # begin back propagation
        for layer in reversed(self.layers):
            dcost_dout = layer.back_propergate(dcost_dout)  

        self.training_age += 1
            
        return cost



def rolling_average(xs: list[float], window: int = 10) -> list[float]:
    avgs = list()
    for p in range(window, len(xs), 1):
        sub = xs[p-window: p]
        avg = sum(sub)/window
        avgs.append(avg)
    return avgs
        

if __name__ == '__main__':
    from idx import IDXReader

    #target_cls = 1  # we are going to try and spot 0s
    TRAINING_IMAGE_CNT = 60000
    TEST_IMAGE_CNT = 10000
    TRAINING_RATE = 0.003

    DATA_SCALE = 1/256
    
    print('reading images from disk ... ', end='', flush=True)
    
    train_img_reader = IDXReader('/home/bench/Projects/MNIST/mnist/train-images.idx3-ubyte')
    train_cls_reader = IDXReader('/home/bench/Projects/MNIST/mnist/train-labels.idx1-ubyte')
    
    test_img_reader = IDXReader('/home/bench/Projects/MNIST/mnist/t10k-images.idx3-ubyte')
    test_cls_reader = IDXReader('/home/bench/Projects/MNIST/mnist/t10k-labels.idx1-ubyte')

    # fetch the image data
    
    training_imgs = list()
    training_ans = list()
    for i in range(TRAINING_IMAGE_CNT):
        img = train_img_reader.get_vector(i)
        training_imgs.append(
            np.array(img, dtype=np.float64).reshape((28*28, 1)) * DATA_SCALE
        )
        cls = train_cls_reader.get_vector(i)[0]
        training_ans.append(cls)

    test_imgs = list()
    test_ans = list()
    for i in range(TEST_IMAGE_CNT):
        img = test_img_reader.get_vector(i)
        test_imgs.append(
            np.array(img, dtype=np.float64).reshape((28*28, 1)) * DATA_SCALE
        )
        cls = test_cls_reader.get_vector(i)[0]
        test_ans.append(cls)
        
    print('Done.')

    print('Setting up NN ... ', end='')

    nn = Network(28*28,
                 [128, 10],
                 [LeakyReluAF()]*2,
                 MeanSquaresLF(),
                 TRAINING_RATE)
    
    print('Done.')

    print('Training ... ', end='', flush=True)

    from math import isnan
    
    costs = list()
    for img, ans in zip(training_imgs, training_ans):
        ans_fmted = [float(ans==i) for i in range(10)]  # [1. if ans == CLASSIFICATION_TARGET else 0.]
        result = nn.infer(img)
        cost = nn.learn(ans_fmted)
        if isnan(cost):
            raise RuntimeWarning('cost is nan.')
        costs.append(cost)
    
    print('Done.')

    import matplotlib.pyplot as plt

    avgs = rolling_average(costs, TRAINING_IMAGE_CNT//1000)
    epochs = list(range(len(avgs)))

    plt.plot(epochs, avgs)
    plt.grid()
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.show()

        
    print('Testing ... ', end='', flush=True)

    confusion = [[0 for _ in range(10)] for _ in range(10)]
    for img, ans in zip(test_imgs, test_ans):
        result = nn.infer(img)

        cls = 0
        for p in range(1, 10, 1):
            if result[p] > result[cls]:
                cls = p

        confusion[ans][cls] += 1
                
    print('Done.')

    ## interpret and show results
    
    total_correct = 0
    for i in range(10):
        total_correct += confusion[i][i]
    preportion_correct = total_correct/TEST_IMAGE_CNT
    
    print(f'{preportion_correct * 100}% correct in total.')
    pprint(confusion)
