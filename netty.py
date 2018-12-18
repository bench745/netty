#back propergation to be implemented within the neurone ajust subclass
#resources:
#https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
#https://medium.com/datadriveninvestor/math-neural-network-from-scratch-in-python-d6da9f29ce65


import random
import time

# the constant e to 46 dp from wikipedia 
e = 2.7182818284590452353602874713526624977572470937 


# returns the y value for a given x co-ordinate on the sigmoid curve
# used to nomalise the output of a neurone between 1 and 0
sigmoid = lambda x : 1/(1 + pow(e,-1*x))


# calculates the gradient of a the sigmoid curve at x (is dy/dx)
# used in the weight ajustment calculations 
sigmoidGradientFunct = lambda x : x*(1-x)


        

# A neurone object. takes inputs can depending on their weighting
# passes an output
class neurone(object):
    
    # a map of the weights for each input to the neurone index 0
    # corresponds with input 0
    inputWeights = []

    # a store of the data from the last think
    inputed = []
    outputed = 0


    def __init__(self, inputs):
        '''creates a neurone object'''

        random.seed(time.time() + random.randint(0,255))  # seeds a random number generater

        # generate a set of starting weights fro the neurones input
        l = []
        for i in range(inputs):
            l.append(random.uniform(-1, 1))
        self.inputWeights = l


    # error is the error of the overall network
    def ajust(self, error):
        '''ajusts the internal input weights of the node based on the error'''
        
        dy_by_d_out = sigmoidGradientFunct(self.outputed)  # work out the gradient of the curve at the output pos
        
        for i in range(len(self.inputWeights)):  # ajust each weight
            ajustmentFactor = error * self.inputed[i] * dy_by_d_out  # calc adj fact, the overall error * the inputed data (so 0 have no effect) * sigout (so when the network is sure we dont ajust hugly)
            self.inputWeights[i] += ajustmentFactor  # ajust the weighting


    def think(self, data):
        '''returns an output based on the input'''
        
        self.inputed = data  # save the inputed data to make ajustments later
        
        tot = 0  # the total of the inputs*the weights
        for i in range(len(self.inputWeights)):
            tot += self.inputWeights[i] * data[i]

        out = sigmoid(tot)  # calculate the sigmoid mapping
        self.outputed = out  # record the output of the neurone
            
        return out  # return the output





class net(object):

    net = []  # a 2D array that contains neurone objects

    #populates the net object
    def __init__(self, ins, layout):
        '''create a neural network. ins is the number of inputs.
        layout is an array that should contain a number of nodes
        for each layer. Each node will take the outputs of each
        node in the previuos layer as an input. The last layer
        must contain only one node'''

        for i in range(len(layout)):  # for each layer
            layer = []  # create a new layer
            for j in range(layout[i]):  # for each of the neurones in the layout
                if i == 0:  # if this is the first layer
                    layer.append(neurone(ins))  # create neurones with inputs corresponding to the number of inputs to the network
                    
                else:
                    layer.append(neurone(layout[i-1]))  # create neurones that can accept data from all the neurone in the previous layer
                    
            self.net.append(layer)

    
    # hones the weight values over a number of iterations. Each time the
    # network tries to guess an answer to the problem and this is compared with
    # the known answer. The weight of each input is then tailored
    def train(self, trainingInputs, trainingOutputs, iters = 1000):
        '''train(self, trainingInputs, trainingOutputs, iters = 0)
        trains the net over a number of iterations. The trainingInputs
        should be a 2D array of inputs. trainingOutputs should be an
        array of outputs corresponding to these.'''
        
        if not (len(trainingInputs) == len(trainingOutputs)):
            raise AttributeError('len(trainingInputs) must equal len(trainingOutputs)')

        for i in range(iters):  # for the number of iterations
            for j in range(len(trainingInputs)):  # for each peice of training data
                out = self.run(trainingInputs[j])  # run the network and get the output

                error = []  # stored the error of the output in its parts
                for k in range(len(trainingOutputs[j])):
                    error.append(trainingOutputs[j][k] - out[k])  # compare the output parts with correct answers parts

                for k in range(len(self.net)):  # run ajust on all the nodes
                    for l in range(len(self.net[k])):
                        self.net[k][l].ajust(sum( [0.5 * pow(error[m],2) for m in range(len(error))] ) / len(error))  # run ajust on the on the mean of the error at the output squared 


    # runs the network with given input data
    def run(self, data):
        '''run the network with the list data as the input'''

        ins = data

        # run each layer intern
        for i in range(len(self.net)):  # for every layer in the network
            out = []  # create an area to store the output of the nodes
            for j in range(len(self.net[i])):  # for each node in the layer
                out.append(self.net[i][j].think(ins))  # have it think with the input data ins, and append to out
            ins = out  # set up to feed the output into the next layer

        return out  # return the final output


    # displays the network
    def disp(self):
        '''diplay the weight values of the table to the console'''

        count = 1
        for layer in self.net:
            row = 'layer: ' + str(count) + '    '  # prepend the row with the layer number
            for node in layer:              
                row += str(node.inputWeights) + ' '  # add each nodes weight to the row
            print(row)
            count += 1

                                
 
        
