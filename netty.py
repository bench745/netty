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


# calculates the product of a list
# will return 1 in the event of an empty list
def product(a):
    prod = 1 if len(a) > 0 else 0
    for item in a:
        prod = prod*item
    return prod




# A neurone object. takes inputs can depending on their weighting
# passes an output
class neurone(object):
    
    # a map of the weights for each input to the neurone index 0
    # corresponds with input 0
    inputWeights = []

    # a store of the data from the last think
    inputted = []
    answered = 0
    outputted = 0


    def __init__(self, inputs):
        '''creates a neurone object'''

        random.seed(time.time() + random.randint(0,255))  # seeds a random number generater

        # generate a set of starting weights fro the neurones input
        l = []
        for i in range(inputs):
            l.append(random.uniform(-1, 1))
        self.inputWeights = l



    # error is the error of the overall network
    def ajust(self, errorFactor, lr = 1):
        '''ajusts the internal input weights of the node based on the error'''
        
        dy_by_d_out = sigmoidGradientFunct(self.outputted)  # work out the gradient of the curve at the output pos
        
        for i in range(len(self.inputWeights)):  # ajust each weight
            ajustmentFactor = errorFactor * self.inputted[i] * dy_by_d_out  # calc adj fact, the overall error * the inputted data (so 0 have no effect) * sigout (so when the network is sure we dont ajust hugly)
            self.inputWeights[i] -= lr*ajustmentFactor  # ajust the weighting



##    # a function that will help inplement back propergation,
##    # gets the effects that each input weight has unpon the output of the neurone
##    def getDeltaWnByDeltaOut(self):
##        '''gets the effects that each input weight has unpon the output of the neurone'''
##        dWnBydOut = []
##        for i in range(len(self.inputWeights)):  # for each weight in the neurone
##            dWnBydOut.append(inputted[i] * sigmoidGradientFunct(self.outputted))  # calculate the effect that the weigth has has on the output of thre neurone
##
##        return dWnBydOut
        

    def think(self, data):
        '''returns an output based on the input'''
        
        self.inputted = data  # save the inputted data to make ajustments later
        
        tot = 0  # the total of the inputs*the weights
        for i in range(len(self.inputWeights)):
            tot += self.inputWeights[i] * data[i]

        self.answered = tot  # remeber our value pre normalisation

        out = sigmoid(tot)  # calculate the sigmoid mapping
        self.outputted = out  # record the output of the neurone
            
        return out  # return the output






class net(object):

    net = []  # a 2D array that contains neurone objects
    topology = []
    noOfLayers = 0

    #populates the net object
    def __init__(self, ins, layout):
        '''create a neural network. ins is the number of inputs.
        layout is an array that should contain a number of nodes
        for each layer. Each node will take the outputs of each
        node in the previuos layer as an input. The last layer
        must contain only one node'''
        
        self.topology = layout
        self.noOfLayers = len(layout)

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
                        self.net[k][l].ajust(sum( [0.5 * pow(error[m],2) for m in range(len(error))] ))  # run ajust on the on the the total of (0.5 times the errors at the output squared) 



    # hones the weight values over a number of iterations. Each time the
    # network tries to guess an answer to the problem and this is compared with
    # the known answer. The weight of each input is then tailored
    def trainProperly(self, trainingInputs, trainingOutputs, iters = 1000, learningRate = 1):
        '''train(self, trainingInputs, trainingOutputs, iters = 0)
        trains the net over a number of iterations. The trainingInputs
        should be a 2D array of inputs. trainingOutputs should be an
        array of outputs corresponding to these.'''
        
        if not (len(trainingInputs) == len(trainingOutputs)):
            raise AttributeError('len(trainingInputs) must equal len(trainingOutputs)')

        for i in range(iters):  # for the number of iterations
            for j in range(len(trainingInputs)):  # for each peice of training data
                out = self.run(trainingInputs[j])  # run the network and get the output

                # calculate the error at each output
                error = []  # stored the error of the output in its parts
                for k in range(len(trainingOutputs[j])):
                    error.append(trainingOutputs[j][k] - out[k])  # compare the output parts with correct answers parts

                # calculate total error
                totalE = sum([0.5 * pow(error[m],2) for m in range(len(error))])  #the sum of 1/2(error at out node)^2
                ##print('\noutput:', out, '   total error:', totalE)

                # work out delta total error by delta out for each node
                deltaEtotalByDeltaOutNet = []  # stores the effect that the output of a node has on each output a 2d array
                for k in range(len(self.net)-1):  # loop through layer in the network bar the output layer
                    deltaEtotalByDeltaOutLayer = []
                    for l in range(len(self.net[k])):  # loop through each node in the layer
                        deltaEmByDeltaOutNodekl = []  # store the effects that the output of this node has on each output error
                        for m in range(len(self.net[-1])):  # for each output
                            ##print('output node:', m)

                            # as the the effect that the presigmoid output of a node has on each output is comprised from
                            # the sum of the chains formed by each route to the out put we need to work out the routes from
                            # our node to the given output.
                            feild = self.topology[k+1:-1]  # create a copy of the area the paths will fall within
                            numPaths = product(feild) if product(feild) > 0 else 1 # the number of paths from node to output
                            feildSiz = len(feild) 
                            paths = []  # the ways to navigate from one side of the feild to the other
                            paths.append([l]*numPaths)
                            for n in range(feildSiz):  # for each layer
                                pattern = []
                                while len(pattern) < numPaths:  # repeat the pattern created below
                                    for o in range(feild[n] if n < feildSiz else 1):  # for the number of choices for the next step
                                        for p in range(product(feild[n+1:])):  pattern.append(o)  #append the correct step in the path
                                paths.append(pattern)  # append this set of steps to the pattern
                                
                            ##print('paths to output node:', paths)
                            # The next step is to compute delta (presigmoid output of output node m) by (delta output of node i)
                            # we will do this by iterating through the routes we just generated. Starting from the output node selected 
                            # by self.net[-1][m] all the way to the node selected by self.net[k][l].
                            # first we multiply the weight that connects the output node to the first intermediate node by the
                            # sigmoidGradFunct(first intermediate node's output) then by the weigths and sigmoidGradFunct()s of the
                            # intermediate nodes all the way to the weight that connects the node to all the intermediate nodes.
                            # we do this for all the routes appending each to a list which we will sum.
                            pathAffects = []  # the affects of each path in a list
                            for n in range(len(paths[0])):  # for the number of paths (the path index ie path 0, path 1)
                                pathImpact = 1  # the impact of the current path being negotiated
                                
                                for o in range(1,len(paths)): #iterate through the steps 
                                    # the address of the current intermediate node is self.net[k+o][paths[o][n]]
                                    # the address of the the output node we are moving to self.net[-1][m]
                                    # the address of the node we're moving from self.net[k][l]
                                    pathImpact *= self.net[k+o][paths[o][n]].inputWeights[paths[o-1][n]]  # times an intermediate weight
                                    ##print('weight:', self.net[k+o][paths[o][n]].inputWeights[paths[o-1][n]])
                                    pathImpact *= sigmoidGradientFunct(self.net[k+o][paths[o][n]].outputted)  # times an intermediate sigmoidGradientFunction(outputted)
                                    ##print('grad of out:',sigmoidGradientFunct(self.net[k+o][paths[o][n]].outputted))
                                        
                                pathImpact *= self.net[-1][m].inputWeights[paths[-1][n]]#times the weight that the output node give the last node in the route
                                ##print('weight at the ouput',self.net[-1][m].inputWeights[paths[-1][n]])
                                pathAffects.append(pathImpact)

                            totalPathAffects = sum(pathAffects)  # (presigmoid output of output node m) by (delta output of node i)

                            # the next step is to calculate delta error at output node m by output of node kl
                            deltaEmByDeltaOutNodekl.append(totalPathAffects * (out[m] - trainingOutputs[j][m]) * sigmoidGradientFunct(self.net[-1][m].outputted))

                        # now we work out the effect that the output of our node has on the total error of the system (by summing) and then append to a
                        # 2d in the footprint of the network
                        deltaEtotalByDeltaOutLayer.append(sum(deltaEmByDeltaOutNodekl))
                    deltaEtotalByDeltaOutNet.append(deltaEtotalByDeltaOutLayer)

                # now we need to compute delta Etotal By delta out for each output node
                deltaEtotalByDeltaOutLayer = []
                for k in range(len(self.net[-1])):  #iterate through the outputs
                    deltaEtotalByDeltaOutLayer.append((out[k] - trainingOutputs[j][k]) * sigmoidGradientFunct(self.net[-1][k].outputted))
                deltaEtotalByDeltaOutNet.append(deltaEtotalByDeltaOutLayer)
                    
                ##print('error factor for each node:', deltaEtotalByDeltaOutNet)
                # now that we have a delta E total by delta out mirror of the network (one dE by dO for each node)
                # now we must ajust the networks weights by iterating through and passing the correspding error to an
                # ajustment function in each node.

                for k in range(len(self.net)):
                    for l in range(len(self.net[k])):
                        self.net[k][l].ajust(deltaEtotalByDeltaOutNet[k][l], learningRate)                                   
                              
                                    

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

                                
 
        
