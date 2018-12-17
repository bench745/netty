import netty

# the training inputs
ins = [[0,0,0],
       [0,0,1],
       [0,1,0],
       [0,1,1],
       [1,0,0],
       [1,1,0],
       [1,1,1]]

# the training outputs [prob of one, prob of zero]
outs = [[0],
        [1],
        [0],
        [1],
        [0],
        [0],
        [1]]

#the pattern it should find is that the output is equal to the third input

# creates a neural net with 3 inputs and one node
n = netty.net(3, [1])

# display the raw nets weight values
n.disp()

# train the net using the training data above over 100 iterations
n.train(ins, outs, 10000)

# display the new weight values
n.disp()

# run the network with a new peice of data
output = n.run([1,0,1])
print('[1,0,1] -> ' + str(output))
