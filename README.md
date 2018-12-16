# netty
a VERY simplistic set of objects for creating neural networks

Disclaimer:
Only tested with a single node, although could work for more complexe networks.
Not in anyway suitible for implementation anywhere. 
Written as a learning excersize, so deffinatly flawed.

Structure:
netty.py --
          |--> e , the mathmaticle constant
          |
          |--> sigmoid , a function such that y = sigmoid(x) plots the sigmoid curve
          |
          |--> neurone , an object that forms one node in a network
          |       |--> __init__ , creats a neurone object
          |       |--> ajust , ajusts the weighting for each input of the neurone based on some error value
          |       '--> think , calculates the neurone's output from a given input
          |
          '--> net , an object that is used to manage a network
                |--> __init__ , creats a network of a given topology (with restrictions)
                |--> train , trains a network with iteration
                |--> run , calculates the networks output from a given input
                '--> disp , prints out the networks weigth values to the console
