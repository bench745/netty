import os
import struct
from functools import reduce


class IDXReader:
    '''
    Provides an API for using readin IDX files.
    '''

    # data types recognised by the file format
    UBYTE = 0x08
    SBYTE = 0x09
    SHORT = 0x0B
    SINT = 0x0C
    FLOAT = 0x0D
    DOUBLE = 0x0E

    
    def __init__(self, path: str):
        '''
            Prepare to read an IDX file.
        
            Parametres:
                path (str): A path to an IDX file

            Returns:
                None
        '''
        
        if not os.path.isfile(path):
            raise ValueError(f'path "{path}" is not a file.')
        self.path = path  # the path of the idx file that corresponds to this object
        
        self.magic_number = 0 # the file magic number
        
        self.item_type = 0  # the data type stored
        self.item_size = 0  # the length of the data type stored
        self.dimension_count = 0  # the number of dimesnions in each vector
        self.dimension_sizes = list()  # the size of each dimension
        self.header_size = 0  # size of the header in bytes

        # parse the header out
        with open(path, 'rb') as f:
            mn_bytes = f.read(4)
            self.item_type = mn_bytes[2]
            self.dimension_count = mn_bytes[3]
            
            self.magic_number = int.from_bytes(mn_bytes, 'big')
            #print(self.magic_number)
            #print(f'{hex(self.item_type)} in {self.dimension_count}-dimensions')
            
            for _ in range(self.dimension_count):
                b = f.read(4)
                sz = int.from_bytes(b, 'big')
                self.dimension_sizes.append(sz)

            self.header_size = f.tell()
            #print(f'{self.dimension_sizes}')

        # get the vector type
        match self.item_type:
            case self.UBYTE:
                self.item_size = 1
            case self.SBYTE:
                self.item_size = 1
            case self.SHORT:
                self.item_size = 2
            case self.SINT:
                self.item_size = 4
            case self.FLOAT:
                self.item_size = 4
            case self.DOUBLE:
                self.item_size = 8
            case _:
                raise ValueError(f'Cannot recognize vector compenent type {hex(self.item_type)}.')
            
    def _bytes_to_item(self, b):
        result = None
        
        # get the vector type
        match self.item_type:
            case self.UBYTE:
                result = int.from_bytes(b, 'big', signed=False)
            case self.SBYTE:
                result = int.from_bytes(b, 'big', signed=True)
            case self.SHORT:
                result = int.from_bytes(b, 'big', signed=True)
            case self.SINT:
                result = int.from_bytes(b, 'big', signed=True)
            case self.FLOAT:
                [result] = struct.unpack('>f', b)
            case self.DOUBLE:
                [result] = struct.unpack('>d', b)
            case _:
                raise ValueError(f'Cannot recognize vector compenent type {hex(self.item_type)}.')

        return result

    def _construct_vector(self, flat, dimesnion_sizes):
        if len(dimesnion_sizes) <= 1:
            return flat
        sublength = dimesnion_sizes[1]
        return [
            self._construct_vector(flat[i*sublength:i*sublength + sublength],
                                   dimesnion_sizes[1:])
            for i in range(dimesnion_sizes[0])
        ]

    
    def get_vector_bytes(self, index: int) -> bytes:
        '''
           Read a vector from the file. Reads in a zero indexed manner.

           Parametres:
               index (int): the vectors index in the file

           Returns:
               vbytes (bytes): the vector as a bytes object 
        '''
        vector_len = reduce(lambda x,y : x*y, self.dimension_sizes[1:], 1)
        vector_size = vector_len * self.item_size

        index = self.header_size + (index * vector_size)
        vbytes = bytes()

        # read in the vectors contents
        with open(self.path, 'rb') as f:
            f.seek(index)
            vbytes = f.read(vector_size)

        return vbytes
    
    
    def get_vector(self, index: int):
        '''
           Read a vector from the file. Reads in a zero indexed manner.

           Parametres:
               index (int): the vectors index in the file

           Returns:
               v (n dimensional list): the vector as an n dimesional list 
        '''
        vector_len = reduce(lambda x,y : x*y, self.dimension_sizes[1:], 1)
        vbytes = self.get_vector_bytes(index)
        
        # turn bytes into a list of python objects
        vflat = list()
        for i in range(vector_len):
            st = i * self.item_size
            vflat.append(self._bytes_to_item(vbytes[st: st + self.item_size]))
            
        # turn flat array into n-dimensional vector
        v = self._construct_vector(vflat, self.dimension_sizes[1:])
        return v  



if __name__ == '__main__':
    # Read vectors from the nmist data files
    img_reader = IDXReader('/home/bench/Projects/MNIST/mnist/t10k-images.idx3-ubyte')
    cls_reader = IDXReader('/home/bench/Projects/MNIST/mnist/t10k-labels.idx1-ubyte')

    for i in range(10):
        print(f'\nimage {i} is a {cls_reader.get_vector(i)[0]}')
        v = img_reader.get_vector(i)
        for row in v:
            for x in row:
                if x != 0:
                    print('#', end='')
                else:
                    print(' ', end='')
            print()
                
