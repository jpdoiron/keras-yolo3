import glob
from numpy import genfromtxt
import numpy as np

train_dir = "data_set/"
image_list = []

digit = lambda x: int(''.join(filter(str.isdigit, x) or -1))

filenames = glob.glob(train_dir + '/*.png')
filenames.sort(key=digit)

originalSize = 200

output = genfromtxt(train_dir + "/output.txt", delimiter = ',')

dict1 = {}
names = []
cnt = 0

list_file = open('Custom_set.txt', 'w')


for filename in filenames: #assuming gif
    if(cnt%100==0):
        print(filename)
    list_file.write(filename)

    _,x,y,x2,y2 = output[cnt].astype(np.int)
    
    x2 = x+x2
    y2=y+y2
    
    list_file.write(' %s,%s,%s,%s,0'%(x,y,x2,y2))
    list_file.write("\n")
    cnt+=1

list_file.close()

