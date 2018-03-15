import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import skimage.io as io
# Set snapshot directory
snapshots_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+"\\Webcam"
# Set up some fixed variable
npixel = 100 # number of pixel in 1 dimension of upperleft area
numIterations = 30
alpha = 0.5
timestamp = ['10:31','11:31','09:01','07:31','12:41','13:41','11:01','10:01','12:01','12:01']
timestamp0 = '07:00'
format = '%H:%M'
# Set up empty x & y
imageList = []
x = np.empty((10,(npixel*npixel+1)))
y = np.empty((10,1))
# importing snapshots
for counter in range(1,11):
    # update image value
    filename = "\\MontBlanc" + str(counter) + ".png"
    shot = io.imread(snapshots_dir + filename)
    imageList.append(shot)
    # update the first feature of x (overall greeness of a picture)
    x[(counter -1),0] = np.mean(shot[:, :, 1])
    # update y value
    y[(counter-1)] = int((datetime.strptime(timestamp[(counter-1)],format) -
                          datetime.strptime(timestamp0,format)).total_seconds()/60)

# Update green intensity value of 10000 pixels into feature vector x
for i in range(x.shape[0]):
    for j in range(1,x.shape[1]):
        for a in range(npixel):
            for b in range(npixel):
               x[i,j] = imageList[i][a,b,1]; #0 red, 1 green, 2 blue
			   
# pls use numpy . see baysian demo 3.6 to see why the above is inefficient code
# idea: select imageList[i][npixel,npixel,1]
# unravel --> shape[1,npixel*npixel] . U can use [:,none,none ]
# concantenate --> shape(10,npixel*npixel)


# Gradient descent
def gradientDescent(x, y, w, alpha, numIterations):
    risk = np.empty(numIterations)
    numIter = np.empty(numIterations)
    N = np.shape(x)[0]
    xTrans = x.transpose()
    for i in range(0, numIterations):
        numIter[i] = i + 1
        loss = y - x.dot(w)
        lossTrans = loss.transpose()
        risk[i] = (1/N) * lossTrans.dot(loss) # compute risk
        gradient = (2/N) * (xTrans.dot(x).dot(w) - xTrans.dot(y)) # gradient
        # update w
        w = w - alpha * gradient

    return numIter, risk # for plotting

# Init w value
w = np.zeros((np.shape(x)[1],1))
# Apply gradient descent
numIter, risk = gradientDescent(x, y, w, alpha, numIterations)
# plot
plt.scatter(x=numIter, y=risk)
plt.xlabel('Num Iterations')
plt.ylabel('Risk')
plt.title('Step size Alpha 0.5')
plt.savefig('prob5.png')

