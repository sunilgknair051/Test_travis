'''
@Sunilgknair051
'''

#import xdrlib
#import math
#import glob
import matplotlib.pyplot as plt
# import pylab
import numpy as np
from scipy.optimize import leastsq
#from pylab import genfromtxt  
#from matplotlib.backends.backend_pdf import PdfPages


#
#files = glob.glob('aray.txt')
#files.sort()
#print len(files)
#mddata = np.zeros((len(files),51,2))
#
#for ti, filen in enumerate(files):
#	f = open(filen,'r')
#	c = 0
#	for l in f:
#		data = [float(a) for a in l.split()]
#		mddata[ti] = data
#		c += 1
#
#xs = mddata[:,1]
#time = mddata[:,0]
#
#
#
#plt.plot(time,xs,label='xs',marker='o')
##plt.plot(xpos,xs,label='xs',marker='o')
##plt.plot(sh,label='stokes', marker='o')
##plt.plot(ys,label='ys',marker='o')
##plt.plot(zs,label='zs',marker='o')
##plt.plot(x, y)
#plt.legend()
#plt.title("YO!")
#plt.xlabel("Time")
#plt.ylabel("ANgVelocity")
#plt.show()
##plt.savefig("yes.pdf")
#

#!/usr/bin/python

#with open("aray.txt") as f:
#    data = f.read()
#
#data = data.split('\n')
#
#x = [row.split(' ')[0] for row in data]
#y = [row.split(' ')[1] for row in data]
#
#fig = plt.figure()
#
#ax1 = fig.add_subplot(111)
#
#ax1.set_title("Plot title...")    
#ax1.set_xlabel('your x label..')
#ax1.set_ylabel('your y label...')
#
#ax1.plot(x,y, c='r', label='points')
#
#leg = ax1.legend()
#
#plt.show()
#

X, Y = [], []
for line in open('aray1.txt', 'r'):
  values = [float(s) for s in line.split()]
  X.append(values[0])
  Y.append(values[1])

plt.plot(X, Y,label='points_single term',marker='o')
plt.legend()
plt.title("Taylor-Green-Plot1")
plt.xlabel("TSteps")
plt.ylabel("Values")
#plt.savefig("yes1.pdf")
#plt.show()


X1, Y1 = [], []
for line in open('aray3.txt', 'r'):
  values = [float(s) for s in line.split()]
  X1.append(values[0])
  Y1.append(values[1])

plt.plot(X1, Y1,label='points_added_terms',marker='o')
plt.legend()
plt.title("Taylor-Green")
plt.xlabel("TSteps")
plt.ylabel("Values")
#plt.savefig("yes2.pdf")
plt.show()




#def plot(x,exp,fit):
#    fig, ax = plt.subplots()
#    ax.plot(x, exp, 'x')
#    ax.plot(x, fit, '-')
#
#def FitFunc(time, ExpPars):
#    return ExpPars[0] * np.exp(- time / ExpPars[1])
#
#def fit_model(FitParsGuess, x, y): 
#
#    def func(FitPars):
#        ymodel = FitFunc(x,FitPars)
#        return y - ymodel
#    FitPars, ier = leastsq(func, FitParsGuess)
#    func_fit = FitFunc(time, FitPars)
#    return FitPars, func_fit  
#
#for line in open('aray1.txt', 'r'):
#  values = [float(s) for s in line.split()]
#  time = X.append(values[0])
#  exp_decay = Y.append(values[1])
##time      = values[0])
##exp_decay = np.array([1,0.8,.7,.6,.55,.47,.43,.39,.35])
#ExpParsGuess = np.array([1.,2.]) 
#FitPars, exp_decay_fit = fit_model(ExpParsGuess, time, exp_decay) 
#plot(time,exp_decay,exp_decay_fit)
#print(FitPars)



#
#mat0 = genfromtxt("aray.txt");
#mat1 = genfromtxt("aray.txt");
#plt.plot(mat0[:,0], mat0[:,1], label = "data0");
#plt.plot(mat1[:,0], mat1[:,1], label = "data1");
#plt.legend();
#plt.show();
