import numpy
import scipy
from scipy import optimize
from scipy import integrate
from scipy import linalg
from operator import itemgetter
import random
import math

"""
This program modifies DisksInPlaneAnnealing to use a binormal measure instead
of the uniform measure. That means that the "size" of a disk is not simply its
area, but the integral of the probability density function over the disk.
We're going to take the mean of the sample as (0,0) and the standard deviation
and variance are both 1.
"""

eps = 0.01
t = int(input("How many random points (according to a binormal\
 distribution) would you like in your set? "))
m = int(input("How many iterations would you like to run? "))

#The probability density function (PDF) is the standard bivariate
#normal distribution

def PDF(y,x):
    return 1/(2*numpy.pi)*numpy.exp(-(x**2+y**2)/2)


#We need to define the measure of a disk as the integral of PDF over that disk
                             
#We'll take one input, a 3-tuple of the disks's center and radius
#and return the integral of the PDF over the disk
#The bounds of integration are simply the radius and the upper and lower
#half-circles                            
def meas(x):
    a = x[0]
    b = x[1]
    r = x[2]
    return scipy.integrate.dblquad(PDF, a-r, a+r, \
                                   lambda x: -math.sqrt(r**2-(x-a)**2)+b, \
                                   lambda x: math.sqrt(r**2-(x-a)**2)+b)

#Testing whether a point p = (p,q) is in a disk x = (a, b, radius)
#It's the same as with the uniform measure

def is_inside(p,x):
    r = x[2]
    squares = [(x[i]-p[i])**2 for i in range(2)]
    if sum(squares) < r**2:
        return True
    else:
        return False

#Counting the number of points in a disk is also the same as the uniform measure
def count_points(points, x): 
    count = 0
    for point in points:
        if is_inside(point, x): 
            count += 1                  
    return count

#The function

def find_epsilon_net(x):
    if count_points(S,x)==0:
            return -1*(meas(x)[0]) #Maximize the measure
    else:
        return count_points(S,x) #If not, return a value that's so big it couldn't work

#When we generate a random sample of points we now want them to be concentrated
#around [0,0], within a close range, and without any relationship between the
#"randomness" of the x- and y-coordinated. That is, we want there to be zero
#covariance. So the mean and covariance matrix are defined as follows.

mean = [0,0]
cov = [[1,0],[0,1]]

result_text = open('/home/jordanml/Documents/DisksInPlane/BinormalMaxMeasResults.txt','wb')

result_text.write("This test used %d sample points and ran %d trials.\r\n" %(t,m))

#count = 0 #Keep track of the number of non-epsilon nets

#We need a place to store the results once we've run the algorithm, to remember them when we want to run it again.
all_results = []
all_points = []
#Keep track of the measures, to eventually compute their average
count = 0
for i in range(m):
    #Generating a random set of points according to this distribution
    S = numpy.random.multivariate_normal(mean, cov, t)
    #Generate a random starting center and radius
    center0 = numpy.random.multivariate_normal(mean, cov,int(1))
    radius0 = numpy.random.normal(0,1)
    rad0 = abs(numpy.array(radius0))
    #Transform these into a starting coordinate
    x0 = numpy.append(center0, rad0)
    result = optimize.anneal(find_epsilon_net, x0, schedule = "boltzmann",\
                             full_output = True, lower=[-2,-2,0],\
                             upper=[2,2,1], maxiter = 500)
    #Write the result of the simulated annealing
    result_text.write(str(result[0:2]))
    result_text.write("\r\n")
    result_text.write(str(S.tolist()))
    count += result[1] #Update the count, adding on the new measure
#   if result[1] == 0:
    result_text.write("\r\n")
#        count += 1
#    else:
#        result_text.write("Undetermined. Shame.\r\n") 
    """We're now going to run a second pass of annealing"""
    #Keep track of the restuls in an array
    result_text.write(str(S.tolist()))
    full_results = numpy.append(result[0],[result[1]]).tolist()
    all_results.append(full_results)
    all_points.append(S.tolist())

average_measure = count/m
result_text.write("The average measure is: ")
result_text.write(str(average_measure))
result_text.write("\r\n")


#result_text.write("There were a total of %d non-epsilon-nets found.\r\n" %(count))

#We're going to now extract the top 5 best results to run again
#First, sort each array in all_results so that the first value is
#The number of points in each disk
backwards = [all_results[i][::-1] for i in range(m) if all_results[i][-1]!=0]

#Now, sort the array so that those with the fewest points come first
backwards.sort()

#Take the first 1/5 of the results in good_results
endpoint = m//5
good_results = backwards[0:int(endpoint)]

#Find the location of each good result in all_result
#And remember its location
tracking = []
for i in range(len(good_results)):
    tracking.append(all_results.index(good_results[i][::-1]))

good_points = []
for i in tracking:
    good_points.append(all_points[i])

# We want to take 1/5 of the results, presumable the "best" ones, and run them through the algorithm again.

result_text.write("\r\nWe're now going to take the %d best results and run them again.\r\n"%(m//5))
for i in range(len(good_results)):
    S = numpy.array(good_points[i])
    x0 = numpy.array(good_results[i][1:][::-1])
    result = optimize.anneal(find_epsilon_net, x0,schedule = "boltzmann",\
                             full_output = True, lower=[-1,-1,0], upper=[1,1,1],\
                             maxiter = 500)
    result_text.write(str(result[0:2]))
#    if result[1] == 0:
#        result_text.write("Not an epsilon net. What a great success!\r\n")
#        S_special=S
#    else:
#        result_text.write("Undetermined.Shame.\r\n")

result_text.close()

    
    
