#
# ev2.py: ev1 with the following modifications:
#          - self-adaptive mutation
#          - stochastic arithmetic crossover
#          - restructured code for better use of OO
#
# Note: EV2 still suffers from many of the weaknesses of EV1,
#       most particularly in the parent/survivor selection processes
#
# To run: python ev2.py --input ev2_example.cfg
#         python ev2.py --input my_params.cfg
#
#

import optparse
import sys
import yaml
import math
import copy
import numpy as np
from random import Random

ind_length = 10
MODE = 0 # { 0 : (1,1)-ES , 1 : (1+1)-ES }
sigma_list = [0.01, 0.1, 1]
init_sigma = 0.1

Gs = 0
G = 35.0

#EV2 Config class
class EV2_Config:
    """
    EV2 configuration class
    """
    # class variables
    sectionName='EV2'
    options={'populationSize': (int,True),
             'generationCount': (int,True),
             'randomSeed': (int,True),
             'minLimit': (float,True),
             'maxLimit': (float,True),
             'runs': (int, True)}

    #constructor
    def __init__(self, inFileName):
        #read YAML config and get EV2 section
        infile=open(inFileName,'r')
        ymlcfg=yaml.load(infile)
        infile.close()
        eccfg=ymlcfg.get(self.sectionName,None)
        if eccfg is None: raise Exception('Missing {} section in cfg file'.format(self.sectionName))

        #iterate over options
        for opt in self.options:
            if opt in eccfg:
                optval=eccfg[opt]

                #verify parameter type
                if type(optval) != self.options[opt][0]:
                    raise Exception('Parameter "{}" has wrong type'.format(opt))

                #create attributes on the fly
                setattr(self,opt,optval)
            else:
                if self.options[opt][1]:
                    raise Exception('Missing mandatory parameter "{}"'.format(opt))
                else:
                    setattr(self,opt,None)

    #string representation for class data
    def __str__(self):
        return str(yaml.dump(self.__dict__,default_flow_style=False))


# Sphere model
#
def fitnessFunc(x):
    return sum(x*x)


#Find index of worst individual in population
def findWorstIndex(l):
    minval=l[0].fit
    imin=0
    for ind in range(len(l)):
        if l[ind].fit < minval:
            minval=l[ind].fit
            imin=ind
    return imin


#Print some useful stats to screen
def printStats(pop,gen):
    print('Generation:',gen)
    avgval=0
    maxval=pop[0].fit
    sigma=pop[0].sigma
    for ind in pop:
        avgval+=ind.fit
        if ind.fit > maxval:
            maxval=ind.fit
            sigma=ind.sigma
        #print(str(ind.x)+'\t'+str(ind.fit)+'\t'+str(ind.sigma))

    #print('Max fitness',maxval)
    #print('Sigma',sigma)
    #print('Avg fitness',avgval/len(pop))
    #print('')


#A simple Individual class
class Individual:
    minSigma=1e-100
    maxSigma=1
    #Note, the learning rate is typically tau=A*1/sqrt(problem_size)
    # where A is a user-chosen scaling factor (optional) and problem_size
    # for real and integer vector problems is usually the vector-length.
    # In our case here, the vector length is 1, so we choose to use a learningRate=1
    learningRate=(2*(ind_length)**0.5)**(-0.5)
    poplearningRate=(2*ind_length)**(-0.5)
    minLimit=None
    maxLimit=None
    cfg=None
    prng=None
    fitFunc=None
    a = 0.89

    def __init__(self,randomInit=True):
        self.x = np.ones([ind_length,1])
        self.fit = self.__class__.fitFunc(self.x)
        self.sigma = init_sigma
        # self.sigma=np.full(shape=[ind_length,1], fill_value=init_sigma)

    def mutate(self):
        self.x=self.x+self.sigma*np.random.normal(0,1,[10,1])

    def mutate_sigma(self):
        global Gs
        global G
        Ps = Gs/G
        if Ps > 0.2:
            self.sigma = self.sigma/self.a
        elif Ps < 0.2:
            self.sigma = self.sigma*self.a
        else:  # Ps == 1/5
            self.sigma = self.sigma

        if self.sigma < self.minSigma:
            self.sigma = self.minSigma
        if self.sigma > self.maxSigma:
            self.sigma = self.maxSigma

    def evaluateFitness(self):
        self.fit=self.__class__.fitFunc(self.x)


#EV2: EV1 with self-adaptive mutation & stochastic crossover
#
def ev2(cfg):
    #start random number generator
    global Gs
    Gs = 0
    prng=Random()
    prng.seed(cfg.randomSeed)

    #set Individual static params: min/maxLimit, fitnessFunc, & prng
    Individual.minLimit=cfg.minLimit
    Individual.maxLimit=cfg.maxLimit
    Individual.fitFunc=fitnessFunc
    Individual.prng=prng
    Individual.cfg=cfg
    #random initialization of population
    population=[]
    for iteration in range(cfg.populationSize):
        ind=Individual()
        population.append(ind)

    #print stats
    #printStats(population,0)

    #evolution main loop
    for iteration in range(cfg.generationCount):

        parent = population[0]
        if(parent.fit<0.005):
            print("converge!", end=' ')
            printStats(population, iteration+1)
            break
        if iteration == cfg.generationCount-1:
            print("Failure ")
            printStats(population, iteration+1)

        #recombine
        child=copy.copy(population[0])

        #random mutation
        child.mutate()

        #update child's fitness value
        child.evaluateFitness()

        if iteration % int(G) == 0:
            child.mutate_sigma()
            Gs = 0
            parent.sigma=child.sigma

        if MODE == 0:
            # survivor selection: (1,1)-ES replace parent
            population[0] = child
            Gs += 1
        elif MODE == 1:
            # survivor selection: (1+1)-ES replace worst
            iworst=findWorstIndex(population)
            if child.fit < population[iworst].fit:
                population[iworst]=child
                Gs += 1


        #print stats
        #printStats(population,i+1)


#
# Main entry point
#
def main(argv=None):
    if argv is None:
        argv = sys.argv

    try:
        #
        # get command-line options
        #
        parser = optparse.OptionParser()
        parser.add_option("-i", "--input", action="store", dest="inputFileName", help="input filename", default=None)
        parser.add_option("-q", "--quiet", action="store_true", dest="quietMode", help="quiet mode", default=False)
        parser.add_option("-d", "--debug", action="store_true", dest="debugMode", help="debug mode", default=False)
        (options, args) = parser.parse_args(argv)

        #validate options
        if options.inputFileName is None:
            raise Exception("Must specify input file name using -i or --input option.")

        #Get EV2 config params
        cfg=EV2_Config(options.inputFileName)

        #print config params
        #print(cfg)

        #run EV2
        global MODE
        global init_sigma
        for m in range(2):
            MODE = (m+1)%2
            print("MODE: ", MODE)
            for s in sigma_list:
                init_sigma = s
                print("sigma: ", s)
                for r in range(cfg.runs):
                    print('Run #', r, end=': ')
                    ev2(cfg)
                print('-'*20)
        #ev2(cfg)

        #if not options.quietMode:
        #    print('EV2 Completed!')

    except Exception as info:
        if 'options' in vars() and options.debugMode:
            from traceback import print_exc
            print_exc()
        else:
            print(info)


if __name__ == '__main__':
    main()

