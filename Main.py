import time
from P_settings import *
# from P_objective import *
import P_objective
from F_NDSort import *
from F_distance import *
from F_mating import *
from P_generator import *
from F_EnvironmentSelect import *
import matplotlib.pyplot as plt
import profile


# Copyright 2018 Yang Shang shang

def EA_Run(Generations, PopSize, M, Run, Problem, Algorithm) :

    Generations, PopSize = P_settings(Algorithm, Problem, M)
    Population, Boundary, Coding = P_objective.P_objective("init", Problem, M, PopSize)
    

    FunctionValue = P_objective.P_objective("value", Problem, M, Population)
    
    FrontValue = F_NDSort(FunctionValue, "half")[0]#
    CrowdDistance = F_distance(FunctionValue, FrontValue)

    since = time.time()
    plt.ion()

    for Gene in range(100):

        MatingPool = F_mating(Population, FrontValue, CrowdDistance)

        Offspring = P_generator(MatingPool, Boundary, Coding, PopSize)

        FunctionValue_Offspring = P_objective.P_objective("value", Problem, M, Offspring)

        Population = np.vstack((Population, Offspring))
        FunctionValue =np.vstack((FunctionValue, FunctionValue_Offspring))
        # Population = np.loadtxt('population.txt')
        # FunctionValue = np.loadtxt('FunctionValue.txt')
        Population, FunctionValue, FrontValue, CrowdDistance, MaxFront = F_EnvironmentSelect(Population, FunctionValue, PopSize)

        plt.clf()
        plt.plot(FunctionValue[:, 0], FunctionValue[:, 1], "*")
        plt.pause(0.01)


        print(Algorithm,"Run :",Gene,"代，Complete：",100*Gene/Generations,"%, time consuming:",np.round(time.time()-since,2),"s")

    FunctionValueNon = FunctionValue[(FrontValue==1)[0],:]
    plt.plot(FunctionValueNon[:, 0], FunctionValueNon[:, 1], "*")
    plt.ioff()
    plt.show()



if __name__ == "__main__":
    Generations = 100
    PopSize = 100
    M = 2
    Run = 1
    Algorithm = "NSGA-II"
    Problem = "DTLZ1"
    # profile.run("EA_Run(Generations, PopSize, M, Run, Problem, Algorithm)")
    EA_Run(Generations, PopSize, M, Run, Problem, Algorithm)




