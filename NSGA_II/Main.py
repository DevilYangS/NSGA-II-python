import time


from public import P_objective, P_settings, P_generator,  NDsort,F_distance,F_mating,F_EnvironmentSelect


import matplotlib.pyplot as plt
import  numpy as np


# Copyright 2018 Yang Shang shang




def EA_Run(Generations, PopSize, M, Run, Problem, Algorithm):

    Generations, PopSize = P_settings.P_settings(Algorithm, Problem, M)
    Population, Boundary, Coding = P_objective.P_objective("init", Problem, M, PopSize)
    FunctionValue = P_objective.P_objective("value", Problem, M, Population)


    FrontValue = NDsort.NDSort(FunctionValue, PopSize)[0]
    CrowdDistance = F_distance.F_distance(FunctionValue, FrontValue)

    since = time.time()

    plt.ion()


    for Gene in range(Generations):

        MatingPool = F_mating.F_mating(Population, FrontValue, CrowdDistance)

        Offspring = P_generator.P_generator(MatingPool, Boundary, Coding, PopSize)

        FunctionValue_Offspring = P_objective.P_objective("value", Problem, M, Offspring)

        Population = np.vstack((Population, Offspring))
        FunctionValue = np.vstack((FunctionValue, FunctionValue_Offspring))



        Population, FunctionValue, FrontValue, CrowdDistance, MaxFront = F_EnvironmentSelect.F_EnvironmentSelect(Population, FunctionValue, PopSize)


        plt.clf()
        plt.scatter(FunctionValue[:, 0], FunctionValue[:, 1])
        plt.pause(0.001)

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
    Problem = "DTLZ2"

    EA_Run(Generations, PopSize, M, Run, Problem, Algorithm)




