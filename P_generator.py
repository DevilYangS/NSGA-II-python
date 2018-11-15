import numpy as np

def P_generator(MatingPool,Boundary,Coding,MaxOffspring):
# % 交叉, 变异并生成新的种群
# % 输入: MatingPool, 交配池, 其中每第i个和第i + 1
# 个个体交叉产生两个子代, i为奇数
# % Boundary, 决策空间, 其第一行为空间中每维的上界, 第二行为下界
# % Coding, 编码方式, 不同的编码方式采用不同的交叉变异方法
# % MaxOffspring, 返回的子代数目, 若缺省则返回所有产生的子代, 即和交配池的大小相同
# % 输出: Offspring, 产生的子代新种群

    N, D = MatingPool.shape
    if MaxOffspring < 1 or MaxOffspring > N:
       MaxOffspring = N
    if Coding == "Real":
       ProC = 1
       ProM = 1/D
       DisC = 20
       DisM = 20
       Offspring = np.zeros((N, D))
       for i in range(0,N,2):
           beta = np.zeros((D,))
           miu = np.random.random((D,)) #np.random.rand(D,)
           beta[miu <= 0.5] = (2 * miu[miu <= 0.5])**(1/(DisC+1))
           beta[miu > 0.5] = (2-2 * miu[miu > 0.5]) ** (-1 / (DisC + 1))
           beta = beta * ((-1) ** (np.random.randint(0, 2, (D,))))
           beta[np.random.random((D,)) > ProC] = 1

           Offspring[i, :] = ((MatingPool[i, :] + MatingPool[i+1, :] )/2) + (np.multiply(beta, (MatingPool[i, :] - MatingPool[i+1, :])/2 ))
           Offspring[i+1, :] = ((MatingPool[i, :] + MatingPool[i+1, :] )/2) - (np.multiply(beta, (MatingPool[i, :] - MatingPool[i+1, :])/2 ))
       Offspring_temp = Offspring[:MaxOffspring,:]
       # print(range(MaxOffspring,Offspring.shape[0]))
       # np.delete(Offspring, range(MaxOffspring,Offspring.shape[0]), axis=0) 并没有真正的对 对象进行操作，仅仅你是个浅操作
       Offspring = Offspring_temp

       if MaxOffspring == 1:
           MaxValue = Boundary[0,:]
           MinValue = Boundary[1,:]
       else:
           MaxValue = np.tile(Boundary[0,:],(MaxOffspring,1))
           MinValue = np.tile(Boundary[1,:],(MaxOffspring,1))

       #np.bitwise_and 用于矩阵的逻辑运算
       k = np.random.random((MaxOffspring, D))
       miu = np.random.random((MaxOffspring, D))
       Temp = np.bitwise_and(k <= ProM, miu <0.5)

       Offspring[Temp] = Offspring[Temp] + np.multiply((MaxValue[Temp] - MinValue[Temp]), ((2 * miu[Temp] + np.multiply(
           1 - 2 * miu[Temp],
           (1 - (Offspring[Temp] - MinValue[Temp]) / (MaxValue[Temp] - MinValue[Temp])) ** (DisM + 1))) ** (1 / (
                   DisM + 1)) - 1))

       Temp = np.bitwise_and(k <= ProM, miu >= 0.5)
       
       Offspring[Temp] = Offspring[Temp] + np.multiply((MaxValue[Temp] - MinValue[Temp]), (1-((2 *(1-miu[Temp])) + np.multiply(
           2 * (miu[Temp]-0.5),
           (1 - (MaxValue[Temp] - Offspring[Temp]) / (MaxValue[Temp] - MinValue[Temp])) ** (DisM + 1))) ** (1 / (
                   DisM + 1)) ))

       Offspring[Offspring > MaxValue] = MaxValue[Offspring>MaxValue]
       Offspring[Offspring < MinValue] = MinValue[Offspring < MinValue]

    elif Coding == "Binary":
        Offspring = []


    return Offspring














