# NSGA-II-python_version-
The basic NSGA-II algorithm is implemented in python to apply to pytorch（updating）
There are still some areas for improvement to speed up the convergence in ‘F_NDSort.py’ when the number
  of objective function is lager than 2.




2019-------7-----10

I have updated the python version of NSGA-II, where the non-dominated sorting operator is updated to NDsort. The NDsort can select any number of individuals, (but its selection speed for many objective problem is not good),and the number of test problem is increased to 2, including DTLZ1 and DTLZ2. If you want more testing problem, you can contact me. I will write it for you. In next plan, I will implement the code of "LMOCSO ", See you. 
