def test(*args, **kwargs):

    print(kwargs)
    for key, value in kwargs.items():
        print("{0} = {1}".format(key, value))
    for arg in args:
        print("{0}".format(arg))




def evolve(x, a, defection = True,  **kwargs):
    test(kwargs)



    return




if __name__ =='__main__':
    import numpy as np
    N = 4
    x = np.ones(N)
    a = np.ones(N) *0.2
    evolve(x, a, Adj=1)