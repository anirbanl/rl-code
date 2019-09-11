import matplotlib.pyplot as plt
import numpy as np

def plot_lines(tuples, name, xlabel, ylabel):
    for x,y,n in tuples:
        plt.plot(x, y, label=n)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    plt.legend()
    plt.show()

if __name__== "__main__":
    x=np.linspace(0, 2, 100)
    y=np.random.normal(0,2, 100)
    t=[]
    t.append((x,y,'gaussian'))
    t.append((x,y**2,'gaussian_sq'))
    t.append((x,y**3,'gaussian_cube'))
    t.append((x,-y,'gaussian_neg'))
    plot_lines(t, 'Gaussian', 'X-axis', 'Y-axis')


