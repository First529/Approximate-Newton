import matplotlib.pyplot as plt

def plot_graph(x, y, xl, yl, title):
    fig = plt.figure(figsize=(12, 6)) # set figure size
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(title)
    plt.yscale('log')
    plt.plot(x[0], y[0], label='Newton') 
    plt.plot(x[1], y[1], label='Sub-sampled Newton') 
    plt.plot(x[2], y[2], label='GD') 
    plt.plot(x[3], y[3], label='Nesterov GD') 
    plt.plot(x[4], y[4], label='Nesterov Sub-sampled Newton') 
    plt.legend()
