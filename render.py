import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation as anim


from IPython.display import HTML
from IPython.display import display


def animate_function(i, action, model, ax):
    edges = model['links']
    num_nodes = model['num_nodes']
    colors = model['colors']
    ax.clear()
    #ax.axis([-.5, 1.0, -1.5 ,1.0])
    ax.axis([0, 2.0, 0 ,2.0])
    frame = action[i]
    axis_x = frame[:,0]
    axis_y = frame[:,1]
    for edge_index in range(len(edges)):

        s_x = axis_x[edges[edge_index][0]]
        s_y = axis_y[edges[edge_index][0]]
        t_x = axis_x[edges[edge_index][1]]
        t_y = axis_y[edges[edge_index][1]]
        line = plt.Line2D((s_x,t_x), (s_y,t_y))
        ax.add_artist(line)
    for node_index in range(len(axis_x)):
        node_color = matplotlib.colors.to_rgb(colors[node_index])
        circle = plt.Circle((axis_x[node_index], axis_y[node_index]), .05, color=node_color)
        ax.add_artist(circle)
        ax.annotate(node_index, xy=(axis_x[node_index], axis_y[node_index]))


def animate(sequence, model):
    fig = plt.figure()
    ax = plt.axes()
    ani = anim.FuncAnimation(fig,
                                            animate_function,
                                            fargs=(sequence, model ,ax),
                                            frames=len(sequence))
    return HTML(ani.to_jshtml())


def save_animation(sequence, model, path):
    fig = plt.figure()
    ax = plt.axes()
    ani = anim.FuncAnimation(fig,
                                            animate_function,
                                            fargs=(sequence, model ,ax),
                                            frames=len(sequence))

    ani.save(path, writer='imagemagick', fps=5)
    plt.close()
