import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_feasible_polygon_with_path(polygon, history, limits, func_title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot polygon (feasible region)
    poly = Poly3DCollection([polygon], alpha=0.3, facecolor='cyan', edgecolor='black')
    ax.add_collection3d(poly)

    # Plot central path (list of x's)
    path = np.array([x for (x, _) in history])
    ax.plot(path[:, 0], path[:, 1], path[:, 2], marker='o', color='red', label='central path')

    # Set axis limits
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_zlim(limits[2])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(func_title + ": central path")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_central_path(func, inequalities, A, b, history, limits, func_title):
    triangle = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0])
    ]

    plot_feasible_polygon_with_path(triangle, history, limits, func_title)

def plot_feasible_polygon_with_path_2d(polygon, history, limits, func_title):
    fig, ax = plt.subplots()

    # Plot feasible polygon
    poly = plt.Polygon(polygon, closed=True, alpha=0.3, facecolor='cyan', edgecolor='black')
    ax.add_patch(poly)

    # Plot central path
    path = np.array([x for (x, _) in history])
    ax.plot(path[:, 0], path[:, 1], color='red', marker='o', label='central path')

    # Set limits
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.title(func_title+": central path")
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def plot_central_path_2d(func, inequalities, A, b, history, limits, func_title):
    polygon = [
        [0.0, 1.0],
        [2.0, 1.0],
        [2.0, 0.0],
        [1.0, 0.0]
    ]
    plot_feasible_polygon_with_path_2d(polygon, history, limits, func_title)


def plot_function_iterations(history, func_title, log_scale=False):

    obj_values= [f for (_, f) in history]
    plt.plot(range(len(obj_values)), obj_values)

    plt.xlabel("Iteration")
    plt.ylabel("Function value")
    if log_scale:
        plt.yscale('log')
    plt.title(func_title+": objective value vs outer iteration number")
    plt.show()

