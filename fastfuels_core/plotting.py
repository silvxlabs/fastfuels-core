"""
This module contains functions for plotting data from the fastfuels_core
package.
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.collections as collections


def plot_voxelized_tree(data, quantity="", **kwargs):
    viz_array = data.copy()
    viz_array[viz_array == 0] = -1
    grid = pv.UniformGrid()
    grid.dimensions = np.array(viz_array.shape) + 1
    grid.spacing = (1, 1, 1)
    grid.cell_data[quantity] = viz_array.flatten(order="F")
    grid = grid.threshold(0)
    # Set colormap max to 8

    pv.set_plot_theme("document")
    grid.plot(show_edges=False, show_grid=True, **kwargs)


def plot_tree_xy(treelist, return_fig_ax=False):
    fig, ax = plt.subplots()
    ax.scatter(treelist.data["X_m"], treelist.data["Y_m"], s=1)
    ax.set_aspect("equal")
    ax.autoscale_view()

    if return_fig_ax:
        return fig, ax
    else:
        plt.show()


def plot_tree_crowns(treelist, return_fig_ax=False):
    fig, ax = plt.subplots()
    patches = []
    face_colors = []
    for tree in treelist:
        circ = plt.Circle((tree.x, tree.y), tree.crown_radius)
        if tree.spcd == "202":
            face_colors.append("r")
        else:
            face_colors.append("g")
        patches.append(circ)
    plot_collection = collections.PatchCollection(
        patches, facecolors=face_colors, edgecolors="k", alpha=0.5
    )
    ax.add_collection(plot_collection)
    ax.autoscale_view()

    # Add red and green circles for legend
    ax.scatter([], [], c="r", alpha=0.5, label="Douglas-fir")
    ax.scatter([], [], c="g", alpha=0.5, label="Ponderosa pine")
    ax.legend()

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    if return_fig_ax:
        return fig, ax
    else:
        plt.show()


def get_canopy_profile(crown_radius, beta_params, n=14):
    """Returns a discretized canopy profile"""

    # beta parameters for ponderosa pine
    a, b, c = beta_params

    # Stirling's approximation for normalizing
    normalizer = np.sqrt(2 * np.pi) * (
        (a ** (a - 0.5) * b ** (b - 0.5)) / ((a + b) ** (a + b - 0.5))
    )

    z = np.linspace(0, 1, n)
    x = np.zeros_like(z)

    # discretized profile
    for i in range(n):
        x[i] = c * (((z[i] ** (a - 1)) * ((1 - z[i]) ** (b - 1)))) / normalizer

    # scale it up such that maximum point in x == crown radius
    factor = crown_radius / np.max(x)
    x *= factor
    z *= factor

    return x, z


def revolve_canopy(x, z):
    """Revolves the discrete profile points about the z axis"""

    n = x.shape[0]
    n_verts = n**2 - 3 * n + 4

    verts = np.zeros((n_verts, 3))
    theta = np.linspace(0, 2 * np.pi, n)

    verts[0, :] = [0, 0, 0]

    idx = 1
    for i in range(1, n - 1):
        for j in range(n - 1):
            verts[idx, 0] = x[i] * np.cos(theta[j])
            verts[idx, 1] = x[i] * np.sin(theta[j])
            verts[idx, 2] = z[i]
            idx += 1

    verts[-1, :] = [0, 0, z[-1]]

    return verts


def triangulate_verts(verts):
    """Triangulate the revolved point to a mesh"""

    n_verts = verts.shape[0]
    n = int((3 + np.sqrt(4 * n_verts - 7)) / 2)
    n_faces = 2 * n**2 - 6 * n + 4
    faces = np.zeros((n_faces, 4), dtype=int)

    idx = 0
    for i in range(1, n - 1):
        faces[idx] = [3, i, 0, i + 1]
        idx += 1
    faces[idx] = [3, n - 1, 0, 1]
    idx += 1

    for i in range(0, n - 3):
        j = i * (n - 1) + 1
        for k in range(j, j + n - 2):
            faces[idx] = [3, k, k + 1, k + n]
            idx += 1
            faces[idx] = [3, k, k + n, k + n - 1]
            idx += 1
        faces[idx] = [3, k + 1, j, k + 2]
        idx += 1
        faces[idx] = [3, k + 1, k + 2, k + n]
        idx += 1

    for i in range(n_verts - n, n_verts - 2):
        faces[idx] = [3, n_verts - 1, i, i + 1]
        idx += 1
    faces[idx] = [3, n_verts - 2, n_verts - n, n_verts - 1]

    return faces


def beta_canopy_verts_and_faces(tree, n=25):
    # def get_tree(xp, yp, zp, crown_radius, beta_params, n=25):
    """Returns nodes and faces of a beta canopy"""

    # 2D profile
    x, z = get_canopy_profile(
        tree.crown_radius,
        (tree.beta_canopy_a, tree.beta_canopy_b, tree.beta_canopy_c),
        n=n,
    )

    # now to 3D
    verts = revolve_canopy(x, z)

    # translate to (x,y,z)
    verts[:, 0] += tree.x
    verts[:, 1] += tree.y
    verts[:, 2] += tree.crown_base_height

    # create mesh
    faces = triangulate_verts(verts)

    return verts, faces


# def plot_beta_canopy_3d_plt(tree, ax=None, n=25):
#     """ Plots the canopy of a beta tree """
#
#     # get the nodes and faces
#     verts, faces = get_tree(tree, n=n)
#
#     # plot
#     if ax is None:
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#
#     ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
#                     color='green', alpha=0.5)
#
#     return ax


def get_tree_mesh(tree, n=25):
    verts, faces = beta_canopy_verts_and_faces(tree, n=n)
    mesh = pv.PolyData(verts, faces)
    return mesh


def plot_beta_canopy_3d_pyv(trees, n=25):
    # show the trees with pyvista (friendly VTK)
    pv.set_plot_theme("document")
    p = pv.Plotter()
    p.show_grid()
    try:
        for tree in trees:
            mesh = get_tree_mesh(tree, n=n)
            p.add_mesh(mesh, color="#00b500", smooth_shading=True)
            p.add_mesh(
                pv.Cylinder(
                    center=[tree.x, tree.y, (tree.height * (1 - tree.crown_ratio)) / 2],
                    direction=[0, 0, 1],
                    radius=tree.dbh / 200,
                    height=tree.height * (1 - tree.crown_ratio),
                ),
                color="#e5e7eb",
                smooth_shading=True,
            )
    except TypeError:
        tree = trees
        mesh = get_tree_mesh(trees, n=n)
        p.add_mesh(mesh, color="#00b500", smooth_shading=True)
        p.add_mesh(
            pv.Cylinder(
                center=[tree.x, tree.y, (tree.height * (1 - tree.crown_ratio)) / 2],
                direction=[0, 0, 1],
                radius=tree.dbh / 200,
                height=tree.height * (1 - tree.crown_ratio),
            ),
            color="#66493A",  # color="#e5e7eb",
            smooth_shading=True,
        )

    p.show()
