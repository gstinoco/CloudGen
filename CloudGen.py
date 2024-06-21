"""
All the codes presented below were developed by:
    Dr. Gerardo Tinoco Guerrero
    Universidad Michoacana de San Nicolás de Hidalgo
    gerardo.tinoco@umich.mx

With the funding of:
    National Council of Humanities, Sciences, and Technologies, CONAHCyT (Consejo Nacional de Humanidades, Ciencias y Tecnologías, CONAHCyT). México.
    Coordination of Scientific Research, CIC-UMSNH (Coordinación de la Investigación Científica de la Universidad Michoacana de San Nicolás de Hidalgo, CIC-UMSNH). México
    Aula CIMNE-Morelia. México

Date:
    April, 2023.

Last Modification:
    October, 2023.
"""

import os
import dmsh
import random
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPoint

def CreateCloud(xb, yb, num):
    '''
    This function generates a cloud of points within a specified boundary defined by the (xb, yb) coordinates.
    It calculates the maximum distance between boundary nodes and uses it to determine node spacing.
    The resulting point cloud contains nodes within the boundary, with boundary nodes marked as 1 in the third column of X.

    Parameters:
        xb          ndarray         Array of x-coordinates of boundary nodes.
        yb          ndarray         Array of y-coordinates of boundary nodes.
        num         int             Number of desired nodes in the cloud.

    Returns:
        X           ndarray         Array with the coordinates of the cloud of points with boundary node markings.
        cells       ndarray         Array representing the mesh cells generated.
    
    Example:
        xb       = np.array([0.0, 1.0, 1.0, 0.0])
        yb       = np.array([0.0, 0.0, 1.0, 1.0])
        num      = 100
        X, cells = CreateCloud(xb, yb, num)
    '''

    # Find the maximum distance between the boundary nodes.
    dist  = np.max(np.sqrt(np.diff(xb.T)**2 + np.diff(yb.T)**2))                    # Calculate the distance between nodes based on the maximum distance.
    dist /= num                                                                     # Calculate the distance between nodes based on the maximum distance.

    # Create the Triangulation
    pb       = np.column_stack((xb, yb))                                            # Stack xb and yb arrays as columns to create point coordinates.
    geo      = dmsh.Polygon(pb)                                                     # Create a Polygon object from the boundary points.
    X, cells = dmsh.generate(geo, dist)                                             # Generate a triangulation using dmsh.

    # Create a polygon
    poly = Polygon(pb).buffer(-dist/4)                                              # Create a buffer zone around the boundary.

    # Check if points are within the buffer zones
    points      = [Point(point[0], point[1]) for point in X]                        # Convert the cloud points to Shapely Point objects.
    inside_poly = np.array([i.within(poly) for i in points])                        # Check wether points are inside the buffer zone.

    # Add a third column to X and mark boundary nodes as 1
    X = np.column_stack((X, 1 - inside_poly.astype(int)))                           # Add a column to X to mark boundary nodes.

    return X, cells

def CreateCloud_Holes(xb, yb, num, h_coor):
    '''
    This function generates a cloud of points within a specified boundary defined by the (xb, yb) coordinates.
    It calculates the maximum distance between boundary nodes and uses it to determine node spacing.
    The resulting point cloud contains nodes within the boundary, with boundary nodes marked as 1 in the third column of X.

    Parameters:
        xb          ndarray         Array of x-coordinates of boundary nodes.
        yb          ndarray         Array of y-coordinates of boundary nodes.
        num         int             Number of desired nodes in the cloud.
        h_coor      ndarray         Array with the coordinates and radius for the holes.
                                    (x-coordinate, y-coordinate, radius)

    Returns:
        X           ndarray         Array with the coordinates of the cloud of points with boundary node markings.
        cells       ndarray         Array representing the mesh cells generated.
    
    Example:
        xb       = np.array([0.0, 1.0, 1.0, 0.0])
        yb       = np.array([0.0, 0.0, 1.0, 1.0])
        num      = 100
        h_coor   = [(0.5, 0.3, 0.05)]
        X, cells = CreateCloud_Holes(xb, yb, num, h_coor)
    '''

    # Find the maximum distance between the boundary nodes.
    dist  = np.max(np.sqrt(np.diff(xb.T)**2 + np.diff(yb.T)**2))                    # Calculate the distance between nodes based on the maximum distance.
    dist /= num                                                                     # Calculate the distance between nodes based on the maximum distance.

    # Create the Triangulation
    pb       = np.column_stack((xb, yb))                                            # Stack xb and yb arrays as columns to create point coordinates.
    geo      = dmsh.Polygon(pb)                                                     # Create a Polygon object from the boundary points.

    # Create holes by subtracting circles from the polygon
    for hole_x, hole_y, hole_radius in h_coor:                                      # Take the coordinates and radius of all the holes.
        geo -= dmsh.Circle([hole_x, hole_y], hole_radius)                           # Subtract the holes from the Polygon.

    X, cells = dmsh.generate(geo, dist)                                             # Generate a triangulation using dmsh.

    # Create a polygon and hole buffers
    poly         = Polygon(pb).buffer(-dist/4)                                      # Create a buffer zone around the boundary.
    hole_buffers = [Point(hole_x, hole_y).buffer(hole_radius).buffer(dist/4) for hole_x, hole_y, hole_radius in h_coor]
                                                                                    # Create a buffer zone around each hole.

    # Initialize an array A to store markings for each point
    points = [Point(point[0], point[1]) for point in X]                             # Convert the cloud points to Shapely Point objects.
    A      = np.zeros([len(points),1])                                              # Initialize A to store the markings.

    # Mark points as 1 if they are outside the polygon, 2 if they are inside any of the hole buffers
    for i, point in enumerate(points):                                              # For each of the points.
        if not point.within(poly):                                                  # Check wether points are not inside the buffer zone of the external boundary.
            A[i] = 1                                                                # Add 1 to that point.
        else:                                                                       # If it is not in the external boundary.
            for j, hole_buffer in enumerate(hole_buffers):                          # For each of the holes.
                if point.within(hole_buffer):                                       # Check wether points are not inside the buffer zone of the holes.
                    A[i] = 2                                                        # Add 2 to that point.
                    break

    # Add a third column to X and mark boundary nodes as 1
    X = np.column_stack((X, A))                                                     # Add a column to X to mark boundary nodes.

    return X, cells

def GraphCloud(p, save = False, nom = '', show = True):
    """
    This function creates a scatter plot of a cloud of points where points are colored based on the position in the cloud:
    Blue markers are used for inner points, and red markers for boundary points.
    The plot is saved as PNG and EPS files with the specified name and displayed to the screen.

    Parameters:
        p           ndarray         Array with the coordinates of the cloud of points with boundary node markings.
        nom         str             Name for the generated plot and file.
        show        bool            Flag to indicate whether the figure should be shown on screen.
        holes       bool            Flag to indicate whether holes are present.

    Returns:
        None
    
    Example:
        p = np.array([[1.0, 2.0, 0],
                      [2.0, 3.0, 1],
                      [3.0, 4.0, 0]])
        GraphCloud(p, "example_plot")
    """

    # Parameter initialization
    if save:
        nomp = nom + '.png'                                                         # Define the file name for the saved plot.
        nome = nom + '.eps'                                                         # Define the file name for the saved plot.
    else:
        show = True

    color = ['blue' if x == 0 else 'red' for x in p[:,2]]                           # Determine marker color based on the third column.

    # Cloud plot
    plt.rcParams["figure.figsize"] = (16, 12)                                       # Configure plot size.
    plt.scatter(p[:,0], p[:,1], c=color, s = 20)                                    # Create scatter plot.
    plt.title(nom)                                                                  # Set plot title.
    plt.axis('equal')
    if show:
        plt.show()                                                                  # Display the plot.
    if save:
        plt.savefig(nomp)                                                           # Save the plot as a PNG file.
        plt.savefig(nome, format='eps', bbox_inches='tight')                        # Save the plot as a EPS file.
    plt.close()                                                                     # Close the plot to release resources.

def GridToCloud(x, y, holes = False, num = 1, h_coor = [(0.5, 0.7, 0.05)]):
    """
    This function takes grid coordinates (x, y) and generates a cloud of points within the range [0, 1] x [0, 1].
    It scales the coordinates to fit in this range, creates boundaries, and then generates the cloud of points.
    
    The resulting cloud of point coordinates and triangulation are returned.

    Parameters:
        x           ndarray         Array of x-coordinates of grid points.
        y           ndarray         Array of y-coordinates of grid points.
        holes       bool            Flag to indicate whether holes are present (default is False).
        num         int             Number of desired nodes in the cloud (default is 1).
        h_coor      ndarray         Array with the coordinates and radius for the holes.
                                    (x-coordinate, y-coordinate, radius)

    Returns:
        X           ndarray         Array with the coordinates of the cloud of points with boundary node markings.
        cells       ndarray         Array representing the triangulation cells generated.

    Example:
        x = np.array([[0.0, 1.0, 2.0],
                      [0.0, 1.0, 2.0],
                      [0.0, 1.0, 2.0]])
        y = np.array([[0.0, 0.0, 0.0],
                      [1.0, 1.0, 1.0],
                      [2.0, 2.0, 2.0]])
        X, cells = GridToCloud(x, y, holes = False, num = 100)
    """

    # Scale the region to fit in [0, 1] x [0, 1]
    scale = max(x.max(),y.max())                                                    # Find the bigger coordinate.
    x     = (x - x.min())/scale                                                     # Scale x-coordinates.
    y     = (y - y.min())/scale                                                     # Scale y-coordinates.

    # The dimensions of the Mesh.
    m, n = x.shape[0], x.shape[1]                                                   # Get the dimensions of the grid

    # The boundaries of the Mesh.
    # Concatenate boundary points in the desired order and reshape into a column vector
    xb = np.hstack([x[m-1, :], x[1:m-1, n-1][::-1], x[0, :][::-1], x[1:m-1, 0]]).reshape(-1,1)
    yb = np.hstack([y[m-1, :], y[1:m-1, n-1][::-1], y[0, :][::-1], y[1:m-1, 0]]).reshape(-1,1)

    if not holes:
        X, cells = CreateCloud(xb, yb, num)                                         # Generate a point cloud without holes
    else:
        X, cells = CreateCloud_Holes(xb, yb, num, h_coor)                           # Generate a point cloud with holes

    return X, cells

def Randomize(p, me):

    ## Variable Initialization.
    m        = len(p[:, 0])                                                         # Get the number of boundary nodes.
    boundary = MultiPoint(p[p[:, 2] == 1][:, :2])                                   # Get the boundary nodes.
    pol      = boundary.convex_hull                                                 # Create a polygon.

    ## Check how big the movement should be.
    if me == '1':
        r = 0.05
    elif me == '2':
        r = 0.02
    elif me == '3':
        r = 0.01
    
    ## Randomly move the nodes.
    for i in range(0, m):
        if p[i, 2] == 0:
            inside_poly = False
            while not inside_poly:
                move_x   = p[i, 0]*random.uniform(-r, r)
                move_y   = p[i, 1]*random.uniform(-r, r)

                new_point = Point(p[i, 0] + move_x, p[i, 1] + move_y)
                inside_poly = pol.contains(new_point)

                if inside_poly:
                    p[i, 0] += move_x
                    p[i, 1] += move_y

    return p

def OrderNodes(p):
    """
    Reorders nodes in a cloud of points so that exterior boundary nodes are placed first in counterclockwise order, followed by interior boundary
    nodes in counterclockwise order, and finally interior nodes.
    This facilitates normal vector calculations on the boundaries.

    Parameters:
        p           ndarray         Array with the coordinates of nodes in the point cloud and a flag indicating whether the node is a boundary node or not.
                                    The columns of p correspond to:
                                        0: x-coordinates.
                                        1: y-coordinates.
                                        2: 0 for interior nodes, 1 for external boundary nodes, and 2 for internal boundary nodes.
                                    (x-coordinate, y-coordinate, radius)
    
    Returns:
        p           ndarray         Array with the rearranged coordinates.

   Example:
        p = np.array([[1.0, 2.0, 0],
                  [2.0, 3.0, 1],
                  [3.0, 4.0, 0],
                  [4.0, 5.0, 2]])
        p_reordered = OrdenNodes(X)
    """

    # Separate exterior boundary nodes, interior boundary nodes, and interior nodes
    interior_nodes    = p[p[:, 2] == 0]                                             # Interior nodes.
    bound_outer_nodes = p[p[:, 2] == 1]                                             # Exterior boundary nodes.
    bound_inner_nodes = p[p[:, 2] == 2]                                             # Interior boundary nodes.

    # Center of Mass
    center = np.mean(bound_outer_nodes[:, 0:2], axis=0)                             # Calculate the center of mass for exterior boundary nodes.

    # Angels
    angles = np.arctan2(bound_outer_nodes[:, 1] - center[1], bound_outer_nodes[:, 0] - center[0])
                                                                                    # Calculate angles from the center of mass to exterior boundary nodes.

    #Sorting of boundary nodes
    idx_sorted = np.argsort(angles)                                                 # Sort exterior boundary nodes counterclockwise.
    bound_outer_nodes = bound_outer_nodes[idx_sorted]                               # Sort exterior boundary nodes counterclockwise.

    # For inner boundary nodes
    if len(bound_inner_nodes) > 0:
        center = np.mean(bound_inner_nodes[:, 0:2], axis=0)                         # Calculate the center of mass for interior boundary nodes
        angles = np.arctan2(bound_inner_nodes[:, 1] - center[1], bound_inner_nodes[:, 0] - center[0])
                                                                                    # Calculate angles from the center of mass to interior boundary nodes

        idx_sorted = np.argsort(angles)                                             # Sort interior boundary nodes counterclockwise
        bound_inner_nodes = bound_inner_nodes[idx_sorted]                           # Sort interior boundary nodes counterclockwise

    # Node concatenation
    p = np.concatenate([bound_outer_nodes, bound_inner_nodes, interior_nodes])
                                                                                    # Concatenate sorted exterior boundary nodes, sorted interior boundary nodes, and interior nodes

    return p

def Normals(p):
    '''
    Calculate boundary nodes and their normal vectors for a given set of nodes.

    Parameters:
        p           ndarray         Array of nodes represented as a 2D array with three columns: 
                                    The columns of X correspond to:
                                        0: x-coordinates.
                                        1: y-coordinates.
                                        2: 0 for interior nodes, 1 for external boundary nodes, and 2 for internal boundary nodes.

    Returns:
        pb          ndarray         Array of boundary nodes, including both exterior and interior boundary nodes.
        vecs        ndarray         Array of normal vectors corresponding to the boundary nodes.
    '''
    # Exterior boundary.
    pb_o, nor_o = CalculateNormals(p, node_type=1)                                  # Calculate boundary nodes and normal vectors for exterior boundary nodes.

    # Interior boundary.
    if np.max(p[:, 2]) == 2:                                                        # Check if there are interior boundary nodes (node type 2) in the input data.
        pb_i, nor_i = CalculateNormals(p, node_type=2)                              # Calculate boundary nodes and normal vectors for interior boundary nodes.
        pb = np.vstack([pb_o, pb_i])                                                # Concatenate the exterior and interior boundary nodes.
        vecs = np.vstack([pb_o[:, 0:2] + nor_o, pb_i[:, 0:2] + nor_i])              # Concatenate the exterior and interior boundary nodes and their normal vectors.
    else:                                                                           # No interior boundary nodes, use exterior boundary nodes and their normal vectors only
        pb = pb_o                                                                   # Use only exterior boundary nodes.
        vecs = pb_o[:, 0:2] + nor_o                                                 # Create the normal vectors.

    return pb, vecs

def CalculateNormals(p, node_type):
    '''
    Calculate boundary nodes and their normal vectors for a specific node type.

    Parameters:
        p           ndarray         An array of nodes represented as a 2D array with three columns: 
                                    The columns of X correspond to:
                                        0: x-coordinates.
                                        1: y-coordinates.
                                        2: 0 for interior nodes, 1 for external boundary nodes, and 2 for internal boundary nodes.
        node_type   int             The type of nodes to calculate normals for (1 for exterior, 2 for interior).

    Returns:
        pb          ndarray         A 2D array of boundary nodes of the specified type.
        nor         ndarray         A 2D array of normal vectors corresponding to the boundary nodes.
    '''
    # Filter nodes based on the specified node type
    msc = (p[:, 2] == node_type)                                                    # Get the index of the boundary nodes according to node type.
    pb = np.vstack([p[msc]])                                                        # Create an array with only the boundary nodes.
    nb = len(pb[:, 0])                                                              # Get the number of boundary nodes.
    nor = np.zeros([nb, 2])                                                         # Create an array for the normal vectors.
    q = p[nb, 0:2]                                                                  # Get the coordinates of an inner node.

    # To make sure the nodes are in counterclockwise order.
    a = 0                                                                           # 'a' initialization with zero.
    for i in range(nb):                                                             # Compute the value of 'a' based on node positions.
        z  = q - pb[i-1, 0:2]
        w  = q - pb[i, 0:2]
        a += z[0]*w[1] - z[1]*w[0]

    # Rotation Matrix.
    rota = np.array([[0, 1], [-1, 0]]) if a > 0 else np.array([[0, -1], [1, 0]])    # Determine the rotation matrix.

    # Vectores for adjacent boundary nodes.
    for i in range(nb - 1):
        v = np.roll(pb[:, 0:2], shift=-1, axis=0) - np.roll(pb[:, 0:2], shift=1, axis=0)
        nor[i, :] = np.dot(rota, v[i, :]) / np.linalg.norm(np.dot(rota, v[i, :]))   # Calculate vectors 'v' between adjacent boundary nodes.

    v = pb[0, 0:2] - pb[nb - 2, 0:2]
    nor[nb - 1, :] = np.dot(rota, v) / np.linalg.norm(np.dot(rota, v))

    return pb, nor

def GraphNormals(pb, vecs, nom, show, holes):
    """
    This function creates a scatter plot of the boundary and the normal vectors of a cloud of points where points are colored based on the
    position in the cloud:
    Blue markers are used for inner points, and red markers for boundary points.
    The plot is saved as PNG and EPS files with the specified name and displayed to the screen.

    Parameters:
        pb          ndarray         Array with the coordinates of the cloud of points with boundary node markings.
        vecs        ndarray         Array of normal vectors corresponding to the boundary nodes.
        nom         str             Name for the generated plot and file.
        show        bool            Flag to indicate whether the figure should be shown on screen.
        holes       bool            Flag to indicate whether holes are present.

    Returns:
        None
    """

    # Parameter initialization
    if holes:
        nomp  = 'Holes/Normals/' + nom + '.png'                                     # Define the file name for the saved plot.
        nome  = 'Holes/Normals/' + nom + '.eps'                                     # Define the file name for the saved plot.
    else:
        nomp  = 'Clouds/Normals/' + nom + '.png'                                    # Define the file name for the saved plot.
        nome  = 'Clouds/Normals/' + nom + '.eps'                                    # Define the file name for the saved plot.
    color = ['blue' if x == 0 else 'red' for x in pb[:,2]]                          # Determine marker color based on the third column.

    # Cloud plot
    plt.rcParams["figure.figsize"] = (16, 12)                                       # Configure plot size.
    plt.scatter(pb[:,0], pb[:,1], c = color, s = 20)                                # Create scatter plot.
    for i in np.arange(len(pb[:,0])):
        x = [pb[i,0], vecs[i,0]]
        y = [pb[i,1], vecs[i,1]]
        plt.plot(x, y, 'k')
        #plt.text(pb[i,0], pb[i,1], str(i), color='black')
    plt.title(nom + ' Normal Vectors')                                              # Set plot title.
    plt.axis('equal')
    if show:
        plt.show()                                                                  # Display the plot.
    plt.savefig(nomp)                                                               # Save the plot as a PNG file.
    plt.savefig(nome, format='eps', bbox_inches='tight')                            # Save the plot as a EPS file.
    plt.close()                                                                     # Close the plot to release resources.

def GraphMesh(x, y, me, nom):
    """
    This function creates a graphic of the selected mesh.
    The plot is saved as a PNG file with the specified name.

    Parameters:
        x           ndarray         Array of x-coordinates of grid points.
        y           ndarray         Array of y-coordinates of grid points.
        me          int             Number of grid nodes in the cloud.
                                    1: 21
                                    2: 41
                                    3: 81
        nom         str             Name for the generated plot and file.

    Returns:
        None
    """

    nomm = 'Meshes/' + me + '/' + nom + '.png'
    plt.rcParams["figure.figsize"] = (12,12)
    plt.plot(x, y, c = 'black')
    plt.plot(x.T, y.T, c = 'black')
    plt.title(nom + '_' + me + ' Mesh')
    plt.savefig(nomm)
    plt.close()