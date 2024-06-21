from scipy.io import loadmat
from scipy.io import savemat
import os
import glob
import CloudGen

# Variable Initialization.
sizes = [1, 2, 3]
first = True

# Should we add holes?
holes = True

for me in sizes:
    mesh = str(me)
    if first:
        # Find all the regions.
        regions = glob.glob(f'Meshes/2/*.mat')
        regions = sorted([os.path.splitext(os.path.basename(region))[0] for region in regions])
        first = False
    
    for reg in regions:
        # Load data from the file.
        mat  = loadmat('Meshes/2/' + reg + '.mat')
        print('Trabajando en la malla ' + reg + '_' + mesh + '.')

        # Get the grid nodes
        x, y  = mat['x'], mat['y']

        # Generate the Cloud
        if holes:
            scale = max(x.max(),y.max())
            x     = (x - x.min())/scale
            y     = (y - y.min())/scale
            if reg == 'ENG':
                h_coords = [(0.3, 0.3, 0.05)]
            elif reg == 'PAT':
                h_coords = [(0.25, 0.32, 0.05)]
            else:
                h_coords = [(x.max()/2, y.max()/2, 0.05)]
            p, tt = CloudGen.GridToCloud(x,y, holes, (2/3)*me, h_coor = h_coords)
        else:
            p, tt = CloudGen.GridToCloud(x,y, holes, (2/3)*me)

        # Get the name to save the information.
        if holes:
            folder = 'Holes/' + mesh + '/'                                      # Define the directory.
        else:
            folder = 'Clouds/' + mesh + '/'                                     # Define the directory.
        
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Generate the Graph of the Cloud
        nom = folder + reg
        CloudGen.GraphCloud(p, save = True, nom = nom, show = False)

        # Save the information of the Cloud.
        nom = nom + '.mat'
        mdic = {"p": p, "tt": tt}
        savemat(nom, mdic)