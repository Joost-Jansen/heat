import os
import numpy as np
import cv2
from shapely import geometry
from shapely.geometry import Polygon
from graph_utils import get_regions_from_pg, plot_floorplan_with_regions

# house_label = {
#     (127, 20): [(20, 120), (234, 120)],
#     (20, 120): [(127, 20), (234, 120), (20, 240)],
#     (234, 120): [(127, 20), (20, 120), (234, 240)],
#     (20, 240): [(20, 120), (234, 240)],
#     (234, 240): [(234, 120), (20, 240)],
# }

# function that returns the corners of a polygon
def get_corners(polys):
    
    # Init corners
    corners = {}

    # Loop over polys
    for i in range(len(polys)):

        # Create map of coordinates
        poly_mapped = geometry.mapping(polys[i])

        # Loop over all coordinates in poly
        for i in range(len(poly_mapped['coordinates'][0])):
            # Get coordinates
            x, y = poly_mapped['coordinates'][0][i]

            # Check if dict exists
            if (x,y) not in corners:
                corners[(x,y)] = []
            
            # Loop over coordinates in the poly that might be connected
            for j in range(len(poly_mapped['coordinates'][0])):
                # If same coordinates skip
                if i == j: continue

                # Get coords
                x2, y2 = poly_mapped['coordinates'][0][j]

                # Check if either x or y is equal
                if x == x2 or y == y2:
                    corners[(x,y)].append((x2, y2))

    # return corners
    return corners

# function that plots the corners obtainbed by get_corners
def plot_corners(corners, save_path, filename):
    # set background to transparent of image
    img = np.zeros((300,300, 3), np.uint8)

    # draw edges between corners
    for i in corners:
        for j in corners[i]:
            cv2.line(img, (int(i[0]), int(i[1])), (int(j[0]), int(j[1])), (255, 0, 0), 3)

    # draw corners
    for i in corners:
        cv2.circle(img, (int(i[0]), int(i[1])), 3, (0, 0, 255), -1)

    # Save files
    cv2.imwrite(os.path.join(save_path, '{}.png'.format(filename)), img)

##################### EDIT THESE!! ####################
data_path = ',/geometry'
save_path = './labels'
vis_path = './vis'

# Set to True to create visualisation of the labels
visualise = False

# Loop over files in geometry data
for filename in sorted(os.listdir(data_path)):
    # Get path of file
    pg_path = os.path.join(data_path, filename)

    # Load data
    example_pg = np.load(pg_path, allow_pickle=True)

    # Get rooms
    room_polygons, door_polygons, wall_polygons = example_pg[0]
    room_types, door_types, wall_types = example_pg[1]

    # Translate corners to correct format
    corners = get_corners(room_polygons)

    # Print current file
    print("Saving file: {}".format(filename[:-7]))

    # Check if want to visualize or save label structure
    if visualise:
        plot_corners(corners, vis_path, filename[:-7] + '_plotted')
    else:
        np.save(os.path.join(save_path, '{}.npy'.format(filename[:-7])), corners)

