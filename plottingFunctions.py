# scripts for plotting multi-channel dice
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from skimage import measure
import numpy as np
import os

organ_dict = {1: "left kidney", 2: "right kidney", 3: "liver", 4: "pancreas"}


# Plot a 3D mesh from a binary  3D label alongside the ground truth
def plot3Dmesh(gt, volumes, save_path, subject):
    fig = go.Figure()

    # lighting settings for PlotLy objects
    lighting = dict(ambient=0.5, diffuse=0.5, roughness=0.5, specular=0.6, fresnel=0.8)

    # cycle over the organs and plot layers
    values = np.unique(gt)
    for k in list(values):
        if k > 0:
            gt_k = np.zeros(gt.shape)
            gt_k[gt == k] = 1

            try:
                gt_verts, gt_faces, gt_normals, gt_values = measure.marching_cubes(gt_k, 0)
                gt_x, gt_y, gt_z = gt_verts.T
                gt_I, gt_J, gt_K = gt_faces.T
                gt_mesh = go.Mesh3d(x=gt_x, y=gt_y, z=gt_z,
                                    intensity=gt_values,
                                    i=gt_I, j=gt_J, k=gt_K,
                                    lighting=lighting,
                                    name=organ_dict[k],
                                    showscale=False,
                                    opacity=1.0,
                                    colorscale='magma'
                                    )
                fig.add_trace(gt_mesh)
            except:
                print("GT mesh extraction failed")

            fig.update_layout(title_text="{0} volume: {1:.0f} ml".format(organ_dict[k], volumes[k] / 1000))
            fig.update_xaxes(visible=False, showticklabels=False)
            fig.write_image(os.path.join(save_path, "{}_{}.png".format(organ_dict[k], subject)))


