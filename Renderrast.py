import os
import time
import argparse
import json

import numpy as np
import torch
import nvdiffrast.torch as dr
import xatlas
import imageio

# Import data readers / generators
from dataset.dataset_mesh import DatasetMesh
from dataset.dataset_nerf import DatasetNERF
from dataset.dataset_llff import DatasetLLFF

# Import topology / geometry trainers
from geometry.dmtet import DMTetGeometry
from geometry.dlmesh import DLMesh
from geometry.flexicubes_geo import FlexiCubesGeometry

import render.renderutils as ru
from render import obj
from render import material
from render import util
from render import mesh
from render import texture
from render import mlptexture
from render import light
from render import render



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relmse'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('-rm', '--ref_mesh', type=str)
    parser.add_argument('-bm', '--base-mesh', type=str, default=None)
    parser.add_argument('--validate', type=bool, default=True)
    parser.add_argument('--isosurface', default='dmtet', choices=['dmtet', 'flexicubes'])
    
    FLAGS = parser.parse_args()

    FLAGS.mtl_override        = None                     # Override material of model
    FLAGS.dmtet_grid          = 64                       # Resolution of initial tet grid. We provide 64 and 128 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
    FLAGS.mesh_scale          = 2.1                      # Scale of tet grid box. Adjust to cover the model
    FLAGS.env_scale           = 1.0                      # Env map intensity multiplier
    FLAGS.envmap              = "data/irrmaps/aerodynamics_workshop_2k.hdr"                   # HDR environment probe
    FLAGS.display             = None                     # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
    FLAGS.camera_space_light  = False                    # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
    FLAGS.lock_light          = False                    # Disable light optimization in the second pass
    FLAGS.lock_pos            = False                    # Disable vertex position optimization in the second pass
    FLAGS.sdf_regularizer     = 0.2                      # Weight for sdf regularizer (see paper for details)
    FLAGS.laplace             = "relative"               # Mesh Laplacian ["absolute", "relative"]
    FLAGS.laplace_scale       = 10000.0                  # Weight for Laplacian regularizer. Default is relative with large weight
    FLAGS.pre_load            = True                     # Pre-load entire dataset into memory for faster training
    FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    FLAGS.ks_max              = [ 1.0,  1.0,  1.0]
    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.cam_near_far        = [0.1, 1000.0]
    FLAGS.learn_light         = True

    FLAGS.local_rank = 0
    FLAGS.multi_gpu  = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    # Environment map 
    path_light = "data/irrmaps/aerodynamics_workshop_2k.hdr"
    lgt = light.load_env(path_light)

    # Geometry
    # v:the coordinate of vertex; vn:the normal of vertex; vt:the coordinate of texture; 
    # f:a face consist of three points
    path_geometry = './data/spot/spot.obj'
    base_mesh = mesh.load_mesh(path_geometry)
    imesh = mesh.compute_tangents(base_mesh)  # shape 
    geometry = DLMesh(base_mesh, FLAGS)
    # mat = initial_guess_material(geometry, False, FLAGS, init_mat=base_mesh.material)
    # Material 
    path_mat = './data/spot/spot.mtl'
    material_ = material.load_mtl(path_mat)
    
    # Render
    RADIUS = 3.0
    glctx = dr.RasterizeGLContext()
    dataset_train  = DatasetMesh(base_mesh, glctx, RADIUS, FLAGS, validate=False)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=FLAGS.batch, collate_fn=dataset_train.collate, shuffle=True)
    for it, target in enumerate(dataloader_train):

        # Mix randomized background into dataset image
        
        buffer = render.render_mesh(glctx, imesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
                                    num_layers=FLAGS.layers, msaa=True, background=None, bsdf=None)
        image = buffer['shaded'].squeeze(0)

        ref_image = util.rgb_to_srgb(target['img'][...,0:3][0])
        image = util.rgb_to_srgb(image)
        # ref_image = ref_image.squeeze(0)
        ref_image = ref_image.cpu().detach().numpy()
        image = image.cpu().detach().numpy()
        imageio.imwrite("ref_image.png", np.clip(np.rint(ref_image * 255), 0, 255).astype(np.uint8))
        # imageio.imwrite("ref_image.png", (ref_image * 255.0).astype(np.uint8))
        # 保存图像到文件
        imageio.imwrite('tensor_image.png', np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8))

        # 读取并显示保存的图像
        print(buffer)
        break