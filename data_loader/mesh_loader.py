import numpy as np
import glob

# BOP functions
import sys
sys.path.append('utils')
sys.path.append('utils/bop_toolkit')
from utils.bop_toolkit.bop_toolkit_lib.inout import load_ply, load_json

# pytorch3d functions
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    OrthographicCameras, RasterizationSettings, MeshRenderer,
    MeshRasterizer, HardPhongShader, TexturesVertex,
    get_world_to_view_transform, PerspectiveCameras
)
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.transforms import euler_angles_to_matrix

from utils.util import LM_idx2radius


class MeshesLoader():
    def __init__(self, mesh_dir, obj_list, render_size):
        self.device = torch.device("cuda:0")
        self.MESH_DIR = mesh_dir
        self.obj_list = obj_list
        self.MESH_DICT, self.TEXTURE_LIST = self.load_meshes()
        self.MESH_INFO = self.load_mesh_info()
        self.PTS_DICT = self.sample_pts()
        self.on_device()
        # self.RENDER_DICT = self.render_default(render_size)


    def load_meshes(self):
        print('loading mesh....')
        mesh_adrs = sorted(glob.glob('%s/*.ply'%self.MESH_DIR))
        mesh_dict = {}
        texture_dict = {}
        for mesh in mesh_adrs:
            mesh_index = int(mesh.split('/')[-1].split('.')[0].split('_')[1])
            if mesh_index in self.obj_list:
                ply_model = load_ply(mesh)
                vertex = torch.tensor(ply_model['pts']).type(torch.float).to(self.device)
                vertex = vertex / LM_idx2radius[mesh_index]
                faces = torch.tensor(ply_model['faces']).type(torch.float).to(self.device)
                colors = torch.tensor(ply_model['colors']).type(torch.float).to(self.device)
                textures = TexturesVertex(verts_features=[colors])
                texture_dict[mesh_index] = colors
                mesh = Meshes(verts=[vertex],
                            faces=[faces],
                            textures=textures)
                mesh_dict[mesh_index] = mesh.to(self.device)
        return mesh_dict, texture_dict


    def load_mesh_info(self):
        print('loading mesh info....')
        mesh_info_adrs = '%s/models_info.json'%self.MESH_DIR
        mesh_info = load_json(mesh_info_adrs)
        mesh_info_dict = {}
        for key in list(mesh_info.keys()):
            mesh_index = int(key)
            if mesh_index in self.obj_list:
                mesh_info_dict[mesh_index] = mesh_info[key]
        return mesh_info_dict


    def sample_pts(self):
        print('sampling points....')
        points_dict = {}
        for obj_id in self.MESH_DICT.keys():
            points_dict[obj_id] = sample_points_from_meshes(self.MESH_DICT[obj_id], 3000)[0].to(self.device)
        return points_dict
    
    
    def on_device(self):
        for obj_id in self.MESH_DICT.keys():
            self.MESH_DICT[obj_id] = self.MESH_DICT[obj_id].to(self.device)
        for obj_id in self.PTS_DICT.keys():
            self.PTS_DICT[obj_id] = self.PTS_DICT[obj_id].to(self.device)


    # def render_default(self, size):
    #     ### for Orthographic Pooling ###
    #     R_f = euler_angles_to_matrix(torch.tensor([-np.pi/2, 0, 0]), 'XYZ').unsqueeze(0).to(self.device)
    #     R_t = torch.eye(3).unsqueeze(0).to(self.device)
    #     R_r = euler_angles_to_matrix(torch.tensor([0, np.pi/2, 0]), 'XYZ').unsqueeze(0).to(self.device)


    #     t_i = torch.tensor(np.array([0, 0, 1.0])).float().unsqueeze(0).to(self.device)

    #     cameras_o_f = OrthographicCameras(
    #         image_size = size,
    #         R=R_f,
    #         T=t_i,
    #         in_ndc=True,
    #         device=self.device)

    #     cameras_o_t = OrthographicCameras(
    #         image_size = size, 
    #         R=R_t, 
    #         T=t_i, 
    #         in_ndc=True, 
    #         device=self.device)

    #     cameras_o_r = OrthographicCameras(
    #         image_size = size, 
    #         R=R_r, 
    #         T=t_i, 
    #         in_ndc=True, 
    #         device=self.device)

    #     raster_settings = RasterizationSettings(
    #         image_size=size,
    #         blur_radius=5e-5,
    #         faces_per_pixel=1)

    #     shader_f = HardPhongShader(device=self.device, cameras=cameras_o_f)
    #     shader_t = HardPhongShader(device=self.device, cameras=cameras_o_t)
    #     shader_r = HardPhongShader(device=self.device, cameras=cameras_o_r)

    #     rasterizer_f = MeshRasterizer(cameras=cameras_o_f, raster_settings=raster_settings)
    #     rasterizer_t = MeshRasterizer(cameras=cameras_o_t, raster_settings=raster_settings)
    #     rasterizer_r = MeshRasterizer(cameras=cameras_o_r, raster_settings=raster_settings)

    #     phong_renderer_f = MeshRenderer(rasterizer=rasterizer_f, shader=shader_f)
    #     phong_renderer_t = MeshRenderer(rasterizer=rasterizer_t, shader=shader_t)
    #     phong_renderer_r = MeshRenderer(rasterizer=rasterizer_r, shader=shader_r)

    #     render_dict = {}
    #     for obj_id in self.MESH_DICT.keys():
    #         render_dict[obj_id] = {
    #                                 'front' : phong_renderer_f(self.MESH_DICT[obj_id]),
    #                                 'top' : phong_renderer_t(self.MESH_DICT[obj_id]),
    #                                 'right' : phong_renderer_r(self.MESH_DICT[obj_id])
    #                               }
    #     return render_dict


    def batch_meshes(self, id_batch):
        id_batch = id_batch.cpu().numpy()
        mesh_list = [self.MESH_DICT[id] for id in id_batch]
        verts_list = [mesh.verts_list()[0] for mesh in mesh_list]
        faces_list = [mesh.faces_list()[0] for mesh in mesh_list]
        texture_list = [self.TEXTURE_LIST[id] for id in id_batch]
        texture_list = TexturesVertex(verts_features=texture_list)
        mesh_batch = Meshes(verts=verts_list,
                            faces=faces_list,
                            textures=texture_list)
        return mesh_batch


    def batch_meshes_info(self, id_batch):
        mesh_info_batch = [self.MESH_INFO[id] for id in id_batch]
        return mesh_info_batch


    def batch_pts(self, id_batch):
        pts_batch = torch.stack([self.PTS_DICT[id] for id in id_batch])
        return pts_batch


    # def batch_render(self, id_batch):
    #     id_batch = id_batch.cpu().numpy()
    #     front = torch.cat([self.RENDER_DICT[id]['front'] for id in id_batch], 0)
    #     top = torch.cat([self.RENDER_DICT[id]['top'] for id in id_batch], 0)
    #     right = torch.cat([self.RENDER_DICT[id]['right'] for id in id_batch], 0)

    #     return front, top, right
