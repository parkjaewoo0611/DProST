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
from pytorch3d.renderer import TexturesVertex
from pytorch3d.ops import sample_points_from_meshes
from utils.LM_parameter import LM_idx2radius


class MeshesLoader():
    def __init__(self, mesh_dir, obj_list, render_size, N_pts=100):
        self.device = torch.device("cuda:0")
        self.MESH_DIR = mesh_dir
        self.obj_list = obj_list
        self.MESH_DICT, self.TEXTURE_LIST = self.load_meshes()
        self.MESH_INFO = self.load_mesh_info()
        self.PTS_DICT = self.sample_pts(N_pts)
        self.FULL_PTS_DICT = self.full_pts()
        self.on_device()


    def load_meshes(self):
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
        mesh_info_adrs = '%s/models_info.json'%self.MESH_DIR
        mesh_info = load_json(mesh_info_adrs)
        mesh_info_dict = {}
        for key in list(mesh_info.keys()):
            mesh_index = int(key)
            if mesh_index in self.obj_list:
                mesh_info_dict[mesh_index] = mesh_info[key]
        return mesh_info_dict


    def sample_pts(self, N_pts):
        points_dict = {}
        for obj_id in self.MESH_DICT.keys():
            points_dict[obj_id] = sample_points_from_meshes(self.MESH_DICT[obj_id], N_pts)[0].to(self.device)
        return points_dict
    
    def full_pts(self):
        points_dict = {}
        for obj_id in self.MESH_DICT.keys():
            points_dict[obj_id] = self.MESH_DICT[obj_id].verts_list()[0].to(self.device)
        return points_dict

    def on_device(self):
        for obj_id in self.MESH_DICT.keys():
            self.MESH_DICT[obj_id] = self.MESH_DICT[obj_id].to(self.device)
        for obj_id in self.PTS_DICT.keys():
            self.PTS_DICT[obj_id] = self.PTS_DICT[obj_id].to(self.device)

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
