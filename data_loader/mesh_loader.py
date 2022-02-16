import glob

# BOP functions
import sys
sys.path.append('utils')
sys.path.append('utils/bop_toolkit')
from utils.bop_toolkit.bop_toolkit_lib.inout import load_ply, load_json, load_im
from utils import get_param
import numpy as np
from PIL import Image
# pytorch3d functions
import torch
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.renderer import TexturesVertex, Textures
from pytorch3d.ops import sample_points_from_meshes

### used to evaluate the ADD score and for ablation study (not in model)
class MeshesLoader():
    def __init__(self, data_dir, obj_list, N_pts=100, use_mesh=False, **kwargs):
        self.use_mesh = use_mesh
        self.N_pts = N_pts
        self.idx2radius = get_param(data_dir, 'idx2radius')
        self.device = torch.device("cuda:0")
        self.MESH_DIR = f'{data_dir}/models'
        self.obj_list = obj_list
        self.MESH_INFO = self.load_mesh_info()
        self.MESH_DICT, self.PTS_DICT, self.FULL_PTS_DICT = self.load_meshes()

    def load_meshes(self):
        mesh_adrs = sorted(glob.glob('%s/*.ply'%self.MESH_DIR))
        mesh_dict = {}
        pts_dict = {}
        full_pts_dict = {}
        for mesh in mesh_adrs:
            mesh_index = int(mesh.split('/')[-1].split('.')[0].split('_')[1])
            if mesh_index in self.obj_list:
                ply_model = load_ply(mesh)
                vertex = torch.tensor(ply_model['pts']).type(torch.float)
                vertex = vertex / self.idx2radius[mesh_index]
                faces = torch.tensor(ply_model['faces']).type(torch.float)
                if 'texture_file' in ply_model.keys():
                    verts_rgb = np.zeros([vertex.shape[0], 3])
                    texture_path = ply_model['texture_file']
                    texture_image = load_im(f'{self.MESH_DIR}/{texture_path}')
                    texture_image = np.array(Image.fromarray(texture_image).resize((1024, 1024), Image.BICUBIC))
                    texture_loc = np.concatenate([(ply_model['texture_uv'][:, [0]] * texture_image.shape[0]).astype(np.int),
                                                  (texture_image.shape[1] - ply_model['texture_uv'][:, [1]] * texture_image.shape[1]).astype(np.int)], 1)
                    for i, loc in enumerate(texture_loc):
                        verts_rgb[i] = texture_image[loc[1], loc[0]]
                    verts_rgb = torch.tensor(verts_rgb).type(torch.float)[None]
                    textures = Textures(verts_rgb=verts_rgb)
                else:
                    colors = torch.tensor(ply_model['colors']).type(torch.float)
                    textures = TexturesVertex(verts_features=[colors])
                mesh = Meshes(verts=[vertex],
                              faces=[faces],
                              textures=textures)
                if self.use_mesh:
                    mesh_dict[mesh_index] = mesh
                pts_dict[mesh_index] = sample_points_from_meshes(mesh, self.N_pts)[0].to(self.device)
                full_pts_dict[mesh_index] = mesh.verts_list()[0].to(self.device)
        return mesh_dict, pts_dict, full_pts_dict


    def load_mesh_info(self):
        mesh_info_adrs = '%s/models_info.json'%self.MESH_DIR
        mesh_info = load_json(mesh_info_adrs)
        mesh_info_dict = {}
        for key in list(mesh_info.keys()):
            mesh_index = int(key)
            if mesh_index in self.obj_list:
                mesh_info_dict[mesh_index] = mesh_info[key]
        return mesh_info_dict

    def batch_meshes(self, id_batch):
        mesh_list = [self.MESH_DICT[id] for id in id_batch.cpu().numpy()]
        mesh_batch = join_meshes_as_batch(mesh_list).to(self.device)
        return mesh_batch

    def batch_meshes_info(self, id_batch):
        mesh_info_batch = [self.MESH_INFO[id] for id in id_batch]
        return mesh_info_batch

    def batch_pts(self, id_batch):
        pts_batch = torch.stack([self.PTS_DICT[id] for id in id_batch])
        return pts_batch
