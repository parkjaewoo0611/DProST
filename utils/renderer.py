import cv2
import torch
import torch.nn.functional as F

# rendering components
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, PerspectiveCameras
)
from lib3d.cropping import deepim_crops_robust as deepim_crops

from util import to_img, get_K_crop_resize, TCO_to_RT
import numpy as np
import cv2


class Renderer():
    def __init__(self, render_size):
        self.device = torch.device("cuda:0")
        self.H_crop, self.W_crop = render_size

        # pytorch3d는 world coordinate의 x가 -x y가 -y
        # pytorch는 XR + t
        self.R_comp = torch.eye(3, 3)[None].to(self.device)
        self.R_comp[0, 0, 0] = -1
        self.R_comp[0, 1, 1] = -1



        ### Phong shader
        self.raster_settings = RasterizationSettings(
                image_size=self.H_crop,
                blur_radius=0.0,
                faces_per_pixel=1)



    def camera_update(self, K, R, T):
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        tx = K[:, 0, 2]
        ty = K[:, 1, 2]
        f = torch.stack([fx, fy]).permute(1, 0)
        p = torch.stack([tx, ty]).permute(1, 0)
        size = torch.tensor([[self.W_crop, self.H_crop]]).repeat(f.shape[0], 1)

        self.cameras = PerspectiveCameras(focal_length=f,
                                          principal_point=p,
                                          image_size = size,
                                          R=R,
                                          T=T,
                                          device=self.device)
        self.phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings
            ),
            shader=HardPhongShader(
                device=self.device, 
                cameras=self.cameras,
            )
        )


    def change_RT(self, R, T):
        R_comp =  self.R_comp.repeat(R.shape[0], 1, 1)
        new_R = torch.bmm(R_comp, R).transpose(1, 2)
        ################### T = -RC#########new_T = Rt @ T @ new_R = ###########3
        new_T = torch.bmm(torch.bmm(R.transpose(1, 2), T.view(-1, 3, 1)).transpose(1, 2), new_R).transpose(1, 2).squeeze(2)
        return new_R, new_T


    def meshes_visualize(self, K, TCO, mesh):
        R, T = TCO_to_RT(TCO)
        R, T = self.change_RT(R, T)
        self.camera_update(K, R, T)

        phong = self.phong_renderer(meshes_world=mesh)
        phong = (((phong  - phong.min()) / phong.max()) - 0.5) * 2

        phong = phong.permute(0, 3, 1, 2)[:, :3, ...]
        phong = F.interpolate(phong, (self.H_crop, self.W_crop), mode='bilinear', align_corners=True)

        return phong


    def phong_to_silho(self, phong):        
        silhouete = 1-torch.sigmoid((phong - 0.99) * 1000)
        return silhouete


    def crop_inputs(self, images, boxes_input, K, TCO_input, points):
        boxes_crop, images_cropped = deepim_crops(images=images,
                                                  obs_boxes=boxes_input,
                                                  K=K,
                                                  TCO_pred=TCO_input,
                                                  O_vertices=points,
                                                  output_size=(self.H_crop, self.W_crop),
                                                  lamb=1.4)
        K_crop = get_K_crop_resize(K=K.clone(),
                                   boxes=boxes_crop,
                                   orig_size=images.shape[-2:],
                                   crop_resize=(self.H_crop, self.W_crop))
        return images_cropped, K_crop.detach(), boxes_crop


    def contour_visualize(self, render, img, color=(0, 255, 0)):
        rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(render, cv2.COLOR_RGB2GRAY)
        _, thr_image = cv2.threshold(gray, 1, 255, 0)
        contours, hierarchy = cv2.findContours(thr_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        result = cv2.drawContours(rgb, contours, -1, color, 3)
        return result


    def result_visualize(self, init_phong, output_phong_list, label_phong, image):
        ####### visualize renderings ###############
        image = to_img(image)

        init_phong = to_img(init_phong)
        initial_contour = self.contour_visualize(init_phong, image, (0, 0, 255))

        render_list = [init_phong]
        contour_list = [initial_contour]

        for output_phong in output_phong_list:
            render = to_img(output_phong)
            contour = self.contour_visualize(render, image, (255, 0, 0))
            render_list.append(render)
            contour_list.append(contour)


        label_render = to_img(label_phong)
        label_contour = self.contour_visualize(label_render, image, (0, 255, 0))
        render_list.append(label_render)
        contour_list.append(label_contour)


        render_list = np.concatenate(tuple(render_list), 1)
        contour_list = np.concatenate(tuple(contour_list), 1)
        result = np.concatenate(tuple([render_list, contour_list]), 0)
        return result
        

    def bbox_visualize(self, image, bbox_list):
        image = image.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
        board = [image]
        box_color = [(1, 0, 0),
                     (0, 1, 0),
                     (0, 0, 1)]
        for i, bbox in enumerate(bbox_list):
            bbox = bbox.detach().cpu().numpy()[0]
            bbox_img = cv2.rectangle(image.copy(), (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color[i])
            board.append(bbox_img)
        board = np.concatenate(tuple(board), 1)
        board = (board + 1) / 2
        return board
        