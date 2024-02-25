import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse
import json

from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics


def save_reconstruction(droid, reconstruction_path):

    from pathlib import Path
    import random
    import string

    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    # parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    parser.add_argument("--scene_name", type=str, help="name of the scene as a string")
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')
    reconstruction_path = "./outputs_"+args.scene_name
    droid = None

    # need high resolution depths
    if reconstruction_path is not None:
        args.upsample = True

    tstamps = []
    for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.calib, args.stride)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        droid.track(t, image, intrinsics=intrinsics)
        # print(f"intrinsic: {intrinsics}")
        print(f"image shape: height {image.shape[2]}, width {image.shape[3]}")

    if reconstruction_path is not None:
        save_reconstruction(droid, reconstruction_path)

    traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))
    # T_edit
    T_edit = np.diag(np.array([1, -1, -1, 1.], dtype=traj_est.dtype))
    # post multiply the estimated trajectory by T_edit
    traj_edit = traj_est @ T_edit
    print(f"traj_est shape: {traj_est.shape}")
    print(f"traj_Est: {traj_est[:5]}")
    droid.saving_pcd(scene_name=args.scene_name)
    np.save("reconstructions/{}/traj_est.npy".format(reconstruction_path), traj_est)
    print("saved reconstruction data to: reconstructions/outputs_{}".format(args.scene_name))
    print(f" traject {traj_est.shape}")

    # read intrinsics from the calibration file
    intrinsics = np.loadtxt(args.calib, delimiter=" ")
    print(f" intrinsics: {intrinsics}")

    #let us save the data to a json file
    # intrinsics = intrinsics.cpu().numpy()
    data = {}
    data["fl_x"] = intrinsics[0]
    data["fl_y"] = intrinsics[1]
    data["k1"] = 0.0
    data["k2"] = 0.0
    data["p1"] = 0.0
    data["p2"] = 0.0
    data["cx"] = intrinsics[2]
    data["cy"] = intrinsics[3]
    # read the image size from the first image
    sample_image = cv2.imread(os.path.join(args.imagedir, os.listdir(args.imagedir)[0]))
    data["w"] = sample_image.shape[1]
    data["h"] = sample_image.shape[0]

    # data["aabb_scle"] = None
    # now add the frame data
    frame_data = []
    image_list = sorted(os.listdir(args.imagedir))[::args.stride]
    for i in range(traj_est.shape[0]):
        frame_data.append({"file_path": "./images/" + image_list[i], "transform_matrix":traj_edit[i].tolist()})
    data["frames"] = frame_data
    # data['applied_transform'] = [
    #     [0, 1, 0, 0],
    #     [1, 0, 0, 0],
    #     [0, 0, -1, 0]]
    data['ply_file_path'] = "points.ply".format(reconstruction_path)

    # with open("reconstructions/{}/transforms.json".format(reconstruction_path), "w") as f:
    #     json.dump(data, f)
    results_dir = os.path.dirname(args.imagedir)
    with open(results_dir+"/transforms.json".format(reconstruction_path), "w") as f:
        json.dump(data, f)
    print("saved data to json file")