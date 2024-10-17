# Quantify how accurate an NeuS2 generated mesh is

import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
import json
import cv2
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Evaluate how accurate a NeuS2 generated mesh is')
    parser.add_argument('mesh', type=str, help='Path to the .obj input mesh file')
    parser.add_argument('transforms', type=str, help='Path to the transforms.json used to generate the NeRF and mesh')
    args = parser.parse_args()

    # TODO: make command line options
    render_debug = False
    save_render = True 

    # TODO: checking if stuff exists

    transforms_path = Path(args.transforms)
    eval_path = transforms_path.parent / "evaluation"

    eval_path.mkdir(exist_ok=True)
    print(f"Saving evaluation to '{eval_path}'")

    with open(transforms_path, "r") as file:
        data = json.loads(file.read())
    frames = data["frames"]

    # Create an Open3D visualizer for onscreen rendering
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Mesh Viewer", width=1920, height=1440, visible=False)

    # Rotate into the correct coordiante system around the axis of rotation
    theta_x = -np.pi/2
    theta_z = -np.pi/2
    rotation_x = np.array([
        [np.cos(theta_x), 0, np.sin(theta_x), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_x), 0, np.cos(theta_x), 0],
        [0,0,0,1]
    ])
    rotation_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0, 0],
        [np.sin(theta_z), np.cos(theta_z), 0, 0],
        [0, 0, 1, 0],
        [0,0,0,1]
    ])


    gt_cameras = []
    for frame in frames:
        cam_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        t = np.array(frame["transform_matrix"])    
        t = rotation_z @ (rotation_x @ t)
        frame["transform_matrix"] = t

        gt_cameras.append(frame)

        cam_axes.transform(t)
        if render_debug:
            vis.add_geometry(cam_axes)
    
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    vis.add_geometry(mesh)
    
    if render_debug:
        origin_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8) 
        vis.add_geometry(origin_axes)


    # Set up intrinsic camera parameters
    cam_K = np.array([1586.3734, 0.0, 960.0, 0.0, 1586.3734, 720.0, 0.0, 0.0, 1.0]).reshape((3,3))
    camera_parameters = o3d.camera.PinholeCameraParameters()
    camera_parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        1920, 1440,  # Image width, height
        cam_K[0, 0],  # fx (focal length in x)
        cam_K[1, 1],  # fy (focal length in y)
        cam_K[0, 2],  # cx (principal point x)
        cam_K[1, 2]   # cy (principal point y)
    )

    opt = vis.get_render_option()
    opt.mesh_show_back_face = True

    ctr = vis.get_view_control()
    ctr.set_constant_z_far(10000.0)

    # Loop through each camera pose, taking the image of the current frame and evaluate mesh quality
    mae_history = []
    for idx, cam in enumerate(tqdm(gt_cameras)):
        # Our pose is currently in the wrong coordinate system (+Z points away from object), flip
        flip_axes = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        extrinsic = cam["transform_matrix"]
        extrinsic[:3, :3] = flip_axes[:3, :3] @ extrinsic[:3, :3]
        camera_parameters.extrinsic = np.linalg.inv(extrinsic) # extrinsic = transform_mat^{-1}
        ctr.convert_from_pinhole_camera_parameters(camera_parameters, True)

        # Render GUI
        # vis.poll_events()
        # vis.update_renderer()

        # Render mesh from viewpoint of camera
        render = np.asarray(vis.capture_screen_float_buffer(True))
        render = cv2.cvtColor(np.array(render * 255, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        if save_render:
            cv2.imwrite(eval_path / f"{idx}_render_raw.png", render)

        # Load GT image, note we assume the image has already been pre-masked (using the generation tool)
        # Will error if there are not 4 channels detected OR the alpha channel is opaque
        gt_img = cv2.imread(transforms_path.parent / Path(cam["file_path"]), cv2.IMREAD_UNCHANGED)
        assert(gt_img.shape[2] >= 4)
        assert(np.any(gt_img[:, :, 3] < 255))
        assert(gt_img.shape[0] == render.shape[0])
        assert(gt_img.shape[1] == render.shape[1])

        # Perform MAE on the render and the GT image, only considering non-masked pixels
        mask = gt_img[:, :, 3] > 0
        gt_mask = gt_img[mask][:, :3]
        render_mask = render[mask][:, :3]

        if save_render:
            full_render_mask = np.array(gt_img)
            full_render_mask[:, :, :3] = render

            diff_mask = np.array(gt_img)
            diff = np.mean(np.abs(gt_img[:, :, :3] - render[:, :, :3]), axis=2)
            diff_mask[:, :, 0] = diff
            diff_mask[:, :, 1] = diff
            diff_mask[:, :, 2] = diff

            cv2.imwrite(eval_path / f"{idx}_gt_mask.png", gt_img)
            cv2.imwrite(eval_path / f"{idx}_render_mask.png", full_render_mask)
            cv2.imwrite(eval_path / f"{idx}_diff.png", diff_mask)
            
        mae = np.mean(np.abs(render_mask - gt_mask))
        mae_history.append(mae)

    print("MAE mean: ", np.mean(mae_history))
    print("MAE std. dev: ", np.std(mae_history))

    vis.close()


if __name__ == "__main__":
    args = main()