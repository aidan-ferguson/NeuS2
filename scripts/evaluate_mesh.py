# Quantify how accurate an NeuS2 generated mesh is

import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
import json
import cv2
from tqdm import tqdm
# from image_similarity_measures.quality_metrics import rmse

# Adapted from image_similarity_measures package
def get_rmse(org_img: np.ndarray, pred_img: np.ndarray, max_p: int = 4095) -> float:
    """
    Root Mean Squared Error

    Calculated individually for all bands, then averaged
    Note rows and columns must be flattened into one dimesnsion
    """
    assert(pred_img.shape == org_img.shape)
    assert(len(org_img.shape) == 2)
    org_img = org_img.astype(np.float32)
    
    rmse_bands = []
    diff = org_img - pred_img
    mse_bands = np.mean(np.square(diff / max_p), axis=0)
    rmse_bands = np.sqrt(mse_bands)
    return np.mean(rmse_bands)

def main():
    parser = argparse.ArgumentParser(description='Evaluate how accurate a NeuS2 generated mesh is')
    parser.add_argument('mesh', type=str, help='Path to the .obj input mesh file')
    parser.add_argument('transforms', type=str, help='Path to the transforms.json used to generate the NeRF and mesh')
    parser.add_argument('--save_render', action='store_true', help='Save the rendered images')
    parser.add_argument('--debug', action='store_true', help='Enable the interactive GUI and show debug elements')
    args = parser.parse_args()
    save_render = args.save_render

    # TODO: input validation

    transforms_path = Path(args.transforms)
    eval_path = transforms_path.parent / "evaluation"

    if args.debug:
        print("Running in debug mode... Will take a minute")
    else:
        eval_path.mkdir(exist_ok=True)
        print(f"Saving evaluation to '{eval_path}'")

    # Access transforms and store default camera options
    with open(transforms_path, "r") as file:
        transforms = json.loads(file.read())
    frames = transforms["frames"]

    # TODO: ACCOUNT FOR DISTORTION BY UNDISTORING GT IMAGES
    d_w = transforms.get("w", None)
    d_h = transforms.get("h", None)
    d_fl_x = transforms.get("fl_x", None)
    d_fl_y = transforms.get("fl_y", None)
    d_fl_y = transforms.get("fl_y", None)
    d_cx = transforms.get("cx", None)
    d_cy = transforms.get("cy", None)

    # Create an Open3D visualizer for onscreen rendering
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Mesh Viewer", 
                      width=int(d_w) if d_w is not None else 1920, 
                      height=int(d_h) if d_h is not None else 1440, 
                      visible=args.debug)

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
        if args.debug:
            vis.add_geometry(cam_axes)
    
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    vis.add_geometry(mesh)
    
    if args.debug:
        origin_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8) 
        vis.add_geometry(origin_axes)

    # No back face culling
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True

    ctr = vis.get_view_control()
    ctr.set_constant_z_far(10000.0)

    # Render GUI in debug mode
    if args.debug:
        while True:
            vis.poll_events()
            vis.update_renderer()

    # Loop through each camera pose, taking the image of the current frame and evaluate mesh quality
    rmse_history = []
    for idx, cam in enumerate(tqdm(gt_cameras)):
        # Set up camera parameters, attempting to get per frame parameters if they exist
        camera_parameters = o3d.camera.PinholeCameraParameters()
        camera_parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(cam.get("w", d_w)), int(cam.get("h", d_h)),  # Image width, height
            cam.get("fl_x", d_fl_x),  # fx (focal length in x)
            cam.get("fl_y", d_fl_y),  # fy (focal length in y)
            cam.get("cx", d_cx),  # cx (principal point x)
            cam.get("cy", d_cy)   # cy (principal point y)
        )

        # Our extrinsic is currently in the wrong coordinate system (+Z points away from object), flip
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
            
        _rmse = get_rmse(gt_mask, render_mask)
        rmse_history.append(_rmse)

    print("Evaluation Summary:")
    print("- RMSE \u03BC: ", np.mean(rmse_history))
    print("- RMSE \u03C3: ", np.std(rmse_history))

    vis.close()


if __name__ == "__main__":
    args = main()