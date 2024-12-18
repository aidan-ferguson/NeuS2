import os.path
import argparse
from common import *
import json
import cv2
import pyngp as ngp # noqa
from PIL import Image
def depth_est(scene_dir, depth_dir="depth", sigma_thrsh=15, snapshot_file="base.ingp"):
	depth_dir_abs = os.path.join(scene_dir, depth_dir)

	if not os.path.exists(depth_dir_abs):
		os.mkdir(depth_dir_abs)

	transforms_file = os.path.join(scene_dir, "transforms.json")

	poses = []
	img_names = []

	with open(transforms_file, 'r') as tf:
		meta = json.load(tf)
		for frame in meta['frames']:
			poses.append(frame['transform_matrix'])
			img_names.append(os.path.basename(frame['file_path']))

	width = int(meta['w'])
	height = int(meta['h'])
	camera_angle_x = meta['camera_angle_x']
	camera_angle_y = meta['camera_angle_y']

	# mode = ngp.TestbedMode.Depth
	mode = ngp.TestbedMode.Nerf
	configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
	testbed = ngp.Testbed(mode)
	testbed.nerf.sharpen = float(0)
	testbed.shall_train = False

	# Load a trained NeRF model
	print("Loading snapshot ", snapshot_file )
	testbed.load_snapshot( snapshot_file)
	testbed.nerf.render_with_camera_distortion = True
	testbed.snap_to_pixel_centers = True
	spp = 1
	testbed.nerf.rendering_min_transmittance = 1e-4
	testbed.fov_axis = 0
	testbed.fov = camera_angle_x * 180 / np.pi
	testbed.fov_axis = 1
	testbed.fov = camera_angle_y * 180 / np.pi

	# Set render mode
	testbed.render_mode = ngp.RenderMode.Depth

	# testbed.nerf.training.depth_supervision_lambda = 1.0
	# Adjust DeX threshold value
	# testbed.dex_nerf = True
	# testbed.sigma_thrsh = sigma_thrsh

	# Set camera matrix
	for img_name, c2w_matrix in zip(img_names, poses):
		# testbed.set_nerf_camera_matrix(np.matrix(c2w_matrix)[:-1, :])
		testbed.set_nerf_camera_matrix(np.matrix(c2w_matrix)[:-1, :])
		# Render estimated depth
		depth_raw = testbed.render(width, height, spp, True)  # raw depth values (float, in m)
		# print (depth_raw.shape)
		# print(np.max(depth_raw), np.min(depth_raw))
		# test1= depth_raw[..., 0]
		# print(test1.shape)
		# print(np.max(test1), np.min(test1))
		#
		# test2= depth_raw[..., 1]
		# print(test2.shape)
		# print(np.max(test2), np.min(test2))
		#
		# test3= depth_raw[..., 2]
		# print(test3.shape)
		# print(np.max(test3), np.min(test3))
		#
		# test4= depth_raw[..., 3]
		# print(test4.shape)
		# print(np.max(test4), np.min(test4))


		depth_raw= depth_raw[..., 0]
		# depth_norm = depth_raw / np.max(depth_raw)
		# depth_norm =(depth_raw - np.min(depth_raw)) / (np.max(depth_raw) - np.min(depth_raw))
		# depth_raw= 0.1-depth_raw
		# print(np.max(depth_raw),np.min(depth_raw) )
		depth_int = 1000 * depth_raw  # transform depth to mm
		# print(np.max(depth_int), np.min(depth_int))
		depth_int = depth_int.astype(np.uint16)
		cv2.imwrite(depth_dir_abs+os.sep + img_name[:-3] + 'png', depth_int)

	ngp.free_temporary_memory()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--scene', required=True)
	parser.add_argument('--snapshot', required=True)
	args = parser.parse_args()
	depth_est(args.scene, depth_dir="depth", sigma_thrsh=15, snapshot_file=args.snapshot)
