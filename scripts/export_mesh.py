import argparse
import os
import trimesh

from common import *

import pyngp as ngp

from tqdm import trange

def parse_args():
    parser = argparse.ArgumentParser(description='Single mesh export')
    parser.add_argument('transforms_path')
    parser.add_argument('out_obj_path')
    parser.add_argument('--force_overwrite', action='store_true')
    parser.add_argument('--iterations', type=int, default=25000)
    parser.add_argument('--resolution', type=int, default=1024)
    return parser.parse_args()

def process(transforms_path, iterations, resolution, out_obj_path, force_overwrite):
    if os.path.exists(out_obj_path) and not force_overwrite:
        print(f'Path {out_obj_path} already exists, exiting.')
        return
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.nerf.render_with_camera_distortion = True
    configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
    base_network = os.path.join(configs_dir, "base.json")
    testbed.reload_network_from_file(base_network)
    testbed.shall_train = True
    testbed.load_training_data(transforms_path)
    for _ in trange(iterations):
        testbed.frame()
    if os.path.exists(out_obj_path) and force_overwrite:
        os.remove(out_obj_path)
    mesh_it = testbed.compute_and_save_marching_cubes_mesh
    mesh_it(out_obj_path,
            [resolution, resolution, resolution],
            thresh=0)
    loaded_mesh = trimesh.load(out_obj_path, process=False)
    loaded_mesh.invert()
    loaded_mesh.export(out_obj_path)


if __name__ == '__main__':
    args = parse_args()

    process(args.transforms_path, args.iterations, args.resolution, args.out_obj_path, args.force_overwrite)
