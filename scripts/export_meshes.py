import argparse
import os
import trimesh

from common import *

import pyngp as ngp # noqa

from tqdm import trange

def parse_args():
    parser = argparse.ArgumentParser(description='Export meshes for an entire folder')
    parser.add_argument('folder')
    return parser.parse_args()

def process(folder):
    for item in os.listdir(folder):
        testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
        testbed.nerf.render_with_camera_distortion = True
        configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
        base_network = os.path.join(configs_dir, "base.json")
        testbed.reload_network_from_file(base_network)
        testbed.shall_train = True
        if os.path.isdir(folder + os.sep + item):
            nerf_folder = folder + os.sep + item
            target_json = None
            for nerf_folder_file in os.listdir(nerf_folder):
                if nerf_folder_file.endswith('.json'):
                    target_json = nerf_folder_file
                    break
            if target_json is None:
                print(f'No JSON file found in {nerf_folder}')
                continue
            testbed.load_training_data(nerf_folder + os.sep + target_json)
            for i in trange(25000):
                testbed.frame()
            res = 1024
            thresh = 0
            target = nerf_folder + os.sep + 'neus2_' + item + '.obj'
            print(f"Writing mesh to {target}")
            if os.path.exists(target):
                os.remove(target)
            mesh_it = testbed.compute_and_save_marching_cubes_mesh
            mesh_it(target,
                    [res, res, res],
                    thresh=thresh)
            loaded_mesh = trimesh.load(target, process=False)
            loaded_mesh.invert()
            loaded_mesh.export(target)
        ngp.free_temporary_memory()
    exit()

            

if __name__ == '__main__':
    args = parse_args()

    process(args.folder)
