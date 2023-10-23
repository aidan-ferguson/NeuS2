import argparse
import os
import trimesh

def parse_args():
    parser = argparse.ArgumentParser(description='Scale meshes in folder according to scale_factor')
    parser.add_argument('folder')
    parser.add_argument('scale', type=float)
    return parser.parse_args()

def process(folder, scale):
    for folders_in_root in os.listdir(folder):
        for f in os.listdir(folder + os.sep + folders_in_root):
            if f.startswith('neus2_') and f.endswith('.obj'):
                target = folder + os.sep + folders_in_root + os.sep + f
                loaded_mesh = trimesh.load(target, process=False)
                loaded_mesh.apply_scale(scale)
                loaded_mesh.export(target)

if __name__ == '__main__':
    args = parse_args()

    process(args.folder, args.scale)

