from conceptgraph.completion.kitti_inventory_20261125 import tracking, depth_tar
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['tracking', 'depth_tar'])
    {'tracking': tracking, 'depth_tar': depth_tar}[parser.parse_args().action]()
