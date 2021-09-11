import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='shut down process by kill command')

    parser.add_argument('--key', type=str, default='')

    args = parser.parse_args()

    os.system('ps -ef | grep ' + args.key + ' | grep -v grep | cut -c 9-16 | xargs kill -9')