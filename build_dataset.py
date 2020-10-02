import logging
from lib.log import logger
import numpy as np
import importlib
import argparse
import os
from inspect import signature


if __name__ == '__main__':
    logger.setup(
        filename='../build_dataset.log',
        filemode='w',
        root_level=logging.DEBUG,
        log_level=logging.DEBUG,
        logger='build_dataset'
    )
    np.random.seed(0)
    parser = argparse.ArgumentParser(description='Build dataset using specified processor')
    parser.add_argument('-p', dest='processor', nargs='?', default='', help="Processor to use for building the dataset")
    parser.add_argument('-s', dest='source', nargs='?', default=None, help="Base dataset's full path in format <dataset>.<index>")
    parser.add_argument('-d', dest='dest', nargs='?', default=None, help="Target dataset's full path in format <dataset>.<index>")
    parser.add_argument('-w', dest='window', nargs='?', default=10, help="Window width for lagged features")
    args = parser.parse_args()

    if not args.processor:
        args.processor = 'merged'
    if not os.path.exists('./processors/{}.py'.format(args.processor)):
        print("Processor module does not exist!")
        exit(0)
    p = importlib.import_module('processors.' + args.processor)
    if not args.source and (not hasattr(p, 'NEED_SOURCE_INDEX') or p.NEED_SOURCE_INDEX):
        print("This processor requires a source dataset!")
        exit(0)
    if not args.dest and (not hasattr(p, 'NEED_DEST_INDEX') or p.NEED_DEST_INDEX):
        print("No target specified!")
        exit(0)


    p.build(args.source, args.dest, W=args.window)