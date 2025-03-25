#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""model pruning with different options script."""

import configargparse
import logging
import os
import random
import sys

import numpy as np

from espnet.utils.cli_utils import strtobool

# NOTE: you need this func to generate our sphinx doc


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Prune a given model and save it using "
        " one CPU ",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    #parser.add_argument("--seed", type=int, default=1, help="Random seed")
    #parser.add_argument("--debugmode", type=int, default=1, help="Debugmode")
    parser.add_argument("--verbose", "-V", type=int, default=1, help="Verbose option")
   # model (parameter) related
    parser.add_argument(
        "--model", type=str, required=True, help="Model file parameters to read"
    )
    parser.add_argument(
        "--model-conf", type=str, default=None, help="Model config file"
    )
    parser.add_argument(
        "--espnet2",
        type=lambda x: strtobool(x.strip()), nargs='?',
        const=True, default=False,
        help="Use espnet2 model",    
    )
    parser.add_argument(
        "--mttask",
        type=lambda x: strtobool(x.strip()), nargs='?',
        const=True, default=False,
        help="Use MTTask for espnet2 model",    
    )
    # search related
    # lbz: prune model related
    parser.add_argument(
        "--prune-asr-model",
        type=lambda x: strtobool(x.strip()), nargs='?',
        const=True, default=False,
        #type=bool,
        #default=False,
        help="Prune asr model",
    )
    parser.add_argument(
        "--prune-asr-model-tile-percentV2",
        type=lambda x: strtobool(x.strip()), nargs='?',
        const=True, default=False,
        #       type=bool,
        #        default=False,
        help="Prune asr model then group weights",
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=0,
        help="Prune asr model then group weights into tiles",
    )
    parser.add_argument(
        "--thres",
        type=float,
        default=0.0,
        help="Prune asr model then group weights into tiles",
    )
    parser.add_argument(
        "--asr-model-stats",
        type=lambda x: strtobool(x.strip()), nargs='?',
        const=True, default=False,
        #       type=bool,
        #        default=False,
        help="Provide asr model stats",
    )
    parser.add_argument(
        "--save-to",
        type=str,
        default="model.pytorch",
        help="save to model name",
    )
    parser.add_argument(
        "--tileFF",
        type=lambda x: strtobool(x.strip()), nargs='?',
        const=True, default=False,
        #       type=bool,
        #        default=False,
        help="tiles only on FF layers",
    )
    return parser


def main(args):
    """Run the main decoding function."""
    parser = get_parser()
    args = parser.parse_args(args)

    
    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose == 2:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    
    # display PYTHONPATH
    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))
    # seed setting
    #random.seed(args.seed)
    #np.random.seed(args.seed)
    #logging.info("set random seed = %d" % args.seed)

    # recog
    
    import prunefonctions
    prunefonctions.prunetransformer(args)


if __name__ == "__main__":
    main(sys.argv[1:])
