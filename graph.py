"""
Description...

File: graph.py
Author: Emilio Balda <emilio.balda@ti.rwth-aachen.de>
Organization:  RWTH Aachen University - Institute for Theoretical Information Technology
"""

import tensorflow as tf
import argparse
import json

def get_arguments():
    model = 'fcnn'

    parser = argparse.ArgumentParser(description="Creates tensorboard visualization files for ")
    parser.add_argument("--model", type=str, default=model,
                        help="model to be loaded: either of these --> fcnn, lenet, nin, densenet. Default value = " + model)
    return parser.parse_args()

def get_all_model_variables(args):

    json_data = open("./models/models.json").read()
    params = json.loads(json_data)[args.model]

    modelvarnames = {
    'model2load': params["index_file"],
    'models_dir': "models/data/",
    'visual_dir': "logs/{}/".format(params["model"]),
    'graph_directory': "{}/{}/".format(params["model"], params["dataset"]),
    'graph_file': params["graph_file"]
    }

    return modelvarnames

def main():
    args = get_arguments()
    allvars = get_all_model_variables(args)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Graph().as_default(), tf.Session(config = config) as sess:
        saver = tf.train.import_meta_graph(allvars['models_dir'] +\
                                           allvars['graph_directory'] +\
                                           allvars['graph_file'])

        saver.restore(sess, tf.train.latest_checkpoint(allvars['models_dir'] +\
                                                       allvars['graph_directory']))
        print("Model restored.")
        print()

        file_writer = tf.summary.FileWriter(allvars['visual_dir']+allvars['model2load']+'/', sess.graph)
        print('Tensorboard Visualization on:')
        print('tensorboard --logdir='+allvars['visual_dir']+allvars['model2load']+'/')
        print()

if __name__ == '__main__':
    main()
