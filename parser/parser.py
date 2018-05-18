import os
import argparse
from ast import literal_eval
import json

class Parser(object):

    def get_arguments(self):
        """Parse all the arguments provided from the CLI.

        Returns:
          A list of parsed arguments.
        """
        # Change arguments separately
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default="fcnn")
        parser.add_argument("--norm", type=str, default="test")
        parser.add_argument("--solver", type=str, default="linear")
        args = parser.parse_args()

        # Get specified model
        json_data = open("./models/models.json").read()
        params = json.loads(json_data)[args.model]
        params["norm"] = args.norm
        params["solver"] = args.norm

        return params
