import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ABAW6 Preprocess')

    parser.add_argument('-python_package_path', default='/misc/scratch11/abaw5_preprocessing', type=str,
                    help='The path to the entire repository.')
    parser.add_argument('-part', default=-1, type=int,
                    help='Which part of the data to preprocess? Int 0 - 7, totally 8 parts. And use -1 to represent all.' )
    args = parser.parse_args()
    sys.path.insert(0, args.python_package_path)

    from project.abaw5.preprocessing import PreprocessingABAW5
    from project.abaw5.configs import config

    pre = PreprocessingABAW5(args.part, config)
    pre.generate_per_trial_info_dict()
    pre.prepare_data()


