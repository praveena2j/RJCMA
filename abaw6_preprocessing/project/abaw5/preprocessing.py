from base.preprocessing import GenericVideoPreprocessing

from base.utils import ensure_dir, save_to_pickle, load_pickle

import os
import glob


import pandas as pd
import numpy as np
import cv2


class PreprocessingABAW5(GenericVideoPreprocessing):
    def __init__(self, part, config, cnn_path=None):
        super().__init__(part, config, cnn_path=cnn_path)
        self.trial_list = {}
        self.task = config['task']
        # self.task_list = ["VA"]
        # self.multitask = config['multi_task']
        self.partition_list = ["train", "validate", "extra", "test"]
        self.generate_task_trial_list()

    def generate_task_trial_list(self):

        # Load all the trials as per the task and partition
        # Overall, we have 3 tasks by 3 partitions, plus an extra universal list, totally 10 lists.
        self.trial_list = {self.task: {partition: [] for partition in self.partition_list}}
        universal_list = self.get_all_file(os.path.join(self.config['root_directory'], self.config['cropped_aligned_folder']), filename_only=True)
        for partition in self.partition_list:
            if partition == "train" or partition == "validate":
                self.trial_list[self.task][partition] = [file.split(".txt")[0] for file in self.get_all_file(os.path.join(self.config['root_directory'], self.config['annotation_folder'], self.task, partition), filename_only=True)]

            elif partition == "extra":
                self.trial_list[self.task][partition] = list(set(universal_list).difference(self.trial_list[self.task]['train']).difference(self.trial_list[self.task]['validate']))

        self.dataset_info['trial_list'] = self.trial_list

    def generate_iterator(self):
        path = os.path.join(self.config['root_directory'], self.config['cropped_aligned_folder'])
        iterator = sorted([f.path for f in os.scandir(path) if f.is_dir()])
        return iterator

    def split_trials(self, full_trial_info):

        if self.part == -1:
            return full_trial_info

        if self.part >= 8:
            raise ValueError("Part should be -1 or 0 - 7")# Return an empty list if the index is out of range


        num_sublists = 8
        total_length = len(full_trial_info)
        sublist_length = total_length // num_sublists
        remainder = total_length % num_sublists

        start_index = self.part * sublist_length + min(self.part, remainder)
        end_index = start_index + sublist_length + (1 if self.part < remainder else 0)

        return full_trial_info[start_index:end_index]

    def generate_per_trial_info_dict(self):

        per_trial_info_path = os.path.join(self.config['output_root_directory'], "processing_records.pkl")

        if os.path.isfile(per_trial_info_path):
            per_trial_info = load_pickle(per_trial_info_path)

        else:

            per_trial_info = []
            iterator = self.generate_iterator()

            subject_id = 0
            for idx, file in enumerate(iterator):
                print(file)
                this_trial = {}

                this_trial['cropped_aligned_path'] = file
                this_trial['trial'] = file.split(os.sep)[-1]
                this_trial['trial_no'] = 1
                this_trial['subject_no'] = subject_id
                this_trial['video_path'] = this_trial['audio_path'] = self.get_video_path(file)
                video = cv2.VideoCapture(this_trial['video_path'])
                this_trial['fps'] = video.get(cv2.CAP_PROP_FPS)
                this_trial['target_fps'] = video.get(cv2.CAP_PROP_FPS)

                this_trial, wild_trial = self.get_partition(this_trial)

                if wild_trial:
                    continue

                # If it is from the test set, then the trial length is taken as the video length, otherwise the label length (without the rows containing -1 or -5).
                annotated_index = np.arange(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))

                load_continuous_label_kwargs = {}
                load_continuous_label_kwargs['cols'] = self.config[self.task + '_label_cols']
                load_continuous_label_kwargs['task'] = self.task

                if this_trial['has_' + self.task + '_label']:
                    continuous_label = self.load_continuous_label(this_trial[self.task + '_label_path'], **load_continuous_label_kwargs)
                    _, annotated_index = self.process_continuous_label(continuous_label, **load_continuous_label_kwargs)

                this_trial['length'] = len(annotated_index)

                get_annotated_index_kwargs = {}
                get_annotated_index_kwargs['source_frequency'] = this_trial['fps']

                get_annotated_index_kwargs['feature'] = "video"
                this_trial['video_annotated_index'] = self.get_annotated_index(annotated_index, **get_annotated_index_kwargs)

                # cnn = self.config['cnn_extractor_config']['model_name']
                # this_trial['cnn_' + cnn + '_annotated_index'] = this_trial['video_annotated_index']

                get_annotated_index_kwargs['feature'] = "mfcc"
                this_trial['mfcc_annotated_index'] = self.get_annotated_index(annotated_index, **get_annotated_index_kwargs)

                get_annotated_index_kwargs['feature'] = "egemaps"
                this_trial['egemaps_annotated_index'] = self.get_annotated_index(annotated_index, **get_annotated_index_kwargs)

                get_annotated_index_kwargs['feature'] = "logmel"
                this_trial['logmel_annotated_index'] = self.get_annotated_index(annotated_index, **get_annotated_index_kwargs)

                get_annotated_index_kwargs['feature'] = "vggish"
                this_trial['vggish_annotated_index'] = self.get_annotated_index(annotated_index, **get_annotated_index_kwargs)

                this_trial['speech_annotated_index'] = annotated_index
                per_trial_info.append(this_trial)

                subject_id += 1

        ensure_dir(per_trial_info_path)
        save_to_pickle(per_trial_info_path, per_trial_info, replace=True)
        selected_info = self.split_trials(per_trial_info)
        self.per_trial_info = selected_info

    def generate_dataset_info(self):

        self.dataset_info['pseudo_partition'] = []

        for idx, record in enumerate(self.per_trial_info):

            if 'partition' in record[self.task]:
                print(record['trial'])
                partition = record[self.task]['partition']
                self.dataset_info['trial'].append(record['processing_record']['trial'])
                self.dataset_info['trial_no'].append(record['trial_no'])
                self.dataset_info['subject_no'].append(record['subject_no'])
                self.dataset_info['length'].append(record['length'])
                self.dataset_info['partition'].append(partition)

                # For verifying semi-supervision
                if partition == "validate":
                    self.dataset_info['pseudo_partition'].append(partition)
                elif partition == "train":
                    n, p = 1, .7  # number of toss, probability of each toss
                    s = np.random.binomial(n, p, 1)
                    if s == 1:
                        self.dataset_info['pseudo_partition'].append('train')
                    else:
                        self.dataset_info['pseudo_partition'].append('extra')
                else:
                    self.dataset_info['pseudo_partition'].append('unused')

        self.dataset_info['data_folder'] = self.config['npy_folder']

        path = os.path.join(self.config['output_root_directory'], f'dataset_info_{self.part}.pkl')
        save_to_pickle(path, self.dataset_info)

    def compact_facial_image(self, path, annotated_index, extension="jpg"):
        from PIL import Image
        trial_length = len(annotated_index)

        frame_matrix = np.zeros((
            trial_length, self.config['video_size'], self.config['video_size'], 3), dtype=np.uint8)

        for j, frame in enumerate(range(trial_length)):
            current_frame_path = os.path.join(path, str(j + 1).zfill(5) + ".jpg")
            if os.path.isfile(current_frame_path):
                current_frame = Image.open(current_frame_path)
                frame_matrix[j] = current_frame.resize((self.config['video_size'], self.config['video_size']))
        return frame_matrix

    def extract_continuous_label_fn(self, idx, npy_folder):

            condition = self.per_trial_info[idx]['has_' + self.task + '_label']

            if condition:

                load_continuous_label_kwargs = {}

                if self.per_trial_info[idx]['has_' + self.task + '_label']:
                    load_continuous_label_kwargs['cols'] = self.config[self.task + '_label_cols']
                    load_continuous_label_kwargs['task'] = self.task
                    continuous_label = self.load_continuous_label(self.per_trial_info[idx][self.task + '_label_path'],
                                                                  **load_continuous_label_kwargs)
                    continuous_label, annotated_index = self.process_continuous_label(continuous_label, **load_continuous_label_kwargs)

                    if self.config['save_npy']:
                        filename = os.path.join(npy_folder, self.task + "_continuous_label.npy")
                        if not os.path.isfile(filename):
                            ensure_dir(filename)
                            np.save(filename, continuous_label)

    def load_continuous_label(self, path, **kwargs):

        cols = kwargs['cols']

        continuous_label = pd.read_csv(path, sep=",",
                                       skipinitialspace=True, usecols=cols,
                                       index_col=False).values.squeeze()

        return continuous_label

    def get_annotated_index(self, annotated_index, **kwargs):

        feature = kwargs['feature']
        source_frequency = kwargs['source_frequency']
        target_frequency = self.config['frequency'][feature]

        if feature == "video" or feature == "vggish" or feature == "mfcc" or feature == "egemaps" or feature == "logmel":
            target_frequency = source_frequency

        sampled_index = np.asarray(np.round(target_frequency / source_frequency * annotated_index), dtype=np.int64)

        return sampled_index

    def process_continuous_label(self, continuous_label, **kwargs):
        task = kwargs['task']
        not_labeled = 0

        if continuous_label.ndim == 1:
            continuous_label = continuous_label[:, np.newaxis]

        if task == "AU":
            not_labeled = -12
        elif task == "EXPR":
            not_labeled = -1
        elif task == "VA":
            not_labeled = -10
        else:
            ValueError("Unknown task!")

        row_wise_sum = np.sum(continuous_label, axis=1)
        annotated_index = np.asarray(np.where(row_wise_sum != not_labeled)[0], dtype=np.int64)

        return continuous_label[annotated_index], annotated_index

    @staticmethod
    def read_txt(txt_file):
        lines = pd.read_csv(txt_file, header=None)[0].tolist()
        return lines

    def get_partition(self, this_trial):

        # Some trials may not be in any task and any partitions. We have to exclude it.
        wild_trial = 1

        this_trial[self.task] = {}
        this_trial['has_' + self.task + '_label'] = 0

        trial_name = this_trial['trial']

        for partition in self.partition_list:
            if trial_name in self.trial_list[self.task][partition]:
                this_trial[self.task]['partition'] = partition
                this_trial[self.task + '_label_path'] = None
                label_path = os.path.join(self.config['root_directory'], self.config['annotation_folder'], self.task,
                                          partition, trial_name + ".txt")
                this_trial[self.task + '_label_path'] = label_path

                if os.path.isfile(label_path):
                    this_trial['has_' + self.task + '_label'] = 1

            wild_trial = 0

        return this_trial, wild_trial

    @staticmethod
    def get_output_filename(**kwargs):
        trial_name = kwargs['trial_name']
        return trial_name

    @staticmethod
    def get_video_path(video_name):
        if video_name.endswith("_right"):
            video_name = video_name[:-6]
        elif video_name.endswith("_left"):
            video_name = video_name[:-5]

        video_name = video_name.replace("cropped_aligned", "raw_videos")
        video_name = [video_name + ".mp4" if os.path.isfile(video_name + ".mp4") else video_name + ".avi"][0]
        return video_name

    @staticmethod
    def get_all_file(path, filename_only=False):

        all_files = glob.glob(os.path.join(path, "*"))
        if filename_only:
            all_files = [file.split(os.sep)[-1] for file in all_files]
        return all_files



if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Hello, world.')

    parser.add_argument('-python_package_path', default='/home/zhangsu/phd4', type=str,
                    help='The path to the entire repository.')
    args = parser.parse_args()
    sys.path.insert(0, args.python_package_path)

    from project.abaw5.configs import config

    pre = PreprocessingABAW5(config)
    pre.generate_per_trial_info_dict()
    pre.prepare_data()
