import cv2
import subprocess
import os
from base.utils import copy_file, get_filename_from_a_folder_given_extension, ensure_dir
import csv
import sys

from tqdm import tqdm
import numpy as np
from PIL import Image
# from torchvision.utils import save_image, make_grid
# from torchvision import transforms
# import torch



class VideoSplit(object):
    r"""
        A base class to  split video according to a list. For example, given
        [(0, 1000), (1200, 1500), (1800, 1900)] as the indices, the associated
        frames will be split and combined  to form a new video.
    """

    def __init__(self, input_filename, output_filename, trim_range):
        r"""
        The init function of the class.
        :param input_filename: (str), the absolute directory of the input video.
        :param output_filename:  (str), the absolute directory of the output video.
        :param trim_range: (list), the indices of useful frames.
        """

        self.input_filename = input_filename
        self.output_filename = output_filename

        self.video = cv2.VideoCapture(self.input_filename)

        # The frame count.
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        # The fps count.
        self.fps = self.video.get(cv2.CAP_PROP_FPS)

        # The size of the video.
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # The range to trim the video.
        self.trim_range = trim_range

        # The settings for video writer.
        self.codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.writer = cv2.VideoWriter(output_filename,
                                      self.codec, self.fps,
                                      (self.width, self.height), isColor=True)

    def jump_to_frame(self, frame_index):
        r"""
        Jump to a specific frame by its index.
        :param frame_index:  (int), the index of the frame to jump to.
        :return: none.
        """
        self.video.set(1, frame_index)

    def read(self, start, end, visualize):
        r"""
        Read then write the frames within (start, end) one frame at a time.
        :param start:  (int), the starting index of the range.
        :param end:  (int), the ending index of the range.
        :param visualize:  (boolean), whether to visualize the process.
        :return:  none.
        """

        # Jump to the starting frame.
        self.jump_to_frame(start)

        # Sequentially write the next end-start frames.
        for index in range(end - start):
            ret, frame = self.video.read()
            self.writer.write(frame)
            if ret and visualize:
                cv2.imshow('frame', frame)
                # Hit 'q' on the keyboard to quit!
                cv2.waitKey(1)

    def combine(self, visualize=False):
        r"""
        Combine the clips  into a single video.
        :param visualize: (boolean), whether to visualize the process.
        :return:  none.
        """

        # Iterate over the pair of start and end.
        for (start, end) in self.trim_range:
            self.read(start, end, visualize)

        self.video.release()
        self.writer.release()
        if visualize:
            cv2.destroyWindow('frame')


def change_video_fps(input_path, output_path, target_fps):
    r"""
    Alter the frame rate of a given video.
    :param videos:  (list),a list of videos to process.
    :param target_fps:  (float), the desired fps.
    :return: (list or str), the list  (or str if only one input video) of videos after the process.
    """
    output_video_list = []
    print("Changing video fps...")

    # If the new name already belongs to a file, then do nothing.
    if os.path.isfile(output_path):
        print("Skipped fps conversion for video {}!".format(output_path))
        pass

    # If not, call the ffmpeg tools to change the fps.
    # -qscale:v 0 can preserve the quality of the frame after the recoding.
    else:
        input_codec = " xvid "
        if ".mp4" in input_path:
            input_codec = " mp4v "
        ffmpeg_command = "ffmpeg -i {} -filter:v fps=fps={} -c:v mpeg4 -vtag {} -qscale:v 0 {}".format(
            '"' + input_path + '"', str(target_fps), input_codec,
            '"' + output_path + '"')

        full_command = "export PATH={conda_path}/bin:$PATH && {ffmpeg_command}".format(conda_path=sys.exec_prefix,
                                                                                       ffmpeg_command=ffmpeg_command)
        subprocess.call(full_command, shell=True)


def combine_annotated_clips(
        input_path,
        output_path,
        trim_range,
        direct_copy=False,
        visualize=False
):


    print("combining annotated clips...")

    # If the new name already belongs to a file, then do nothing.
    if os.path.isfile(output_path):
        print("Skipped video combination for video {}!".format(output_path))
        pass

    # If not, call the video combiner.
    else:
        if not direct_copy:
            video_split = VideoSplit(input_path, output_path, trim_range)
            video_split.combine(visualize)
        else:
            copy_file(input_path, output_path)


class OpenFaceController(object):
    def __init__(self, openface_config, output_directory):
        self.openface_config = openface_config
        self.output_directory = output_directory

    def get_openface_command(self):
        openface_path = self.openface_config['directory']
        input_flag = self.openface_config['input_flag']
        output_features = self.openface_config['output_features']
        output_action_unit = self.openface_config['output_action_unit']
        output_image_flag = self.openface_config['output_image_flag']
        output_image_size = self.openface_config['output_image_size']
        output_image_format = self.openface_config['output_image_format']
        output_filename_flag = self.openface_config['output_filename_flag']
        output_directory_flag = self.openface_config['output_directory_flag']
        output_directory = self.output_directory
        output_image_mask_flag = self.openface_config['output_image_mask_flag']

        command = openface_path + input_flag + " {input_filename} " + output_features \
                  + output_action_unit + output_image_flag + output_image_size \
                  + output_image_format + output_filename_flag + " {output_filename} " \
                  + output_directory_flag + output_directory + output_image_mask_flag
        return command

    def process_video(self, input_filename, output_filename):

        # Quote the file name if spaces occurred.
        if " " in input_filename:
            input_filename = '"' + input_filename + '"'

        command = self.get_openface_command()
        command = command.format(
            input_filename=input_filename, output_filename=output_filename)

        if not os.path.isfile(os.path.join(self.output_directory, output_filename + ".csv")):
            subprocess.call(command, shell=True)

        return os.path.join(self.output_directory, output_filename)


class FacenetController(object):
    def __init__(self, mtcnn, face_landmark_detector, device, image_size, landmark_handler, batch_size, input_path, output_path, output_filename):
        self.mtcnn = mtcnn
        self.face_landmark_detector = face_landmark_detector
        self.device = device
        self.image_size = image_size
        self.landmark_handler = landmark_handler
        self.batch_size = batch_size
        self.input_path = input_path
        self.output_path = os.path.join(output_path, output_filename)
        self.is_video = input_path.endswith(".mp4")
        self.done = os.path.isfile(os.path.join(self.output_path + ".csv"))


    def get_dataloader(self):

        if self.is_video:
            import mmcv
            video = mmcv.VideoReader(self.input_path)
            frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
            indices = [i for i in range(video.frame_cnt)]
            sizes = [frame.size for frame in frames]
        else:
            image_paths = get_filename_from_a_folder_given_extension(self.input_path, 'jpg')
            frames = [Image.fromarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) ) for image_path in image_paths]
            indices = [image_path.split(os.sep)[-1][:-4] for image_path in image_paths]
            sizes = [frame.size for frame in frames]

        dataloader = [[frames[i:i + self.batch_size], indices[i:i + self.batch_size], sizes[i:i + self.batch_size]] for i in range(0, len(frames), self.batch_size)]

        return dataloader

    def crop_align_image(self, face, landmark):
        croped_image = self.landmark_handler.crop_image(face, landmark)
        return croped_image

    def pre_crop_by_bbox(self, dataloader, bboxes):
        for idx, bbox in tqdm(enumerate(bboxes), total=len(dataloader)):

            img_path = os.path.join(self.output_path, str(idx).zfill(5) + ".jpg")

            if os.path.isfile(img_path):
                continue
            frame = dataloader[idx]
            bbox_loc = bbox[1:]
            cropped_frame = frame[0][0].crop((bbox_loc))
            ensure_dir(img_path)
            cropped_frame.resize((self.image_size, self.image_size)).save(img_path)

    def process(self, dataloader):
        black_face = torch.zeros((3, self.image_size, self.image_size))
        csv_column = [
            'frame', 'face_id', 'time_stamp', 'confidence', 'success', *['x_' + str(i) for i in range(68)], *['y_' + str(i) for i in range(68)]
        ]
        csv_path = os.path.join(self.output_path + ".csv")
        ensure_dir(csv_path)

        if os.path.isfile(csv_path):
            return self.output_path

        with open(csv_path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(csv_column)

            for frames, idxes, sizes in tqdm(dataloader, total=len(dataloader)):
                faces, probs = self.mtcnn(frames, return_prob=True)

                for face, prob, idx, in zip(faces, probs, idxes):

                    success = 1

                    if face is None:
                        success = 0
                        prob = 0.0
                    else:
                        prob = prob[0]

                    image_path = os.path.join(self.output_path + "_aligned",
                                              str(idx).zfill(5) + ".jpg")
                    ensure_dir(image_path)

                    if success:
                        face_PIL = transforms.ToPILImage()(face.to(torch.uint8))
                        face = face.permute(1, 2, 0).to(torch.uint8)
                        landmarks_list = self.face_landmark_detector.get_landmarks(face)
                        if landmarks_list is None:
                            landmarks_list = [np.zeros((68, 2), dtype=np.float32)]
                    else:
                        face_PIL = transforms.ToPILImage()(black_face)
                        landmarks_list = [np.zeros((68, 2), dtype=np.float32)]
                    face_PIL.save(image_path)

                    csv_row = [
                        str(idx), '0', '0', str(np.round(prob, 2)), str(success), *landmarks_list[0][:, 0], *landmarks_list[0][:, 1]
                    ]
                    writer.writerow(csv_row)

        return self.output_path













