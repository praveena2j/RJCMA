from base.video import change_video_fps, combine_annotated_clips, OpenFaceController, FacenetController
from base.audio import convert_video_to_wav, change_wav_frequency, extract_mfcc, extract_egemaps
from base.speech import extract_transcript, add_punctuation, extract_word_embedding, align_word_embedding
from base.utils import ensure_dir, get_filename_from_a_folder_given_extension, save_to_pickle
# from base.facial_landmark import facial_image_crop_by_landmark
import os

from operator import itemgetter
from tqdm import tqdm

import pandas as pd
import numpy as np
from PIL import Image


class GenericVideoPreprocessing(object):
    def __init__(self, part, config, cnn_path=None):

        self.part = part
        self.config = config
        self.cnn_path = cnn_path
        self.dataset_info = self.init_dataset_info()
        self.device =self.config['device']
        if "extract_continuous_label" in config and config['extract_continuous_label']:
            self.extract_continuous_label = config['extract_continuous_label']

        if "extract_class_label" in config and config['extract_class_label']:
            self.extract_class_label = config['extract_class_label']

        if "change_video_fps" in config and config['change_video_fps']:
            self.change_video_fps = config['change_video_fps']
            self.fps_changed_video_folder = config['fps_changed_video_folder']

        if "trim_video" in config and config['trim_video']:
            self.trim_video = config['trim_video']
            self.trimmed_video_folder = config['trimmed_video_folder']

        if "crop_align_face" in config and config['crop_align_face']:
            self.crop_align_face = config['crop_align_face']
            self.cropped_aligned_folder = config['cropped_aligned_folder']
            self.use_mtcnn = config['use_mtcnn']

        if "extract_cnn" in config and config['extract_cnn']:
            self.extract_cnn = config['extract_cnn']
            self.cnn_folder = config['cnn_folder']

        if "convert_to_wav" in config and config['convert_to_wav']:
            self.convert_to_wav = config['convert_to_wav']
            self.wav_folder = config['wav_folder']

        if "change_audio_frequency" in config and config['change_audio_frequency']:
            self.change_audio_frequency = config['change_audio_frequency']
            self.freq_changed_audio_folder = config['freq_changed_audio_folder']

        if "extract_facial_landmark" in config and config['extract_facial_landmark']:
            self.extract_facial_landmark = config['extract_facial_landmark']
            self.facial_landmark_folder = config['facial_landmark_folder']

        if "extract_action_unit" in config and config['extract_action_unit']:
            self.extract_action_unit = config['extract_action_unit']
            self.facial_landmark_folder = config['action_unit_folder']

        if "extract_mfcc" in config and config['extract_mfcc']:
            self.extract_mfcc = config['extract_mfcc']
            self.mfcc_folder = config['mfcc_folder']

        if "extract_egemaps" in config and config['extract_egemaps']:
            self.extract_egemaps = config['extract_egemaps']
            self.egemaps_folder = config['egemaps_folder']

        if "extract_compare" in config and config['extract_compare']:
            self.extract_compare = config['extract_compare']
            self.compare_folder = config['compare_folder']

        if "extract_logmel" in config and config['extract_logmel']:
            self.extract_logmel = config['extract_logmel']
            self.logmel_folder = config['logmel_folder']

        if "extract_vggish" in config and config['extract_vggish']:
            self.extract_vggish = config['extract_vggish']
            self.vggish_folder = config['vggish_folder']

        if "extract_bert" in config and config['extract_bert']:
            self.extract_bert = config['extract_bert']
            self.bert_folder = config['bert_folder']

        if "extract_glove" in config and config['extract_glove']:
            self.extract_glove = config['extract_glove']
            self.glove_folder = config['glove_folder']

        if "extract_transcript" in config and config['extract_transcript']:
            self.extract_transcript = config['extract_transcript']
            self.transcript_folder = config['transcript_folder']

        if "add_punctuation" in config and config['add_punctuation']:
            self.add_punctuation = config['add_punctuation']
            self.punctuation_folder = config['punctuation_folder']

        if "extract_word_embedding" in config and config['extract_word_embedding']:
            self.extract_word_embedding = config['extract_word_embedding']
            self.word_embedding_folder = config['word_embedding_folder']

        if "align_word_embedding" in config and config['align_word_embedding']:
            self.align_word_embedding = config['align_word_embedding']
            self.aligned_word_embedding_folder = config['aligned_word_embedding_folder']

        if "extract_eeg" in config and config['extract_eeg']:
            from base.eeg import GenericEegController
            self.extract_eeg = config['extract_eeg']
            self.eeg_folder = config['eeg_folder']

        self.per_trial_info = {}

    def get_output_root_directory(self):
        return self.config['output_root_directory']

    def prepare_data(self):
        if hasattr(self, 'extract_vggish'):
            from base.vggish.vggish import VGGish
            from base.vggish.hubconf import model_urls
            vggish_model = VGGish(device=self.device, urls=model_urls, pretrained=True, preprocess=False, postprocess=False, progress=True)
            vggish_model.eval()

        if hasattr(self, 'add_punctuation'):
            from deepmultilingualpunctuation import PunctuationModel
            punc_cap_model = PunctuationModel(model="oliverguhr/fullstop-punctuation-multilang-large")


        if hasattr(self, 'extract_word_embedding'):
            from transformers import BertTokenizer, BertModel
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
            bert.eval()

        if hasattr(self, 'crop_align_face') and self.use_mtcnn:
            from facenet_pytorch import MTCNN
            mtcnn = MTCNN(keep_all=False, device=self.device, image_size=self.config['video_size'], margin=self.config['margin_size'], select_largest=False, post_process=False)
        else:
            mtcnn = None

        if hasattr(self, 'extract_facial_landmark') and self.extract_facial_landmark and self.use_mtcnn:
            import face_alignment
            face_landmark_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:{}'.format(str(self.device.index)))
        else:
            face_landmark_detector = None

        for idx in tqdm(range(len(self.per_trial_info)), total=len(self.per_trial_info)):

            self.per_trial_info[idx]['processing_record'] = {}
            get_output_filename_kwargs = {}
            get_output_filename_kwargs['subject_no'] = self.per_trial_info[idx]['subject_no']
            get_output_filename_kwargs['trial_no'] = self.per_trial_info[idx]['trial_no']
            get_output_filename_kwargs['trial_name'] = self.per_trial_info[idx]['trial']
            output_filename = self.get_output_filename(**get_output_filename_kwargs)
            npy_folder = os.path.join(self.config['output_root_directory'], self.config['npy_folder'], output_filename)
            ensure_dir(npy_folder)

            self.per_trial_info[idx]['processing_record']['trial'] = output_filename

            self.per_trial_info[idx]['video_npy_path'] = os.path.join(npy_folder, "video.npy")

            # Load the continuous labels
            if hasattr(self, 'extract_continuous_label'):
                self.extract_continuous_label_fn(idx, npy_folder)

            # Load the continuous labels
            if hasattr(self, 'extract_class_label'):
                self.extract_class_label_fn(self.per_trial_info[idx])

            ### VIDEO PREPROCESSING
            # Pick only the annotated frames from a video.
            if hasattr(self, 'trim_video'):
                self.trim_video_fn(idx, output_filename)

            # Change video fps for the video of this trial.
            if hasattr(self, 'change_video_fps'):
                self.change_video_fps_fn(idx, output_filename)

            if hasattr(self, 'pre_crop_by_bbox'):
                self.pre_crop_by_bbox(idx, output_filename)

            # Extract facial landmark, warp, crop, and save each frame.
            if hasattr(self, 'crop_align_face'):
                self.crop_align_face_fn(idx, output_filename, npy_folder, mtcnn, face_landmark_detector)

            if hasattr(self, "extract_facial_landmark"):
                self.extract_facial_landmark_fn(idx, output_filename, npy_folder)

            if hasattr(self, "extract_action_unit"):
                self.extract_action_unit_fn(idx, output_filename, npy_folder)

            if hasattr(self, "extract_cnn"):
                self.extract_cnn_fn(idx, output_filename, npy_folder)

            ### AUDIO PREPROCESSING
            # Convert video to wav.
            if hasattr(self, 'convert_to_wav'):
                self.convert_to_wav_fn(idx, output_filename)

            # Change the sampling frequency of the audio.
            if hasattr(self, 'change_audio_frequency'):
                self.change_audio_frequency_fn(idx, output_filename)

            # Extract mfcc
            if hasattr(self, 'extract_mfcc'):
                self.extract_mfcc_fn(idx, output_filename, npy_folder)

            # Extract egemaps
            if hasattr(self, 'extract_egemaps'):
                self.extract_egemaps_fn(idx, output_filename, npy_folder)

            # Extract logmel and vggish
            if hasattr(self, 'extract_logmel'):
                self.extract_logmel_fn(idx, output_filename, npy_folder)

            # Extract logmel and vggish
            if hasattr(self, 'extract_compare'):
                self.extract_compare_fn(idx, output_filename, npy_folder)


            if hasattr(self, 'extract_vggish'):
                self.extract_vggish_fn(idx, output_filename, npy_folder, vggish_model)

            if hasattr(self, 'extract_glove'):
                self.extract_glove_fn(idx, output_filename, npy_folder)

            if hasattr(self, 'extract_bert'):
                self.extract_bert_fn(idx, output_filename, npy_folder)

            # Speech processing
            if hasattr(self, 'extract_transcript'):
                self.extract_transcript_fn(idx, output_filename)

            if hasattr(self, 'add_punctuation'):
                self.add_punctuation_fn(idx, output_filename, punc_cap_model)

            if hasattr(self, 'extract_word_embedding'):
                self.extract_word_embedding_fn(idx, output_filename, bert_tokenizer, bert)

            if hasattr(self, "align_word_embedding"):
                self.align_word_embedding_fn(idx, output_filename, npy_folder)

            # EEG processing
            if hasattr(self, 'extract_eeg'):
                self.extract_eeg_fn(idx, output_filename, npy_folder)

        path = os.path.join(self.config['output_root_directory'], 'processing_records.pkl')
        ensure_dir(path)
        save_to_pickle(path, self.per_trial_info)

        self.generate_dataset_info()

    def extract_bert_fn(self, idx, output_filename, npy_folder):
        pass

    def align_word_embedding_fn(self, idx, output_filename, npy_folder):
        input_path = os.path.join(self.config['output_root_directory'],
                                  *self.per_trial_info[idx]['embedding_path'][-2:])

        output_path = os.path.join(self.config['output_root_directory'], self.config['npy_folder'],
                                   output_filename, "bert.npy")

        ensure_dir(output_path)

        self.per_trial_info[idx]['processing_record']['bert_path'] = output_path.split(os.sep)
        self.per_trial_info[idx]['bert_path'] = output_path.split(os.sep)

        if self.config['save_npy']:
            filename = os.path.join(npy_folder, "bert.npy")
            if not os.path.isfile(filename):

                bert_features = align_word_embedding(input_path, self.per_trial_info[idx]['target_fps'], self.per_trial_info[idx]['video_annotated_index'])
                np.save(filename, bert_features)

    def extract_word_embedding_fn(self, idx, output_filename, tokenizer, bert):
        input_path = os.path.join(self.config['output_root_directory'],
                                  *self.per_trial_info[idx]['punctuation_path'][-2:])
        output_path = os.path.join(self.config['output_root_directory'], self.config['word_embedding_folder'],
                                   output_filename + ".csv")
        ensure_dir(output_path)

        bert.cuda()
        extract_word_embedding(input_path, output_path, tokenizer, bert)

        self.per_trial_info[idx]['processing_record']['embedding_path'] = output_path.split(os.sep)
        self.per_trial_info[idx]['embedding_path'] = output_path.split(os.sep)

    def add_punctuation_fn(self, idx, output_filename, model):
        input_path = os.path.join(self.config['output_root_directory'],
                                  *self.per_trial_info[idx]['transcript_path'][-2:])
        output_path = os.path.join(self.config['output_root_directory'], self.config['punctuation_folder'],
                                   output_filename + ".csv")
        ensure_dir(output_path)

        add_punctuation(input_path, output_path, model)

        self.per_trial_info[idx]['processing_record']['punctuation_path'] = output_path.split(os.sep)
        self.per_trial_info[idx]['punctuation_path'] = output_path.split(os.sep)

    def extract_transcript_fn(self, idx, output_filename):
        input_path = os.path.join(self.config['output_root_directory'], *self.per_trial_info[idx]['audio_path'][-2:])
        output_path = os.path.join(self.config['output_root_directory'], self.config['transcript_folder'],
                                   output_filename + ".csv")
        ensure_dir(output_path)
        extract_transcript(input_path, output_path, self.config['speech_model']['path'])

        self.per_trial_info[idx]['processing_record']['transcript_path'] = output_path.split(os.sep)
        self.per_trial_info[idx]['transcript_path'] = output_path.split(os.sep)

    def extract_eeg_fn(self, idx, output_filename, npy_folder):

        if self.per_trial_info[idx]['has_eeg']:
            not_done = 0
            for feature in self.config['eeg_config']['features']:
                filename = os.path.join(npy_folder, feature + ".npy")
                if not os.path.isfile(filename):
                    not_done = 1

            if "eeg_processed_path" in self.per_trial_info[idx]:
                output_path = os.path.join(self.config['output_root_directory'],
                                           self.per_trial_info[idx]['eeg_processed_path'][-2:])
            else:

                # input_path = self.per_trial_info[idx]['eeg_path']
                input_path = os.path.join(self.config['root_directory'], *self.per_trial_info[idx]['eeg_path'][-3:])
                output_path = os.path.join(self.config['output_root_directory'], self.config['eeg_folder'],
                                           output_filename)

                if not_done:
                    from base.eeg import GenericEegController
                    eeg_handler = GenericEegController(input_path, config=self.config['eeg_config'])

            self.per_trial_info[idx]['processing_record']['eeg_processed_path'] = output_path
            self.per_trial_info[idx]['eeg_processed_path'] = output_path.split(os.sep)

            if self.config['save_npy']:
                for feature in self.config['eeg_config']['features']:

                    filename = os.path.join(npy_folder, feature + ".npy")
                    self.per_trial_info[idx]['eeg_' + feature + '_npy_path'] = filename
                    if not os.path.isfile(filename):
                        # Save video npy
                        feature_np = eeg_handler.extracted_data[feature]
                        np.save(filename, feature_np)

    def generate_dataset_info(self):

        for idx, record in self.per_trial_info.items():
            self.dataset_info['trial'].append(record['processing_record']['trial'])
            self.dataset_info['trial_no'].append(record['trial_no'])
            self.dataset_info['subject_no'].append(record['subject_no'])
            self.dataset_info['length'].append(len(self.per_trial_info[idx]['continuous_label']))
            self.dataset_info['partition'].append(record['partition'])

        self.dataset_info['multiplier'] = self.config['multiplier']
        self.dataset_info['data_folder'] = self.config['npy_folder']

        path = os.path.join(self.config['output_root_directory'], 'dataset_info.pkl')
        save_to_pickle(path, self.dataset_info)

    def trim_video_fn(self, idx, output_filename):

        input_path = os.path.join(self.config['root_directory'], self.config['raw_data_folder'], *self.per_trial_info[idx]['video_path'][-2:])
        output_path = os.path.join(self.config['output_root_directory'], self.config['trimmed_video_folder'],
                                   output_filename + ".mp4")

        ensure_dir(output_path)
        trim_range = self.per_trial_info[idx]['video_trim_range']
        combine_annotated_clips(input_path, output_path, trim_range, direct_copy=False, visualize=False)

        self.per_trial_info[idx]['processing_record']['trimmed_video_path'] = output_path.split(os.sep)

        self.per_trial_info[idx]['video_path'] = output_path.split(os.sep)

    def change_video_fps_fn(self, idx, output_filename):

        input_path = os.path.join(self.config['output_root_directory'], *self.per_trial_info[idx]['video_path'][-2:])
        output_path = os.path.join(self.config['output_root_directory'], self.config['fps_changed_video_folder'],
                                   output_filename + "." + self.per_trial_info[idx]['extension'])

        ensure_dir(output_path)
        change_video_fps(input_path, output_path, self.per_trial_info[idx]['target_fps'])

        self.per_trial_info[idx]['processing_record']['fps_video_path'] = output_path.split(os.sep)

        self.per_trial_info[idx]['video_path'] = output_path.split(os.sep)

    def pre_crop_by_bbox(self, idx, output_filename):
        pass

    def crop_align_face_fn(self, idx, output_filename, npy_folder, mtcnn=None, face_landmark_detector=None):

        # if isinstance(self.per_trial_info[idx]['video_path'], str):
        #     self.per_trial_info[idx]['video_path'] = self.per_trial_info[idx]['video_path'].split(os.sep)
        input_path = os.path.join(self.config['output_root_directory'], *self.per_trial_info[idx]['video_path'][-2:])
        # input_path = os.sep.join(self.per_trial_info[idx]['video_path'])
        # if hasattr(self, "pre_crop_by_bbox"):
        #     input_path = os.path.join(self.config['output_root_directory'], *self.per_trial_info[idx]['processing_record']['pre_cropped_path'][-2:], )

        output_path = os.path.join(self.config['output_root_directory'],
                                                     self.config['cropped_aligned_folder'])

        if not self.use_mtcnn:
            output_path = os.path.join(output_path, output_filename)
            # openface = OpenFaceController(openface_config=self.config['openface'],
            #                               output_directory=output_path)
            #
            # output_path = openface.process_video(input_filename=self.per_trial_info[idx]['video_path'],
            #                                      output_filename=output_filename)

        else:

            config_landmark = self.config['landmark']
            landmark_handler = facial_image_crop_by_landmark(**config_landmark)
            fnet = FacenetController(
                mtcnn=mtcnn, face_landmark_detector=face_landmark_detector,
                device=self.device, image_size=self.config['video_size'], landmark_handler=landmark_handler,
                batch_size=128, input_path=input_path, output_path=output_path,
                output_filename=output_filename)

            if not fnet.done and not os.path.isfile(os.path.join(self.config['output_root_directory'],
                                           self.config['cropped_aligned_folder'], output_filename)):
                dataloader = fnet.get_dataloader()
                output_path = fnet.process(dataloader)
            else:
                output_path = os.path.join(self.config['output_root_directory'],
                                           self.config['cropped_aligned_folder'], output_filename)

        self.per_trial_info[idx]['processing_record']['cropped_aligned_path'] = output_path.split(os.sep)
        if self.config['save_npy']:
            filename = os.path.join(npy_folder, "video128.npy")
            self.per_trial_info[idx]['video_npy_path'] = filename
            if not os.path.isfile(filename):
                # Save video npy
                annotated_index = self.per_trial_info[idx]['video_annotated_index']
                video_matrix = self.compact_facial_image(output_path,
                                                         annotated_index=annotated_index,
                                                         extension="jpg")
                np.save(filename, video_matrix)

    def extract_facial_landmark_fn(self, idx, output_filename, npy_folder):

        output_path = os.path.join(self.config['output_root_directory'], self.config['facial_landmark_folder'],
                                   output_filename + ".csv")

        self.per_trial_info[idx]['processing_record']['facial_landmark_path'] = output_path.split(os.sep)

        if self.config['save_npy']:

            filename = os.path.join(npy_folder, "landmark.npy")
            if not os.path.isfile(filename):
                # Save facial landmarks
                start_col, end_col = 5, 141
                feature = "facial_landmark"
                annotated_index = self.per_trial_info[idx]['video_annotated_index']
                landmark = self.compact_audio_feature(output_path, annotated_index, start_col, end_col, feature)

                np.save(filename, landmark)

    def extract_action_unit_fn(self, idx, output_filename, npy_folder):

        output_path = os.path.join(self.config['output_root_directory'], self.config['action_unit_folder'],
                                   output_filename + ".csv")

        self.per_trial_info[idx]['processing_record']['action_unit_path'] = output_path.split(os.sep)
        if self.config['save_npy']:
            filename = os.path.join(npy_folder, "action_unit.npy")
            if not os.path.isfile(filename):
                # Save facial landmarks
                start_col = 141
                end_col = 158
                feature = "action_unit"
                annotated_index = self.per_trial_info[idx]['video_annotated_index']
                action_unit = self.compact_audio_feature(output_path, annotated_index, start_col, end_col, feature)

                np.save(filename, action_unit)

    def extract_cnn_fn(self, idx, output_filename, npy_folder):

        output_path = os.path.join(self.config['output_root_directory'], self.config['npy_folder'],
                                   output_filename + ".npy")
        ensure_dir(output_path)

        input_path = self.per_trial_info[idx]['video_npy_path']
        self.per_trial_info[idx]['processing_record']['cnn_path'] = ""

        model_config = self.config['cnn']
        if self.config['save_npy']:

            filename = os.path.join(npy_folder, "cnn_" + self.config['cnn']['model'] + ".npy")
            # filename = os.path.join(npy_folder, "cnn_mask" + ".npy")
            if self.cnn_path is not None:
                filename = os.path.join(npy_folder, "cnn_" + self.cnn_path  + ".npy")


            if not os.path.isfile(filename):
                cnn_features = self.compute_cnn(input_path, **model_config)
                np.save(filename, cnn_features)

    def compute_cnn(self, input_path, **kwargs):
        from base.dataset import preprocess_video_dataset
        from PIL import Image
        import torch
        from torch.utils.data import DataLoader

        if kwargs['model'] == "res50":
            from models.backbone import ResnetBackbone
            path = os.path.join(self.config['load_directory'], kwargs['state_dict'])
            if self.cnn_path is not None:
                path = os.path.join(self.config['load_directory'], self.cnn_path, kwargs['state_dict'])
            model = ResnetBackbone(mode='ir', use_pretrained=False)
            state_dict = torch.load(path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
        elif "ires" in kwargs['model']:
            from models.backbone import IResnetBackbone
            path = os.path.join(self.config['load_directory'], kwargs['state_dict'])
            if self.cnn_path is not None:
                path = os.path.join(self.config['load_directory'], self.cnn_path, kwargs['state_dict'])
            state_dict = torch.load(path, map_location='cpu')

            if kwargs['image_size'] == 48:
                model = IResnetBackbone(mode="50")
            elif kwargs['image_size'] == 135:
                model = IResnetBackbone(mode="100")
            else:
                raise ValueError("Unknown ires model!")
            # from models.prototype import AttentionVisualBackbone
            # model = AttentionVisualBackbone()

            model.backbone.load_state_dict(state_dict)
            model.to(self.device)
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
        else:
            raise ValueError("Unknown cnn models!")

        video = np.load(input_path)
        video = preprocess_video_dataset(video, config=kwargs)
        video_loader = DataLoader(video, batch_size=kwargs['batch_size'], shuffle=False, drop_last=False)

        cnn_features = []
        for images in tqdm(video_loader, total=len(video_loader)):
            images = images.to(self.device)
            # outputs = model.extract(images)
            outputs = model(images, extract_cnn=True)
            cnn_features.append(outputs.detach().cpu().numpy())
        cnn_features = np.vstack(cnn_features)

        return cnn_features

    def change_audio_frequency_fn(self, idx, output_filename):

        input_path = os.path.join(self.config['root_directory'], *self.per_trial_info[idx]['audio_path'][-2:])
        if len(self.per_trial_info[idx]['audio_path']) > 0 and isinstance(self.per_trial_info[idx]['audio_path'], str):
            input_path = os.path.join(self.config['root_directory'], *self.per_trial_info[idx]['audio_path'].split(os.sep)[-2:])

        output_path = os.path.join(self.config['output_root_directory'], self.config['wav_folder'],
                                   output_filename + "_" + str(self.config['target_frequency']) + ".wav")

        ensure_dir(output_path)
        change_wav_frequency(input_path=input_path, output_path=output_path,
                             target_frequency=16000)

        self.per_trial_info[idx]['processing_record']['freq_changed_audio_path'] = output_path.split(os.sep)
        self.per_trial_info[idx]['audio_path'] = output_path.split(os.sep)

    def convert_to_wav_fn(self, idx, output_filename):

        if os.name == "nt":
            # input_path = os.path.join(self.config['root_directory'], *self.per_trial_info[idx]['audio_path'][2:])
            input_path = self.per_trial_info[idx]['audio_path']
        else:
            # input_path = os.path.join(self.config['root_directory'], *self.per_trial_info[idx]['audio_path'][2:])
            input_path = self.per_trial_info[idx]['audio_path']
        output_path = os.path.join(self.config['output_root_directory'], self.config['wav_folder'],
                                   output_filename + "_" + str(self.config['target_frequency']) + ".wav")
        ensure_dir(output_path)

        convert_video_to_wav(input_path=input_path, output_path=output_path,
                             target_frequency=16000)

        self.per_trial_info[idx]['processing_record']['wav_path'] = output_path.split(os.sep)
        self.per_trial_info[idx]['audio_path'] = output_path.split(os.sep)

    def extract_mfcc_fn(self, idx, output_filename, npy_folder):

        input_path = os.path.join(self.config['output_root_directory'], *self.per_trial_info[idx]['audio_path'][-2:])
        output_path = os.path.join(self.config['output_root_directory'], self.config['mfcc_folder'],
                                   output_filename + ".csv")
        ensure_dir(output_path)
        opensmile_config_path = os.path.join(self.config['load_directory'], self.config['mfcc_config_file'])

        if "target_fps" in self.config:
            hop_sec = 1 / self.config['target_fps']
        else:
            hop_sec = 1 / (self.per_trial_info[idx]['fps'] *self.config['multiplier']['mfcc'])

        extract_mfcc(input_path=input_path, output_path=output_path, hop_sec=hop_sec, opensmile_config_path=opensmile_config_path)

        self.per_trial_info[idx]['processing_record']['mfcc_path'] = output_path.split(os.sep)
        self.per_trial_info[idx]['mfcc_path'] = output_path.split(os.sep)

        if self.config['save_npy']:
            filename = os.path.join(npy_folder, "mfcc.npy")

            if not os.path.isfile(filename):
                start_col = 3
                end_col = 42
                feature = "mfcc"
                annotated_index = self.per_trial_info[idx]['mfcc_annotated_index']
                mfcc_matrix = self.compact_audio_feature(output_path, annotated_index, start_col, end_col, feature)
                mfcc_matrix = np.nan_to_num(mfcc_matrix)
                np.save(filename, mfcc_matrix)

    def extract_egemaps_fn(self, idx, output_filename, npy_folder):

        input_path = os.path.join(self.config['output_root_directory'], *self.per_trial_info[idx]['audio_path'][-2:])
        output_path = os.path.join(self.config['output_root_directory'], self.config['egemaps_folder'],
                                   output_filename + ".csv")
        ensure_dir(output_path)

        extract_egemaps(input_path=input_path, output_path=output_path,
                        opensmile_config_path=None, length=self.per_trial_info[idx]['length'], target_frequence=self.per_trial_info[idx]['fps'])

        self.per_trial_info[idx]['processing_record']['egemaps_path'] = output_path.split(os.sep)
        self.per_trial_info[idx]['egemaps_path'] = output_path.split(os.sep)

        if self.config['save_npy']:
            filename = os.path.join(npy_folder, "egemaps.npy")
            if not os.path.isfile(filename):
                start_col = 3
                end_col = 91
                feature = "egemaps"
                annotated_index = self.per_trial_info[idx]['egemaps_annotated_index']
                egemaps_matrix = self.compact_audio_feature(output_path, annotated_index, start_col, end_col, feature)
                egemaps_matrix = np.nan_to_num(egemaps_matrix)
                np.save(filename, egemaps_matrix)

    def extract_logmel_fn(self, idx, output_filename, npy_folder):

        from base.audio import extract_logmel
        input_path = os.path.join(self.config['output_root_directory'], *self.per_trial_info[idx]['audio_path'][-2:])
        output_path = os.path.join(self.config['output_root_directory'], self.config['logmel_folder'],
                                   output_filename + ".npy")
        ensure_dir(output_path)

        if "target_fps" in self.config:
            hop_sec = 1 / self.config['target_fps']
        else:
            hop_sec = 1 / self.per_trial_info[idx]['fps']

        output_path = os.path.join(npy_folder, "logmel.npy")
        extract_logmel(input_path=input_path, output_path=output_path, window_sec=0.96, hop_sec=hop_sec,
                       annotated_idx=self.per_trial_info[idx]['logmel_annotated_index'])



        self.per_trial_info[idx]['processing_record']['logmel_path'] = output_path.split(os.sep)
        self.per_trial_info[idx]['logmel_path'] = output_path.split(os.sep)

        # if self.config['save_npy']:
        #     filename = os.path.join(npy_folder, "logmel.npy")
        #     if not os.path.isfile(filename):
        #         feature = "logmel"
        #         annotated_index = self.per_trial_info[idx]['vggish_annotated_index']
        #         logmel_matrix = self.compact_audio_feature(output_path, annotated_index, feature=feature)
        #
        #         np.save(filename, logmel_matrix)

    def extract_vggish_fn(self, idx, output_filename, npy_folder, model):

        from base.audio import extract_vggish
        input_path = os.path.join(self.config['output_root_directory'], *self.per_trial_info[idx]['audio_path'][-2:])
        output_path = os.path.join(self.config['output_root_directory'], self.config['vggish_folder'],
                                   output_filename + ".csv")
        ensure_dir(output_path)

        if "target_fps" in self.config:
            hop_sec = 1 / self.config['target_fps']
        else:
            hop_sec = 1 / self.per_trial_info[idx]['fps']

        extract_vggish(input_path=input_path, output_path=output_path, window_sec=0.96, hop_sec=hop_sec, model=model, input_size=100)

        self.per_trial_info[idx]['processing_record']['vggish_path'] = output_path.split(os.sep)
        self.per_trial_info[idx]['vggish_path'] = output_path.split(os.sep)

        if self.config['save_npy']:
            filename = os.path.join(npy_folder, "vggish.npy")
            if not os.path.isfile(filename):
                start_col = 0
                end_col = 128
                feature = "vggish"
                annotated_index = self.per_trial_info[idx]['vggish_annotated_index']
                vggish_matrix = self.compact_audio_feature(output_path, annotated_index, start_col, end_col, feature)

                np.save(filename, vggish_matrix)

    def extract_continuous_label_fn(self, idx, npy_folder):

        if not 'continuous_label' in self.per_trial_info[idx]:
            raw_continuous_label = self.load_continuous_label(self.per_trial_info[idx]['continuous_label_path'])

            self.per_trial_info[idx]['continuous_label'] = raw_continuous_label[self.per_trial_info[idx]['annotated_index']]

        if self.config['save_npy']:
            filename = os.path.join(npy_folder, "continuous_label.npy")
            if not os.path.isfile(filename):
                ensure_dir(filename)
                np.save(filename, self.per_trial_info[idx]['continuous_label'])

    def extract_class_label_fn(self, record):
        pass

    def load_continuous_label(self, path, **kwargs):
        raise NotImplementedError

    def compact_audio_feature(self, path, annotated_index, start_col=0, end_col=0, feature=""):

        length = max(annotated_index)

        sep = ","
        if feature == "facial_landmark":
            sep = ","
        elif feature == "vggish":
            sep = ";"

        feature_matrix = pd.read_csv(path, sep=sep, usecols=range(start_col, end_col)).values

        # If the continuous label is longer than the video, then repetitively pad (edge padding) the last element.
        length_difference = length - len(feature_matrix) + 1

        if length_difference > 0:
            feature_matrix = np.vstack(
                (feature_matrix, np.repeat(feature_matrix[-1, :][None, :], length_difference, axis=0)))

        feature_matrix = feature_matrix[annotated_index]

        return feature_matrix

    def compact_facial_image(self, path, annotated_index, extension="jpg"):

        length = len(annotated_index)

        try:
            facial_image_list = get_filename_from_a_folder_given_extension(path + "_aligned", extension)
        except:
            facial_image_list = get_filename_from_a_folder_given_extension(path, extension)

        # If the continuous label is longer than the video, then repetitively pad (edge padding) the last element.
        length_difference = length - len(facial_image_list)
        if length_difference:
            [facial_image_list.extend([facial_image_list[-1]]) for _ in range(length_difference)]

        facial_image_list = list(itemgetter(*annotated_index)(facial_image_list))

        frame_matrix = np.zeros((
            length, self.config['video_size'], self.config['video_size'], 3), dtype=np.uint8)

        for j, frame in enumerate(facial_image_list):
            current_frame = Image.open(frame)
            frame_matrix[j] = current_frame.resize((self.config['video_size'], self.config['video_size']))

        return frame_matrix

    def process_continuous_label(self, continuous_label):
        return list(range(len(continuous_label)))

    def generate_iterator(self):
        return NotImplementedError

    def generate_per_trial_info_dict(self):
        raise NotImplementedError

    def get_video_trim_range(self):
        trim_range = []
        return trim_range

    def get_annotated_index(self, annotated_index):
        return annotated_index

    @staticmethod
    def get_output_filename(**kwargs):

        output_filename = "P{}-T{}".format(kwargs['subject_no'], kwargs['trial_no'])
        return output_filename

    @staticmethod
    def init_dataset_info():
        dataset_info = {
            "trial": [],
            "subject_no": [],
            "trial_no": [],
            "length": [],
            "partition": [],
        }
        return dataset_info



