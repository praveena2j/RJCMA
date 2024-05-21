import os
import torch

emotion = ["Arousal", "Valence"]
task = "VA"
video_size = 128
margin_size = 12
VIDEO_EMBEDDING_DIM = 512
MFCC_DIM = 39
EGEMAPS_DIM = 88
VGGISH_DIM = 128
BERT_DIM = 768
VIDEO_TEMPORAL_DIM = 128
MFCC_TEMPORAL_DIM = 32
VGGISH_TEMPORAL_DIM = 32
BERT_TEMPORAL_DIM = 512

config = {
    "device": torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),

    "annotation_folder": "6th_ABAW_Annotations",

    "crop_align_face": 1, # Always specify to 1 to use
    "use_mtcnn": 0, # The author already cropped the faces, no need to do it ourselves.
    "load_cropped_aligned_facial_image": 1, # Therefore we load them directly.
    "cropped_aligned_folder": "cropped_aligned", # Corresponding to the exact same folder from ABAW authors.
    "video_size": video_size, # These are for face detection and cropping. Not used since we already have it from ABAW authors.
    "margin_size": margin_size,
    "landmark": {
        "landmark_number": 68,
        "output_image_size": video_size,
        "margin_size": margin_size
    },

    "extract_cnn": 0, # Not used. Directly feed CNN instead of video to the model can greatly boost efficiency, but sacrifice effectiveness.
    "cnn_folder": "cnn",
    "cnn": {
        "model": "ires100_75",
        "state_dict": "model_state_dict.pth",
        "batch_size": 32,
        "image_size": 48,
        "crop_size": 40,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5]
    },

    # Corresponding to the three sub-tracks of ABAW contest.
    "AU_label_cols": ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"],
    "VA_label_cols": ["valence", "arousal"], # This is what we used.
    "EXPR_label_cols": ["Neutral"],

    # Generate txt labels to npy format.
    "extract_continuous_label": 1,

    # Not used, because we directly generate 16K hz wav from raw videos.
    "change_audio_frequency": 0,
    "load_freq_changed_audio": 0,
    "freq_changed_audio_folder": "freq_changed_audio",
    "target_frequency": 16000,

    # This is used to convert mp4 to 16K mono wav.
    "convert_to_wav": 1,
    "wav_folder": "wav",

    "extract_mfcc": 0, # Extract MFCC.
    "mfcc_folder": "mfcc",
    "mfcc_config_file": "opensmile_mfcc.conf",

    "extract_egemaps": 0, # This one can be really slow to extract.
    "egemaps_folder": "egemaps",

    "extract_logmel": 1, # Not used, because it's too large
    "logmel_folder": "logmel",

    "extract_vggish": 1, # Used to extract vggish
    "vggish_folder": "vggish",

    "extract_transcript": 1, # Use Vosk to transcribe the video with timestamp.
    "transcript_folder": "transcript",

    "add_punctuation": 1, # Then add punctuation on the extracted transcripts.
    "punctuation_folder": "punctuation",

    "extract_word_embedding": 1, # Then extract bert
    "word_embedding_folder": "word_embedding",

    "align_word_embedding": 1, # Finally populate the bert feature according to the timestamps
    "aligned_word_embedding_folder": "aligned_word_embedding",

    "save_npy": 1, # Save all extracted feature to npy

    "emotion_dimension": {
        "AU": 12,
        "EXPR": 7,
        "VA_V": 1,
        "VA_A": 1
    },

    # The root path of your dataset
    # It should include three sub folders coming from ABAW authors: `5th_ABAW_Annotations`, `cropped_aligned`, `raw_video`.
    # Inside `5th_ABAW_Annotations`, there should be three folders, each contains the labels for a sub-track.
    # For the Valence-arousal (VA) one, I don't remember if it was named VA by the ABAW authors. If not, name it as VA by yourself. My code will read it by `VA`.
    # Inside folder `VA`, rename the two sub-folders to 'train' and 'validate'. My code will load them, do things, and generate the partition information accordingly.

    "root_directory": "/misc/scratch11/6th-ABAW_Competition_2024/6th-ABAW_Competition_23Feb2024/ABAW6",

    # Set to the same as root_directory so that the intermediate data will also be saved in here.
    "output_root_directory": "/misc/scratch11/6th-ABAW_Competition_2024/6th-ABAW_Competition_23Feb2024/ABAW6_128",

    # Put all the model state dict here, such as Vosk, and opensmile config file.
    "load_directory": "/misc/scratch11/abaw5_preprocessing/load",

    # The VA task.
    "task": task,

    # Raw videos from the ABAW authors.
    "raw_data_folder": "raw_videos",

    # All the npys go here.
    "npy_folder": "compacted_48",

    "frequency": {
        "video": None,
        "continuous_label": None,
        "mfcc": None,
        "egemaps": None,
        "vggish": None,
        "logmel": None,
        "bert": None,
    },

    "multiplier": {
        "video": 1,
        "cnn": 1,
        # "AU_continuous_label": 1,
        # "EXPR_continuous_label": 1,
        "VA_continuous_label": 1,
        "continuous_label": 1,
        "mfcc": 1,
        "egemaps": 1,
        "vggish": 1,
        "logmel": 1,
        "bert": 1,
    },

    "feature_dimension": {
        "video": (48, 48, 3),
        "cnn": (512,),
        "AU_continuous_label": (12,),
        "EXPR_continuous_label": (1,),
        "VA_continuous_label": (1,),
        "continuous_label": (1,),
        "mfcc": (39,),
        "egemaps": (88,),
        "vggish": (128,),
        "logmel": (96, 64),
        "bert": (768,)
    },

    "speech_model": {
        # Download it from https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
        # Then put it in the  folder "load"
        "path": "/misc/scratch11/abaw5_preprocessing/load/vosk-model-en-us-0.22"
    },
}
