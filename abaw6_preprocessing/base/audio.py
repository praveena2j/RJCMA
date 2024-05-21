
import sys
import os
import subprocess

import numpy as np

import opensmile

from base.utils import edit_file

try:
  import soundfile as sf

  def wav_read(wav_file):
    wav_data, sr = sf.read(wav_file, dtype='int16')
    return wav_data, sr

except ImportError:

  def wav_read(wav_file):
    raise NotImplementedError('WAV file reading requires soundfile package.')


def convert_video_to_wav(input_path, output_path, target_frequency=16000):

    # ffmpeg command to execute
    # -ac 1 for mono, -ar 16000 for sample rate 16k, -q:v 0 for keeping the quality.
    ffmpeg_command = "ffmpeg -i {input_path} -ac 1 -ar {frequency}  -q:v 0 -f wav {output_path}".format(
        input_path=input_path, output_path=output_path, frequency=target_frequency)

    full_command = "export PATH={conda_path}/bin:$PATH && {ffmpeg_command}".format(conda_path=sys.exec_prefix,
                                                                                   ffmpeg_command=ffmpeg_command)
    # execute if the output does not exist
    if not os.path.isfile(output_path):
        subprocess.call(full_command, shell=True)


def change_wav_frequency(input_path, output_path, target_frequency=16000):
    # Change the sampling frequency of the input to the target_frequency.

    ffmpeg_command = "ffmpeg -i {input_path} -ar {frequency} {output_path}".format(
        input_path=input_path, output_path=output_path, frequency=target_frequency)

    # os.system("export PATH=/home/zhangsu/anaconda3/envs/pre/bin:$PATH && printf '%s\n' $PATH && ffmpeg")

    full_command = "export PATH={conda_path}/bin:$PATH && {ffmpeg_command}".format(conda_path=sys.exec_prefix,
                                                                                   ffmpeg_command=ffmpeg_command)

    if not os.path.isfile(output_path):
        subprocess.call(full_command, shell=True)


def extract_mfcc(input_path, output_path, window_sec=0.025, hop_sec=0.01, opensmile_config_path=None):

    if os.path.isfile(output_path):
        return

    line_window_sec = "frameSize = " + str(window_sec) + "\n"
    line_hop_sec = "frameStep = " + str(hop_sec) + "\n"

    edit_file(opensmile_config_path, line_numbers=[68, 69], new_strings=[line_window_sec, line_hop_sec])

    smile = opensmile.Smile(
        feature_set=opensmile_config_path,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        logfile="smile.log"
    )

    mfcc_df = smile.process_file(input_path)
    mfcc_df.to_csv(output_path)


def extract_egemaps(input_path, output_path, opensmile_config_path, length, target_frequence):

    if os.path.isfile(output_path):
        return

    starts = np.arange(length) / target_frequence
    ends = starts + 1
    files = [input_path for i in range(length)]


    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        logfile="smile.log"
    )

    egemaps_df = smile.process_files(files=files, starts=starts, ends=ends)
    egemaps_df.to_csv(output_path)


def extract_vggish(input_path, output_path, window_sec, hop_sec, model, input_size=500):

    if os.path.isfile(output_path):
        return

    from base.vggish import vggish_input
    examples_batch = vggish_input.wavfile_to_examples(input_path, window_sec=window_sec, hop_sec=hop_sec)

    examples_segment = []
    examples_output = []
    if len(examples_batch) > input_size:
        num_segments = len(examples_batch) // input_size
        for i in range(num_segments):
            start = i * input_size
            end = (i + 1) * input_size
            if i == num_segments - 1:
                end = len(examples_batch)
            examples_segment.append(examples_batch[start:end])
    else:
        examples_segment = [examples_batch]

    for example in examples_segment:
        vggish_feature = model.forward(example)
        examples_output.append(vggish_feature.cpu().detach().numpy())

    examples_output = np.vstack(examples_output)
    np.savetxt(output_path, examples_output, delimiter=";")


def extract_logmel(
        input_path,
        output_path,
        window_sec=0.025,
        hop_sec=0.01,
        annotated_idx=[]
):

    if not os.path.isfile(output_path):
        from base.vggish import vggish_input
        logmel_matrix = vggish_input.wavfile_to_examples(input_path, window_sec=window_sec, hop_sec=hop_sec)

        # If the continuous label is longer than the video, then repetitively pad (edge padding) the last element.
        length = max(annotated_idx)
        length_difference = length - len(logmel_matrix) + 1

        if length_difference > 0:
            logmel_matrix = np.vstack(
                (logmel_matrix, np.repeat(logmel_matrix[-1, :][None, :], length_difference, axis=0)))

        logmel_matrix = np.array(logmel_matrix, dtype=np.float16)[annotated_idx]

        np.save(output_path, logmel_matrix)