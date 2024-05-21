import os
import sys
import warnings
import re
import wave
import json
import pandas as pd
import numpy as np
from tqdm import tqdm


class Word(object):
    ''' A class representing a word from the JSON format for vosk speech recognition API '''

    def __init__(self, dict):
        '''
        Parameters:
          dict (dict) dictionary from JSON, containing:
            conf (float): degree of confidence, from 0 to 1
            end (float): end time of the pronouncing the word, in seconds
            start (float): start time of the pronouncing the word, in seconds
            word (str): recognized word
        '''

        self.conf = dict["conf"]
        self.end = dict["end"]
        self.start = dict["start"]
        self.word = dict["word"]

    def to_string(self):
        ''' Returns a string describing this instance '''
        return "{:20} from {:.2f} sec to {:.2f} sec, confidence is {:.2f}%".format(
            self.word, self.start, self.end, self.conf*100)

    def to_output(self):
        return self.start*1000, self.end*1000, self.word, self.conf*100

# Get the transcripts from a wav file.
def transcribe(model, wave_path, KaldiRecognizer):
    import soundfile as sf

    wave_file = wave.open(wave_path)

    rec = KaldiRecognizer(model, wave_file.getframerate())
    rec.SetWords(True)

    f = sf.SoundFile(wave_path)

    results = []

    data = wave_file.readframes(f.frames)
    if rec.AcceptWaveform(data):
        part_result = json.loads(rec.Result())
        results.append(part_result)

    return results

# Extract the transcript from a wav file and format it as a pandas dataframe
def extract_transcript(input_path, output_path, model_path):

    if not os.path.isfile(output_path):

        from vosk import Model, KaldiRecognizer

        # with torch.no_grad():
        model = Model(model_path)
        results = transcribe(model, input_path, KaldiRecognizer)

        transcribed_list = []
        for sentence in results:
            if len(sentence) == 1:
                continue

            for obj in sentence['result']:
                output = Word(obj).to_output()
                transcribed_list.append(output)

        data = None
        if len(transcribed_list) > 0:
            transcript = np.stack(transcribed_list)
            data = transcript

        # exclude void string
        if data is not None:
            exclude_void = []
            for row in data:
                if row[2] != "":
                    exclude_void.append(row)

            data = np.stack(exclude_void)
        df = pd.DataFrame(data, columns=['start', 'end', 'word', 'confidence'])
        df.to_csv(output_path, sep=";", index=False)


def add_punctuation(input_path, output_path, model=None):

    if not os.path.isfile(output_path):
        df = pd.read_csv(input_path, sep=";", header=None, keep_default_na=False)
        record = df.values

        new_record = None
        if len(record) > 1:
            text = [" ".join(record[1:, 2])]

            # with torch.no_grad:
            punctuated_text = model.restore_punctuation(text[0])

            punctuated_text = re.findall(r"[\w']+|[.,!?;]", punctuated_text)

            pointer = 1
            new_record = []
            num_occurrences_hyphen = 0

            # Loop the text, when a single quote is find, split it into three records as [left]['][right].
            for idx, string in enumerate(punctuated_text):

                if num_occurrences_hyphen > 0:
                    num_occurrences_hyphen -= 1
                    continue

                if string in ".,!?;":
                    if pointer == 0:
                        start, end = "0.0", "1.0"
                    else:
                        start = str(float(record[pointer-1, 1]))
                        end = str(float(record[pointer-1, 1]) + 1.0)

                    word = string
                    confidence = "100.0"

                else:
                    raw_word = record[pointer, 2]
                    if string.lower() == raw_word:
                        num_occurrences_single_quote = raw_word.count("\'")
                        if num_occurrences_single_quote > 0:
                            if num_occurrences_single_quote == 1:
                                left, right = string.split("\'")

                                # Left
                                start, end, word, confidence = record[pointer, 0], record[pointer, 1], left, record[pointer, 3]
                                new_record.append([start, end, word, confidence])

                                # Quote
                                start, end, word, confidence = record[pointer, 0], record[pointer, 1], "\'", record[pointer, 3]
                                new_record.append([start, end, word, confidence])

                                string = right
                            else:
                                raise ValueError("More than one single quote appears!")
                    else:
                        raw_word = record[pointer, 2]
                        print(idx, raw_word)
                        num_occurrences_hyphen = raw_word.count("-")

                        if num_occurrences_hyphen == 0:
                            raise ValueError("Unknown case happened!")
                        string = string[0] + record[pointer, 2][1:]

                    start, end, word, confidence = record[pointer, 0], record[pointer, 1], string, record[pointer, 3]
                    pointer += 1

                new_record.append([start, end, word, confidence])
            new_record = np.stack(new_record)

        if new_record is not None:
            exclude_void_str = []
            for row in new_record:
                if row[2] != "":
                    exclude_void_str.append(row)

            new_record = np.stack(exclude_void_str)

        df = pd.DataFrame(data=new_record, index=None, columns=["start", "end", "word", "confidence"])
        df.to_csv(output_path, sep=";", index=False)


def extract_word_embedding(input_path, output_path, tokenizer, bert, max_length=256):
    if not os.path.isfile(output_path):

        df_try = pd.read_csv(input_path, header=None, sep=";")
        if not len(df_try) == 1:

            from torch.utils.data import TensorDataset, DataLoader
            print(input_path)
            df = pd.read_csv(input_path, header=None, sep=";", skiprows=1)
            str_words = [str(word) for word in df.values[:, 2]]
            paragraph = [" ".join(str_words)][0]
            num_tokens = len(df)

            token_ids, token_masks, paragraph = tokenize(paragraph, tokenizer, max_length=max_length)

            dataset = TensorDataset(token_ids, token_masks)
            data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

            token_vecs_sum = calculate_token_embeddings(data_loader, bert)
            bert_features = exclude_padding(token_vecs_sum, token_masks)

            # The indices help to restore the bert feature for a one-to-one correspondence to the input tokenizers.
            idx_intact, idx_target, idx_non_sub_words, idx_grouped_sub_words = get_sub_word_idx(paragraph, tokenizer)

            average_merged_bert_features = average_merge_embeddings(num_tokens, idx_intact, idx_target, bert_features, idx_non_sub_words, idx_grouped_sub_words)
            assert len(average_merged_bert_features) == num_tokens


            combined_df = np.c_[df.values, average_merged_bert_features]
            combined_df = compress_single_quote(combined_df)

            combined_df.to_csv(output_path, sep=";", index=False)


def compress_single_quote(pd_frame):

    i = 0
    tokens = pd_frame[:, 2]
    pd_frame = np.delete(pd_frame, 2, axis=1)

    merged_frame = []

    while i < len(pd_frame) - 1:
        current_frame, current_token = pd_frame[i], tokens[i]
        next_frame, next_token = pd_frame[i+1], tokens[i+1]

        if next_token == "\'":

            if i + 2 <= len(pd_frame) - 1:
                j = i + 2

            else:
                j = i + 1

            token = "".join(tokens[i:j + 1])
            average = np.mean(pd_frame[i:j+1, :], axis=0)

            average = np.insert(average, 2, token)
            merged_frame.append(average)

            i = j + 1

        else:
            current_frame = np.insert(current_frame, 2, current_token)
            merged_frame.append(current_frame)

            i += 1

    merged_frame = np.stack(merged_frame)
    columns = ["start", "end", "word", "confidence", *np.arange(768)]
    merged_df = pd.DataFrame(merged_frame, columns=columns, index=None)


    return merged_df

# def compress_single_quote(ndarray, idx_str_column):
#
#     words = ndarray[:, idx_str_column]
#     num_tokens_with_single_quote = len(words)
#     idx_grouped_single_quote = []
#     for idx, string in enumerate(words):
#         if string == "\'":
#             idx_single_quote = [idx-1, idx, idx+1]
#             idx_grouped_single_quote.append(idx_single_quote)
#
#     num_tokens = num_tokens_with_single_quote - len(idx_grouped_single_quote) * 2
#     num_group = 0
#     num_idx = 0
#     idx_target = []
#     for idxes in idx_grouped_single_quote:
#         idx_target.append(idxes[0] - num_idx + num_group)
#         num_group += 1
#         num_idx += len(idxes)
#
#     idx_intact = list(set(np.arange(num_tokens)).difference(idx_target))
#
#     if len(idx_grouped_single_quote) >= 1:
#         idx_non_single_quote = list(set(np.arange(num_tokens_with_single_quote)).difference(np.hstack(idx_grouped_single_quote)))
#     else:
#         idx_non_single_quote = list(np.arange(num_tokens_with_single_quote))
#
#     idx_non_str_column = list(np.arange(772))
#     idx_non_str_column.pop(idx_str_column)
#     average_merged_matrix = np.zeros((num_tokens, 771))
#
#     ndarray_without_str = ndarray[:, idx_non_str_column]
#     for idx, idxes in enumerate(idx_grouped_single_quote):
#         average = np.mean(ndarray_without_str[idxes, :], axis=0)
#         average_merged_matrix[idx_target[idx]] = average
#
#     average_merged_matrix[idx_intact] = ndarray_without_str[idx_non_single_quote]
#
#     merged_words = []
#     skip = 0
#     for idx, word in enumerate(words):
#         if skip:
#             skip = 0
#             continue
#
#         merged_words.append(word)
#
#         if word == "\'":
#             merged = "".join([words[idx-1], words[idx], words[idx+1]])
#             merged_words = merged_words[:-2]
#             merged_words.append(merged)
#             skip = 1
#
#     columns = ["start", "end", "confidence", *np.arange(768)]
#     merged_df = pd.DataFrame(average_merged_matrix, columns=columns, index=None)
#     merged_df.insert(loc=2, column='word', value=merged_words)
#
#     return merged_df


def average_merge_embeddings(length, idx_intact, idx_target, bert_features, idx_non_sub_words, idx_grouped_sub_words):

    # idx_target = sorted(list(set(np.arange(length)).difference(idx_intact)))
    average_merged_matrix = np.zeros((length, 768), dtype=np.float32)

    for idx, idxes in enumerate(idx_grouped_sub_words):
        average = np.mean(bert_features[idxes], axis=0)
        average_merged_matrix[idx_target[idx]] = average

    average_merged_matrix[idx_non_sub_words] = bert_features[idx_intact]
    return average_merged_matrix


def get_sub_word_idx(paragraph, tokenizer):


    tokenized_input = []
    for sentence in paragraph:
        tokenized_input.extend(tokenizer.tokenize(sentence))

    idx_sub_word = []
    idx_grouped_sub_word = []

    true_idx_word = []
    true_idx_grouped_sub_word = []

    i = 0
    idx_intact = []
    idx_head_of_sub_word = []
    i_token = 0

    while i < len(tokenized_input) - 1:
        j = i+1

        current_token = tokenized_input[i]
        next_token = tokenized_input[j]

        # When the next token is "##xyz" or a hyphen,
        # add it to the group, then proceed to the next token.
        # If that token starts with "##" again, then repeat, else go to the next token.
        #   If this next token is hyphen,, then add the processor, current, and the successor to the group, and repeat.

        # xxx, ##yyy
        if next_token.startswith("##") or next_token == "-":
            idx_head_of_sub_word.append(i)
            idx_sub_word.extend([i, j])

            k = j
            while  k < len(tokenized_input) - 1:
                i = k + 1
                after_next_token = tokenized_input[k+1]

                # xxx, ##yyy, ##zzz
                if after_next_token.startswith("##"):
                    idx_sub_word.extend([k+1])
                    k += 1

                # xxx, ##yyy, -zzz
                elif  after_next_token == "-":

                    # If not at the rear
                    if k + 2 < len(tokenized_input):
                        idx_sub_word.extend([k+1, k+2])
                        k += 2

                    # If at the rear
                    else:
                        idx_sub_word.extend([k + 1])
                        k += 1


                elif k+2 < len(tokenized_input) - 1:
                    after_after_next_token = tokenized_input[k+2]

                    # xxx, -yyy, -zzz
                    if after_after_next_token == "-":
                        idx_sub_word.extend([k + 1, k + 2, k + 3])
                        k += 3

                    # xxx, -yyy, ##zzz
                    elif after_after_next_token.startswith("##"):
                        idx_sub_word.extend([k + 1, k + 2])
                        k += 2
                    else:
                        break

                else:
                    break

            idx_grouped_sub_word.append(idx_sub_word)
            true_idx_grouped_sub_word.append(i_token)
            idx_sub_word = []
            i_token += 1

            if j == len(tokenized_input) - 1:
                break

        else:
            idx_intact.append(i)
            true_idx_word.append(i_token)
            i += 1
            i_token += 1

    assert len(idx_grouped_sub_word) == len(true_idx_grouped_sub_word)
    assert len(idx_intact) == len(true_idx_word)
    return idx_intact, true_idx_grouped_sub_word, true_idx_word, idx_grouped_sub_word



# def get_sub_word_idx(paragraph, tokenizer, bert_features, num_tokens):
#
#     def consecutive(data, stepsize=1):
#         return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
#
#     tokenized_input = []
#     for sentence in paragraph:
#         tokenized_input.extend(tokenizer.tokenize(sentence))
#
#     idx_sub_word = []
#     idx_grouped_sub_word = []
#     for idx, string in enumerate(tokenized_input):
#         if idx < 1:
#             continue
#
#         elif "##" == string[:2]:
#             idx_sub_word.extend([idx])
#
#             if "##" not in tokenized_input[idx - 1]:
#                 idx_sub_word.insert(0, idx - 1)
#
#         elif "-" == string:
#             if "##" not in tokenized_input[idx - 1]:
#                 idx_sub_word = [idx - 1, idx, idx + 1]
#                 idx_grouped_sub_word.append(idx_sub_word)
#                 idx_sub_word = []
#             else:
#                 idx_sub_word.insert(-1, idx)
#
#         else:
#             if len(idx_sub_word) > 0:
#                 if tokenized_input[idx - 1] == "-":
#                     idx_sub_word.extend([idx])
#
#                 idx_grouped_sub_word.append(idx_sub_word)
#                 idx_sub_word = []
#
#     found_double_hyphen = 0
#     if len(idx_grouped_sub_word) > 0:
#         for idx, group in enumerate(idx_grouped_sub_word):
#             if idx == 0:
#                 continue
#             if len(group) == 3:
#                 if tokenized_input[group[1]] == "-" and len(idx_grouped_sub_word[idx-1]) == 3 and tokenized_input[idx_grouped_sub_word[idx-1][1]] == "-" and group[0] == idx_grouped_sub_word[idx-1][2]:
#                     found_double_hyphen = 1
#
#
#     if found_double_hyphen:
#         refined_group = sorted(set(np.hstack(idx_grouped_sub_word)))
#         refined_group = consecutive(refined_group)
#     else:
#         refined_group = idx_grouped_sub_word
#
#     num_group = 0
#     num_idx = 0
#     idx_target = []
#     for idxes in refined_group:
#         idx_target.append(idxes[0] - num_idx + num_group)
#         num_group += 1
#         num_idx += len(idxes)
#
#     if len(refined_group) >= 1:
#         idx_non_sub_word = list(set(np.arange(len(bert_features))).difference(np.hstack(refined_group)))
#     else:
#         idx_non_sub_word = list(np.arange(len(bert_features)))
#
#     idx_intact = list(set(np.arange(num_tokens)).difference(idx_target))
#     return idx_intact, idx_target, idx_non_sub_word, refined_group


def exclude_padding(token_vecs_sum, token_masks):
    bert_features = []
    flag = 0
    for attention_mask, token_vecs in zip(token_masks, token_vecs_sum):
        mask = attention_mask.detach().cpu().numpy()
        idx_of_one = np.where(mask == 1)[0]

        if len(idx_of_one) == len(mask):
            raise ValueError("The sentence is too long, enlarge the token number!")

        mask[0] = 0
        mask[max(idx_of_one)] = 0

        word_indicator = (mask == 1)
        sentence = token_vecs[word_indicator]
        bert_features.append(sentence)

    bert_features = np.vstack(bert_features)
    return bert_features


def calculate_token_embeddings(data_loader, model):
    import torch
    device = torch.device("cuda")
    token_embeddings = []
    for batch in data_loader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch

        # Telling the models not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        try:
            hidden_states = outputs[2]
        except:
            hidden_states = outputs.hidden_states

        hidden_states = torch.stack(hidden_states).permute(1, 2, 0,
                                                           3).detach().cpu().numpy()  # Batch x Token x Layer x Feature
        token_embeddings.append(hidden_states)
    token_embeddings = np.vstack(token_embeddings)

    # Only use the sum of the embeddings from the last 4 layers.
    token_vecs_sum = []
    for token in token_embeddings:
        # Sum the vectors from the last four layers.
        sum_vec = np.sum(token[:, -4:, :], axis=1)

        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)

    return token_vecs_sum


def tokenize(paragraph, tokenizer, max_length=256):
    from nltk import tokenize as tk
    import torch
    paragraph = tk.sent_tokenize(paragraph)

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sentence in paragraph:
        encoded_dict = tokenizer.encode_plus(
            sentence,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_length,  # Pad & truncate all sentences.
            padding ='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            is_split_into_words=False
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks, paragraph


def align_word_embedding(input_path, fps, annotated_idx):
    if os.path.isfile(input_path):
        annotated_time_stamp = 1 / fps * np.array(annotated_idx) * 1000
        df = pd.read_csv(input_path, header=None, sep=";", skiprows=1)
        aligned_embedding = np.zeros((len(annotated_idx), 768), dtype=np.float32)

        for idx, stamp in tqdm(enumerate(annotated_time_stamp), total=len(annotated_time_stamp)):
            embedding = np.zeros((1, 768), dtype=np.float32)
            diff = np.sum(np.asarray((stamp - df.values[:, :2]>0), dtype=int), axis=1)
            idx_nearest = np.where(diff==1)[0]
            if len(idx_nearest) > 0:
                if len(idx_nearest) > 1:
                    idx_nearest = idx_nearest[0]
                embedding = df.values[idx_nearest, 4:]

            aligned_embedding[idx] = embedding
    else:
        aligned_embedding = np.zeros((len(annotated_idx), 768), dtype=np.float32)

    return aligned_embedding









