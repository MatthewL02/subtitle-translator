import torch 
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from tqdm.auto import tqdm
import os
import re
from datetime import timedelta
from datetime import datetime
from tqdm import tqdm
import time
import sys
import intel_extension_for_pytorch as ipex
from IPython.display import Audio
from pprint import pprint
import subprocess

#argument format set at start: filename 

#variables set at start

#device, xpu in this case
device = torch.device('xpu')
filename = sys.argv[1]

#loads translation model according to program, default is facebook mbart-large-50-many-to-one-mmt
translationmodel="facebook/mbart-large-50-many-to-one-mmt"

#loads ASR method according to program, default is whisper-large-v3
transcriptionmodel="openai/whisper-large-v3"

#defines whisper transcription language
transcription_language="chinese"

#defines source and target language for translation
source_language = "zh_CN"
target_language = "en_XX"

print("Testing, this version "+source_language+" --> "+target_language)

#defines max characters, mutable--for instance if you're translating to Chinese you don't want as many~
max_character=84

#defines device used:

device = torch.device('xpu')

print(str(device)+" is being used")

#tensor type
torch_dtype = torch.bfloat16

print("Using "+str(torch_dtype)+" tensors")

#filename splitter

def strip_file_format(file_path):
    """
    Strip the file format from the end of the file string.

    Parameters:
    - file_path (str): The file path string.

    Returns:
    - str: The file name without the extension.
    """
    # Split the file path into base name and extension
    base_name, _ = os.path.splitext(file_path)

    return base_name

# Example usage:
file_path = filename
file_name = strip_file_format(file_path)
filename = file_name

#pulls movie file
#name of movie file, set as argument when run in shell
command = "ffmpeg -y -i " + file_path + " -ab 160k -ac 2 -ar 16000 -vn " + filename+".wav"
subprocess.call(command, shell=True)

print("Audio ripped and at " + filename+".wav")

print("Finding timestamps")
SAMPLING_RATE = 16000

torch.set_num_threads(1)

USE_ONNX = False # change this to True if you want to test onnx model

  
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=USE_ONNX)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

wav = read_audio(filename+".wav", sampling_rate=SAMPLING_RATE)
# get speech timestamps from full audio file
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
print(str(len(speech_timestamps))+" Timestamps found")

print("Processing Audio")
master_cuts = []
i = 0
while i+1 < len(speech_timestamps):
    #merges all gaps <2s 
    if speech_timestamps[i+1]['start']-speech_timestamps[i]['end'] < 32000:
        master_cuts.append([str(timedelta(seconds=speech_timestamps[i]['start']/16000)),
                            str(timedelta(seconds=speech_timestamps[i+1]['end']/16000))])
        i +=1
        
    else:
        master_cuts.append([str(timedelta(seconds=speech_timestamps[i]['start']/16000)),
                            str(timedelta(seconds=speech_timestamps[i]['end']/16000))])
    i += 1

master_cuts.append([str(timedelta(seconds=speech_timestamps[i]['start']/16000)),
            str(timedelta(seconds=speech_timestamps[i]['end']/16000))])
        
print("Audio processed, there are "+str(len(master_cuts))+" audio chunks in your file")

print("Proceeding to transcription")



#opensrt generator
def opensrtify(subtitles,output_file):
    with open(output_file, 'w') as f:
        # Iterate through each subtitle and write it to the SRT file
        for i, subtitle in enumerate(subtitles, start=1):
            start_time, end_time, text = subtitle
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")
    print(f"SRT file '{output_file}' has been created successfully.")
    """
    Write subtitles data from a list of lists to an SRT file.

    Parameters:
    - subtitles (list): List of lists containing subtitles data.
                        Each sublist should contain three elements: start time, end time, and subtitle text.
    - output_file (str): Path to the output SRT file.
    """
    return()

print("Clearing GPU cache")
torch.xpu.empty_cache()

#master subtitle loading file
intermediate_subtitle = []

print("Loading transcription model "+transcriptionmodel)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    transcriptionmodel, torch_dtype=torch_dtype,  use_safetensors=True
    )

processor = AutoProcessor.from_pretrained(transcriptionmodel)

model.to(device)

pipe = pipeline("automatic-speech-recognition", model=transcriptionmodel,device=device,generate_kwargs={"language":transcription_language,"task":"transcribe"},return_timestamps=True,batch_size=16)

print("Proceeding to transcription, using "+transcriptionmodel+" and device "+str(device))
print("This may take some time")


counter = 0
length = len(master_cuts)


for i in tqdm(master_cuts):
    command = "ffmpeg -y -ss " + i[0] + " -to " + i[1] + " -i " + filename + ".wav -c copy tmp1.wav"
    subprocess.call(command, shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
    subtitletmp = pipe("tmp1.wav")['text']
    intermediate_subtitle.append([i[0],i[1],subtitletmp])
    counter += 1


print("Transcription finished, moving to translation")

def parse_timestamp(timestamp):
    time_format = '%H:%M:%S.%f'
    time_obj = datetime.strptime(timestamp, time_format)
    total_seconds = (time_obj.hour * 3600) + (time_obj.minute * 60) + time_obj.second + (time_obj.microsecond / 1e6)
    return total_seconds
    
def split_subtitle_text(start_time, end_time, subtitle_text):
    # Parse timestamps into floating-point numbers representing seconds
    start_seconds = parse_timestamp(start_time)
    end_seconds = parse_timestamp(end_time)

    # Define the maximum length for each chunk
    max_chunk_length = 84

    duration = end_seconds - start_seconds

    pattern = r'(?<=[,.:;!?])' 
    chunks = re.split(pattern, subtitle_text)


    current_chunk = ''
    current_chunk_length = 0
    current_start_time = start_time
    chunked_subtitles = []

    for chunk in chunks:
        if current_chunk_length + len(chunk) <= max_chunk_length:
            current_chunk += chunk
            current_chunk_length += len(chunk)
        else:
            chunk_end_time = datetime.strptime(current_start_time, '%H:%M:%S.%f') + timedelta(seconds=duration * (current_chunk_length / len(subtitle_text)))
            chunked_subtitles.append([current_chunk.strip(), current_start_time, chunk_end_time.strftime('%H:%M:%S.%f')])
            current_chunk = chunk
            current_chunk_length = len(chunk)
            current_start_time = chunk_end_time.strftime('%H:%M:%S.%f')
    if current_chunk.strip():
        chunk_end_time = datetime.strptime(current_start_time, '%H:%M:%S.%f') + timedelta(seconds=duration * (current_chunk_length / len(subtitle_text)))
        chunked_subtitles.append([current_start_time, chunk_end_time.strftime('%H:%M:%S.%f'),current_chunk.strip()])




    return chunked_subtitles


def remove_repeated_sentences(transcription):
    lines = transcription.split('\n')
    cleaned_lines = []
    seen_sentences = set()

    for line in lines:
        if '\u4e00' <= line <= '\u9fff':
            if line not in seen_sentences:
                cleaned_lines.append(line)
                seen_sentences.add(line)
        else:
            if line not in seen_sentences:
                cleaned_lines.append(line)
                seen_sentences.add(line)
    cleaned_transcription = '\n'.join(cleaned_lines)

    return cleaned_transcription

print("Cleaning up noise...")
intermediate_subtitle_1 = []
for i in intermediate_subtitle:
    intermediate_subtitle_1.append({"text":remove_repeated_sentences(i[2])})

print("Clearing GPU cache")
torch.xpu.empty_cache()

#cleanup for subtitles
def remove_special_characters(text):
    pattern = r'\{|\}|<\|.*?\|>'
    cleaned_text = re.sub(pattern, '', text)

    return cleaned_text

print("Loading translation model " + translationmodel)

#translation pipeline

tokenizer = AutoTokenizer.from_pretrained(translationmodel,src_lang=source_language,tgt_lag=target_language)
model = AutoModelForSeq2SeqLM.from_pretrained(translationmodel,max_length=1000,torch_dtype=torch_dtype)
model.to(device)
print("Translation model loaded")
translationpipe = pipeline(model=model,device=device,tokenizer=tokenizer,batch_size=16,task="translation_zh_to_en",src_lang=source_language,tgt_lang=target_language)



print("This may take some time")
final_subtitle=[]

counter = 0
for i in tqdm(intermediate_subtitle_1):
    final_subtitle.append([intermediate_subtitle[counter][0],intermediate_subtitle[counter][1],remove_repeated_sentences(str(translationpipe(i['text'])))])
    counter += 1


print("Translation complete, writing to srt file")

srtimportprep = []

for i in final_subtitle:
    if len(i[2]) < max_character:
        srtimportprep.append(i)
    else:
        srtimportprep += split_subtitle_text(i[0],i[1],i[2])

print(srtimportprep)
opensrtify(srtimportprep,filename)
