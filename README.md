# subtitle-translator
An implementation of an AI-based subtitle translator, utilizing different models for transcription and translation into multiple languages.

Subtitle-translator is designed for easy translation of subtitles between languages, even where subtitle files may not exist. 

Presently the only argument presented is for a single video file, which will have a multilingual subtitle generated for it. 

Overall functionality:
1) ffmpeg is used to generate an audio file from the requisite video file.
2) Silero VAD is utilized to find all excerpts that are believed to contain distinct audio content.
3) These excerpts are excised with ffmpeg and transcribed using Whisper.
4) The transcriptions are filtered to remove extraneous material, particularly the repeating sentences Whisper seems rather fond of putting in.
5) These transcriptions are then translated using a separate LLM.
6) Utilizing the timing data found during the initial transcriptions, a SRT file is generated for the video file that breaks up phrases as appropriate [particularly cogent for Chinese to English as the Chinese subtitles have significantly fewer characters in them than English].


Current main issues: Whisper-v3 is proving very prone to hallucinations. MADLAD appears to be lacking as a translator, so I'm working on a Yi-6B finetune for Chinese-to-English subtitle translation specifically. Thus far responses are good, but erratic, after training on a dataset of Chinese and English bilingual subtitles. 

IMPORTANT NOTES:
This software has specific modules addressed utilizing Intel's pytorch extension, rather than CUDA or ROCM, as I own an Arc A770. If you intend to use this software on a non-Intel GPU or without installing the rather finicky Intel pytorch library, you will need to change the torch.device and remove the 'import intel_extension_for_pytorch' and replace with your own device as appropriate. If I complete this to my satisfaction I will release cpu-only and Nvidia versions as well, as it doesn't require much more than changing a few lines of code. 

