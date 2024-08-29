import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
from intents import predict_intent
from llama_model import llama_response
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # The last time a transcription was printed.
    last_transcription_time = datetime.utcnow()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded.\n")
    audio_data = b''
    i=0
    while True:
        try:
            now = datetime.utcnow()

            if not data_queue.empty():
                phrase_complete = False

                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                    audio_data = b''

                # phrase_time = now

                audio_data = audio_data + b''.join(data_queue.queue)
                data_queue.queue.clear()

                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                #result = audio_model.transcribe(audio_np, fp16=False)
                text = result['text'].strip()

                text=text.replace("Thank you for watching","").replace("Thanks for watching","")

                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
                print('', end='', flush=True)

                last_transcription_time = now  # Update the last transcription time
                
                i+=1
                print(i)
            elif i>=1:
                if now - last_transcription_time > timedelta(seconds=7):
                    # print('timer')
                    break  # Exit the loop after 5 seconds of inactivity

                sleep(0.25)

        except KeyboardInterrupt:
            break

    print("\n\nFinal Transcription:")
    # print(transcription)
    final_transcription = ' '.join(transcription)
    intent=predict_intent(final_transcription)
    final_response=llama_response(intent,final_transcription)
    print(final_response)


if __name__ == "__main__":
    main()

# import argparse
# import os
# import numpy as np
# import speech_recognition as sr
# import whisper
# import torch
# from datetime import datetime, timedelta
# from queue import Queue
# from time import sleep
# from sys import platform
# from intents import predict_intent
# from llama_model import llama_response

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", default="medium", help="Model to use",
#                         choices=["tiny", "base", "small", "medium", "large"])
#     parser.add_argument("--non_english", action='store_true',
#                         help="Don't use the english model.")
#     parser.add_argument("--energy_threshold", default=1000,
#                         help="Energy level for mic to detect.", type=int)
#     parser.add_argument("--record_timeout", default=2,
#                         help="How real time the recording is in seconds.", type=float)
#     parser.add_argument("--phrase_timeout", default=3,
#                         help="How much empty space between recordings before we "
#                              "consider it a new line in the transcription.", type=float)
#     if 'linux' in platform:
#         parser.add_argument("--default_microphone", default='pulse',
#                             help="Default microphone name for SpeechRecognition. "
#                                  "Run this with 'list' to view available Microphones.", type=str)
#     args = parser.parse_args()

#     # The last time a recording was retrieved from the queue.
#     phrase_time = None
#     # Thread safe Queue for passing data from the threaded recording callback.
#     data_queue = Queue()
#     # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
#     recorder = sr.Recognizer()
#     recorder.energy_threshold = args.energy_threshold
#     recorder.dynamic_energy_threshold = False

#     if 'linux' in platform:
#         mic_name = args.default_microphone
#         if not mic_name or mic_name == 'list':
#             print("Available microphone devices are: ")
#             for index, name in enumerate(sr.Microphone.list_microphone_names()):
#                 print(f"Microphone with name \"{name}\" found")
#             return
#         else:
#             for index, name in enumerate(sr.Microphone.list_microphone_names()):
#                 if mic_name in name:
#                     source = sr.Microphone(sample_rate=16000, device_index=index)
#                     break
#     else:
#         source = sr.Microphone(sample_rate=16000)

#     model = args.model
#     if args.model != "large" and not args.non_english:
#         model = model + ".en"
#     audio_model = whisper.load_model(model)

#     record_timeout = args.record_timeout
#     phrase_timeout = args.phrase_timeout

#     with source:
#         recorder.adjust_for_ambient_noise(source)

#     def record_callback(_, audio: sr.AudioData) -> None:
#         data = audio.get_raw_data()
#         data_queue.put(data)

#     recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

#     print("Model loaded.\n")
    
#     while True:
#         transcription = ['']
#         audio_data = b''
#         last_transcription_time = datetime.utcnow()
#         i = 0

#         while True:
#             try:
#                 now = datetime.utcnow()

#                 if not data_queue.empty():
#                     phrase_complete = False

#                     if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
#                         phrase_complete = True
#                         audio_data = b''

#                     phrase_time = now

#                     audio_data = audio_data + b''.join(data_queue.queue)
#                     data_queue.queue.clear()

#                     audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
#                     result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
#                     text = result['text'].strip()

#                     text = text.replace("Thank you for watching", "").replace("Thanks for watching", "")

#                     if phrase_complete:
#                         transcription.append(text)
#                     else:
#                         transcription[-1] = text

#                     os.system('cls' if os.name == 'nt' else 'clear')
#                     for line in transcription:
#                         print(line)
#                     print('', end='', flush=True)

#                     last_transcription_time = now  # Update the last transcription time
                    
#                     i += 1
#                     print(i)
#                 elif i >= 1:
#                     if now - last_transcription_time > timedelta(seconds=2):
#                         # Perform intent classification and llama response generation
#                         final_transcription = ' '.join(transcription)
#                         intent = predict_intent(final_transcription)
#                         final_response = llama_response(intent, final_transcription)
#                         print("\n\nFinal Transcription:")
#                         print(final_transcription)
#                         print("\nLlama Response:")
#                         print(final_response)
#                         break  # Break out to reset the loop

#                     sleep(0.25)

#             except KeyboardInterrupt:
#                 break

# if __name__ == "__main__":
#     main()

# user_history.append({
#                             "User": final_transcription,
#                             "Model": model_response
#                              })
#                         break