from pydub import AudioSegment
import os
import random

folder_path= 'fold10'
clean_audio_path = 'clean_testset_wav'
urban_sound_path = f'UrbanSound8K/audio/{folder_path}'

noisy_output_path = f'noisy_data/{folder_path}'
os.makedirs(noisy_output_path, exist_ok=True)

clean_audio_files = os.listdir(clean_audio_path)
print(clean_audio_files)
urban_sound_noise_files = [f for f in os.listdir(urban_sound_path) if f.endswith(".wav")]

num_noise_additions = 10  # Adjust this as needed
for clean_audio_file in clean_audio_files:
    print("clean_audio_file ", clean_audio_file)
    clean_audio = AudioSegment.from_wav(os.path.join(clean_audio_path, clean_audio_file))
    noise_length = 0
    noisy_audio = clean_audio
    for i in range(num_noise_additions):
        print("ITERN ", i)
        noise_file = random.choice(urban_sound_noise_files)
        noise_audio = AudioSegment.from_wav(os.path.join(urban_sound_path, noise_file))
        noise_length = noise_length + len(noise_audio)
        if clean_audio.frame_rate != noise_audio.frame_rate:
            noise_audio = noise_audio.set_frame_rate(clean_audio.frame_rate)
        noise_audio = noise_audio - 5
        noisy_audio = noisy_audio.overlay(noise_audio, loop=True)
        if noise_length > len(clean_audio):
            break
    output_file = os.path.join(noisy_output_path, f'noisy_{clean_audio_file}')
    noisy_audio.export(output_file, format="wav")

print("Noisy dataset creation complete.")
