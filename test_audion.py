from moviepy.editor import concatenate_audioclips, AudioFileClip

# def concatenate_audio_moviepy(audio_clip_paths, output_path):
#     clips = [AudioFileClip(c) for c in audio_clip_paths]
#     final_clip = concatenate_audioclips(clips)
#     final_clip.write_audiofile(output_path)

# concatenate_audio_moviepy('./history','output.mp3') :


snd = AudioFileClip("./history/voice1.mp3", fps = 44100)
second_reader = snd.coreader()
second_reader.close()
snd.close()