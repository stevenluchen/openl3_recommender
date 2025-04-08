from pydub import AudioSegment
import os
import librosa
import openl3
import numpy as np

def process_seed_track(mp3_path, wav_path, emb_path,
                       input_repr="mel128", content_type="music", embedding_size=512):
  """
  Converts MP3 to WAV and computes OpenL3 embedding. Saves WAV and .npy embedding to specified paths.
  Used for processing seeds
  """

  audio = AudioSegment.from_file(mp3_path, format="mp3")
  audio = audio.set_frame_rate(48000).set_channels(1)  
  audio.export(wav_path, format="wav")
  print(f"Converted to WAV: {os.path.basename(mp3_path)}")

  audio_data, sr = librosa.load(wav_path, sr=None, mono=True)
  emb, _ = openl3.get_audio_embedding(audio_data, sr,
                                      input_repr=input_repr,
                                      content_type=content_type,
                                      embedding_size=embedding_size)
  emb_mean = np.mean(emb, axis=0)

  np.save(emb_path, emb_mean)
  print(f"Saved embedding: {os.path.basename(emb_path)}")

def load_embeddings_from_dir(embedding_dir):
  """
  Given path to a directory, load all .npy files into the envionment.
  """
  embeddings = []
  names = []

  for fname in os.listdir(embedding_dir):
    if fname.endswith(".npy"):
      path = os.path.join(embedding_dir, fname)
      emb = np.load(path)
      embeddings.append(emb)
      name = fname.replace("_embedding.npy", "")
      names.append(name)

  print(f"Loaded {len(embeddings)} embeddings.")
  return embeddings, names

def clear_directory_contents(dirs, extensions=(".mp3", ".wav", ".npy")):
  for dir_path in dirs:
    if os.path.exists(dir_path):
      deleted = 0
      for fname in os.listdir(dir_path):
        if fname.endswith(extensions):
          os.remove(os.path.join(dir_path, fname))
          deleted += 1
      print(f"Deleted {deleted} files from {dir_path}")
    else:
      print(f"Directory not found: {dir_path}")