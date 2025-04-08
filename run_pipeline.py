from src.audio_utils import *
from src.deezer_api import *
from src.track_selection import *
import os

def main():
  # define directory names
  base_dir = "/data/"
  mp3_seed_dir = os.path.join(base_dir, "mp3_seeds")
  wav_seed_dir = os.path.join(base_dir, "wav_seeds")
  emb_seed_dir = os.path.join(base_dir, "seed_embeddings")
  mp3_track_dir = os.path.join(base_dir, "mp3_tracks")
  wav_track_dir = os.path.join(base_dir, "wav_tracks")
  emb_track_dir = os.path.join(base_dir, "track_embeddings")

  # process all MP3s in the seeds directory
  for filename in os.listdir(mp3_dir):
    if filename.endswith(".mp3"):
      song_name = os.path.splitext(filename)[0]
      mp3_path = os.path.join(mp3_dir, filename)
      wav_path = os.path.join(wav_dir, song_name + ".wav")
      emb_path = os.path.join(emb_seed_dir, song_name + "_embedding.npy")

      process_seed_track(mp3_path, wav_path, emb_path)

  seed_embs, seed_names = load_embeddings_from_dir(emb_seed_dir)
  print(seed_names) # ensure seeds are loaded correctly

  # get all new albums and their songs from the past week
  albums = get_deezer_new_albums(days_back = 7)

  # saves .mp3, .wav. and embeddings for new music locally as well
  run_full_deezer_pipeline(albums, mp3_track_dir, wav_track_dir, emb_track_dir)

  track_embs, track_names = load_embeddings_from_dir(emb_track_dir)
  similar_tracks = find_similar_tracks(track_embs, track_names, seed_embs, seed_names, top_k = 10)
  print(*similar_tracks, sep='\n')

  directories = [mp3_track_dir, wav_track_dir, emb_track_dir]
  clear_directory_contents(directories)