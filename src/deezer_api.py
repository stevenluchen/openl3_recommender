import requests
import openl3
import librosa
import numpy as np
import os
import re
import tqdm
from datetime import datetime, timedelta
from pydub import AudioSegment

def safe_filename(name):
  return re.sub(r'[\\/:"*?<>|]+', '_', name)

def get_deezer_new_albums(limit=20, country_id=0, days_back=5):
  """
  Pulls new releases from Deezer editorial. Default is Global (ID 0).
  Country IDs are Deezer-specific (e.g., 2 = US, 6 = FR, etc.)
  """
  recent_date_cutoff = datetime.now() - timedelta(days=days_back)

  url = f"https://api.deezer.com/editorial/{country_id}/releases"
  r = requests.get(url)
  if r.status_code != 200:
    raise Exception("Failed to fetch releases.")
  data = r.json()['data']
  
  albums = [{
    'title': album['title'],
    'artist': album['artist']['name'],
    'id': album['id']
  } for album in data[:limit] if datetime.strptime(album['release_date'], '%Y-%m-%d') > recent_date_cutoff]

  print(f"Loaded {len(albums)} albums from Deezer, released in the past {days_back} days.")
  return albums

def get_deezer_tracks_from_album(album_id):
  """
  Extracts individual tracks from album given Deezer album ID. 
  """
  url = f"https://api.deezer.com/album/{album_id}"
  r = requests.get(url)
  if r.status_code != 200:
    raise Exception("Failed to fetch tracks.")
  data = r.json()
  tracks = data.get('tracks', {}).get('data', [])

  return [{
    'title': safe_filename(track['title']),
    'artist': track['artist']['name'],
    'id': track['id'],
    'preview_url': track['preview']
  } for track in tracks if track.get('preview')]

def compile_previewable_tracks(album_list):
  all_tracks = []
  for album in album_list:
    tracks = get_deezer_tracks_from_album(album['id'])
    all_tracks.extend(tracks)
  print(f"Compiled {len(all_tracks)} previewable tracks from {len(album_list)} albums.")
  return all_tracks

def download_and_convert_preview(track, mp3_dir, wav_dir):
  """
  Downloads preview track from Deezer and converts to WAV.
  """
  mp3_path = os.path.join(mp3_dir, f"{track['artist']} - {track['title']}.mp3")
  wav_path = os.path.join(wav_dir, f"{track['artist']} - {track['title']}.wav")

  r = requests.get(track['preview_url'])
  if r.status_code != 200:
    raise Exception("Failed to download preview.")

  with open(mp3_path, 'wb') as f:
    f.write(r.content)

  audio = AudioSegment.from_file(mp3_path, format="mp3")
  audio = audio.set_frame_rate(48000).set_channels(1)  
  audio.export(wav_path, format="wav")
  return wav_path

def compute_openl3_embedding(wav_path, emb_path, model):
  """
  Given path to a .wav file, compute OpenL3 embeddings and save in local directory. 
  """
  audio, sr = librosa.load(wav_path, sr=None, mono=True)
  emb, _ = openl3.get_audio_embedding(audio, sr,
                                      content_type="music",
                                      embedding_size=512,
                                      model=model,
                                      verbose=0)
  emb_mean = np.mean(emb, axis=0)
  emb_path = os.path.join(emb_path, os.path.basename(wav_path).replace(".wav", "_embedding.npy"))
  np.save(emb_path, emb_mean)
  return emb_mean

def run_full_deezer_pipeline(album_list, mp3_dir, wav_dir, track_dir):
  all_tracks = compile_previewable_tracks(album_list)
  candidate_embeddings = []
  valid_tracks = []

  model = openl3.models.load_audio_embedding_model(input_repr="mel128", 
                                                   content_type="music",
                                                   embedding_size=512)
  
  for track in tqdm(all_tracks, desc='Computing embeddings'):
    wav_path = download_and_convert_preview(track, mp3_dir, wav_dir)
    if not wav_path:
      continue
    emb = compute_openl3_embedding(wav_path, track_dir, model)
    candidate_embeddings.append(emb)
    valid_tracks.append(track)

  if not candidate_embeddings:
    print("No valid previews were processed.")
    return []
