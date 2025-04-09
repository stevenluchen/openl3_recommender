# Audio Embedding-Based Music Recommender

This project is a deep audio embedding pipeline that helps you discover new music based on your personal taste - no genre tags, no popularity bias, just the sound.

Platforms like Spotify and Apple Music lean heavily on collaborative filtering and metadata. This project instead uses **pure audio similarity** - so if you like the vibe of a track, this tool will help you find more like it, regardless of popularity, genre, or tags.

Using [OpenL3](https://github.com/marl/openl3) embeddings and the [Deezer API](https://developers.deezer.com/api), this tool pulls recent releases, computes 30-second preview embeddings, and recommends songs that are *sonically similar* to a set of user-provided seed tracks. 

---

## Features

- Extracts 30s previews from newly released tracks using Deezer API
- Computes OpenL3 audio embeddings from raw waveform
- Uses PCA + cosine similarity to compare songs in reduced embedding space
- Enforces diversity constraints (per-artist caps, multi-seed coverage)
- Outputs a curated list of tracks and similarity score to seed tracks

---

## Setup Instructions

### 1. Clone this repo

```bash
git clone https://github.com/stevenluchen/openl3_recommender
```

### 2. Acquire .mp3 files of 2-5 seed songs

Do this by whatever means are convenient to the user (I do not condone piracy or other illegal means of acquisition). Preferably, use seed songs with variation in sound/genre. I decided on the songs:

* **Amoeba** by Clairo
* **I Really Want to Stay at Your House** by Rosa Walton and Hallie Coggins
* **Project Dreams** by Roddy Ricch and Marshmello
* **Undercover Martyn** by Two Door Cinema Club

Make of the developer's music taste what you will. 

Upload the .mp3 files to the `/data/mp3_seeds` directory; ensure that there also exists a directory called `/data/wav_seeds`. Note that these directories do not exist in this repository and you may have to create them manually. 

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run pipeline

```bash
python run_pipeline.py
```

This will:

* Pull new albums from Deezer released in the past $k$ days
* Download and embed track previews
* Match against your stored seed embeddings
* Save track embeddings in a new directory `/data/track_embeddings`
* Return top recommendations and which seed they resemble most closely

---

## Example Output

```yaml
[{'artist': 'Allegaeon',
  'track': 'Chaos Theory',
  'similarity': 0.7331178619874356,
  'closest_seed': 'irwtsayh'},

 {'artist': 'Black Sherif',
  'track': 'One',
  'similarity': 0.7238382769840812,
  'closest_seed': 'projectdreams'},

 {'artist': 'KeBlack',
  'track': 'Avec',
  'similarity': 0.6361437303765004,
  'closest_seed': 'projectdreams'},

{'artist': 'Djo',
  'track': 'Golden Line',
  'similarity': 0.5662374132360284,
  'closest_seed': 'amoeba'},
...
```

---

## Notes

* [OpenL3](https://github.com/marl/openl3) is an open-source Python library for computing deep audio and image embeddings. As a reference, embeddings for my four seed songs, as well as embeddings for songs released on Deezer between **April 1, 2025 and April 8, 2025** can be found in the `\data\` directory.
* Deezer only allows users to access the first 30 seconds of a song as a preview. This is useful as computing embeddings for hundreds of full-length songs can be prohibitive; however, this also means that only the first 30 seconds of returned songs is guaranteed to sound "similar" to its seed.
* The pipeline was tested on a NVIDIA Tesla T4 GPU and takes about 10 minutes to execute. Exact execution time will depend on your machine specs, seed songs, and how many new albums have come out on Deezer in the past week.

---
