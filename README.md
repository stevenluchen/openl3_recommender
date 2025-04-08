# Audio Embedding-Based Music Recommender

This project is a deep audio embedding pipeline that helps you discover new music based on your personal taste — no genre tags, no popularity bias, just the sound.

Using OpenL3 embeddings and the Deezer API, this tool pulls recent releases, computes 30-second preview embeddings, and recommends songs that are *sonically similar* to a set of user-provided seed tracks. 

---

## Features

- Extracts 30s previews from newly released tracks using Deezer API
- Computes OpenL3 audio embeddings from raw waveform
- Uses PCA + cosine similarity to compare songs in reduced embedding space
- Enforces diversity constraints (per-artist caps, multi-seed coverage)
- Outputs a curated list of tracks and similarity score to seed tracks

---

## Setup Instructions

### 1. Acquire .mp3 files of 2-5 seed songs

Do this by whatever means are convenient to the user. Preferably, use seed songs with variation in sound/genre. I decided on the songs:

* **Amoeba** by Clairo
* **I Really Want to Stay at Your House** by Rosa Walton and Haillie Coggins
* **Project Dreams** by Roddy Ricch and Marshmello
* **Undercover Martyn** by Two Door Cinema Club
