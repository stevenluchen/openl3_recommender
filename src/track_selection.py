import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

def find_similar_tracks(candidate_embeddings, candidate_tracks, seed_embeddings, seed_names, top_k):
  seed_matrix = np.stack(seed_embeddings)
  
  pca = PCA(n_components=50)
  reduced_embs = normalize(pca.fit_transform(candidate_embeddings))
  reduced_seeds = normalize(pca.transform(seed_matrix))
  similarities = cosine_similarity(reduced_embs, reduced_seeds)

  max_scores = np.max(similarities, axis=1)
  best_seed_idx = np.argmax(similarities, axis=1)

  all_candidates = []
  for i, track in enumerate(candidate_tracks):
    artist, title = track.split(" - ", maxsplit=1)
    all_candidates.append({
      'artist': artist,
      'track': title,
      'similarity': float(max_scores[i]),
      'closest_seed': seed_names[best_seed_idx[i]]
    })

  all_candidates.sort(key=lambda x: x['similarity'], reverse=True)

  # select best matches per seed
  max_per_artist = math.floor(0.2 * top_k)
  per_seed_quota = math.ceil(top_k / len(seed_names))
  selected_tracks = []
  artist_counts = defaultdict(int)
  used_indices = set()

  for j, seed in enumerate(seed_names):
    seed_matches = [(i, similarities[i][j]) for i in range(len(all_candidates))]
    seed_matches.sort(key=lambda x: -x[1])

    count = 0
    for idx, sim in seed_matches:
      if idx in used_indices:
        continue
      artist = all_candidates[idx]['artist']
      if artist_counts[artist] < max_per_artist:
        selected_tracks.append(all_candidates[idx])
        artist_counts[artist] += 1
        used_indices.add(idx)
        count += 1
      if count >= per_seed_quota:
        break

  # fill remaining up to top_k with remaining best matches
  if len(selected_tracks) < top_k:
    for i in range(len(all_candidates)):
      if i in used_indices:
        continue
      artist = all_candidates[i]['artist']
      if artist_counts[artist] < max_per_artist:
        selected_tracks.append(all_candidates[i])
        artist_counts[artist] += 1
        used_indices.add(i)
      if len(selected_tracks) >= top_k:
        break
  
  selected_tracks.sort(key=lambda x: -x['similarity'])
  return selected_tracks[:top_k]
