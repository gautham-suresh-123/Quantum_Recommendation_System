"""
Recommendation Pipeline Module (Research Grade).

Ingests user preference parameters, applies pre-fitted Scaler and PCA models from artifact,
evaluates 4-Qubit VQC measurement probability S_Q, computes Classical Content Similarity S_C,
and returns un-clamped Hybrid Recommendations S_H = alpha * S_Q + (1 - alpha) * S_C.

Supports explicit Research Mode (MovieLens dataset catalog) and Demo Mode.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    MODEL_FILE,
    N_QUBITS,
    VQC_LAYERS,
    ALL_GENRES,
    HYBRID_ALPHA,
    MOVIES_FILE,
    RATINGS_FILE
)
from src.preprocessing import (
    encode_genres,
    build_interaction_features,
    transform_preprocessing_pipeline,
    download_and_load_data
)
from src.quantum_model import VariationalQuantumCircuit

# 50 Multi-Lingual Benchmark Movies Catalog
DEMO_BENCHMARK_MOVIES = [
    { "id": 1, "title": "Interstellar", "year": 2014, "genres": ["Sci-Fi", "Drama"], "language": "English", "rating": 8.7, "moods": ["Thought-Provoking", "Futuristic"], "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival." },
    { "id": 2, "title": "Inception", "year": 2010, "genres": ["Sci-Fi", "Action", "Thriller"], "language": "English", "rating": 8.8, "moods": ["Thought-Provoking", "Exciting"], "description": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea." },
    { "id": 3, "title": "Dangal", "year": 2016, "genres": ["Drama", "Action"], "language": "Hindi", "rating": 8.3, "moods": ["Emotional", "Feel Good"], "description": "Former wrestler Mahavir Singh Phogat and his two wrestler daughters struggle towards glory at the Commonwealth Games." },
    { "id": 4, "title": "Vikram", "year": 2022, "genres": ["Action", "Thriller", "Crime"], "language": "Tamil", "rating": 8.3, "moods": ["Exciting"], "description": "A high-octane action thriller following a special agent investigating a masked vigilante group." },
    { "id": 5, "title": "Manjummel Boys", "year": 2024, "genres": ["Adventure", "Thriller", "Drama"], "language": "Malayalam", "rating": 8.4, "moods": ["Exciting", "Emotional"], "description": "A group of friends from a small town embark on a vacation to Kodaikanal, where a life-or-death rescue operation unfolds in Guna Caves." },
    { "id": 6, "title": "RRR", "year": 2022, "genres": ["Action", "Drama"], "language": "Telugu", "rating": 7.8, "moods": ["Exciting", "Feel Good"], "description": "A fearless revolutionary and an officer in the British army forge an unlikely friendship before revealing their true missions." },
    { "id": 7, "title": "Parasite", "year": 2019, "genres": ["Thriller", "Drama", "Comedy"], "language": "Korean", "rating": 8.5, "moods": ["Thought-Provoking"], "description": "Greed and class discrimination threaten the newly formed symbiotic relationship between two families." },
    { "id": 8, "title": "Spirited Away", "year": 2001, "genres": ["Animation", "Adventure", "Fantasy"], "language": "Japanese", "rating": 8.6, "moods": ["Feel Good", "Thought-Provoking"], "description": "A young girl wanders into a world ruled by gods, witches and spirits." },
    { "id": 9, "title": "The Intouchables", "year": 2011, "genres": ["Comedy", "Drama"], "language": "French", "rating": 8.5, "moods": ["Feel Good", "Funny"], "description": "After he becomes a quadriplegic from a paragliding accident, an aristocrat hires a young man from the projects to be his caregiver." },
    { "id": 10, "title": "Kantara", "year": 2022, "genres": ["Action", "Adventure", "Fantasy"], "language": "Kannada", "rating": 8.2, "moods": ["Exciting", "Thought-Provoking"], "description": "When greed paves the way for betrayal and murder, a young tribal man reluctantly takes up the torch of his ancestors." },
    
    { "id": 11, "title": "The Matrix", "year": 1999, "genres": ["Sci-Fi", "Action"], "language": "English", "rating": 8.7, "moods": ["Futuristic", "Exciting"], "description": "A computer hacker learns from mysterious rebels about the true nature of his reality." },
    { "id": 12, "title": "The Dark Knight", "year": 2008, "genres": ["Action", "Crime", "Drama"], "language": "English", "rating": 9.0, "moods": ["Exciting", "Thought-Provoking"], "description": "When the Joker wreaks havoc on Gotham, Batman must accept one of the greatest tests of his ability to fight injustice." },
    { "id": 13, "title": "3 Idiots", "year": 2009, "genres": ["Comedy", "Drama"], "language": "Hindi", "rating": 8.4, "moods": ["Feel Good", "Funny"], "description": "Two friends search for their long lost companion while recollecting their college days." },
    { "id": 14, "title": "Jai Bhim", "year": 2021, "genres": ["Drama", "Crime"], "language": "Tamil", "rating": 8.8, "moods": ["Emotional", "Thought-Provoking"], "description": "When a tribal man is arrested for a case of alleged theft, his wife turns to a brave lawyer for justice." },
    { "id": 15, "title": "Drishyam", "year": 2013, "genres": ["Crime", "Drama", "Thriller"], "language": "Malayalam", "rating": 8.3, "moods": ["Thought-Provoking", "Exciting"], "description": "A desperate man takes extreme measures to protect his family after they commit an accidental crime." },
    { "id": 16, "title": "K.G.F: Chapter 1", "year": 2018, "genres": ["Action", "Crime", "Drama"], "language": "Kannada", "rating": 8.2, "moods": ["Exciting"], "description": "In the 1970s, a fierce rebel rises against brutal oppressors in the gold fields of Kolar." },
    { "id": 17, "title": "Baahubali: The Beginning", "year": 2015, "genres": ["Action", "Drama", "Fantasy"], "language": "Telugu", "rating": 8.0, "moods": ["Exciting"], "description": "An adventurous young man helps his love rescue her former queen from the tyrannical ruler." },
    { "id": 18, "title": "Your Name", "year": 2016, "genres": ["Animation", "Drama", "Romance"], "language": "Japanese", "rating": 8.4, "moods": ["Romantic", "Emotional"], "description": "Two high school strangers find themselves linked in a bizarre way when they begin swapping bodies." },
    { "id": 19, "title": "Amélie", "year": 2001, "genres": ["Comedy", "Romance"], "language": "French", "rating": 8.3, "moods": ["Feel Good", "Romantic"], "description": "Amélie is an innocent girl in Paris with her own sense of justice who decides to help those around her." },
    { "id": 20, "title": "Pan's Labyrinth", "year": 2006, "genres": ["Fantasy", "Drama", "War"], "language": "Spanish", "rating": 8.2, "moods": ["Thought-Provoking", "Scary"], "description": "In the Falangist Spain of 1944, a bookish young stepdaughter escapes into an eerie fantasy world." },

    { "id": 21, "title": "Titanic", "year": 1997, "genres": ["Drama", "Romance"], "language": "English", "rating": 7.9, "moods": ["Romantic", "Emotional"], "description": "A seventeen-year-old aristocrat falls in love with a kind artist aboard the ill-fated R.M.S. Titanic." },
    { "id": 22, "title": "Avatar", "year": 2009, "genres": ["Sci-Fi", "Action", "Adventure"], "language": "English", "rating": 7.9, "moods": ["Futuristic", "Exciting"], "description": "A paraplegic Marine dispatched to Pandora becomes torn between following orders and protecting the Na'vi." },
    { "id": 23, "title": "Toy Story", "year": 1995, "genres": ["Animation", "Comedy", "Adventure"], "language": "English", "rating": 8.3, "moods": ["Feel Good", "Funny"], "description": "A cowboy doll is profoundly threatened when a new spaceman action figure replaces him as top toy." },
    { "id": 24, "title": "Jurassic Park", "year": 1993, "genres": ["Sci-Fi", "Adventure", "Thriller"], "language": "English", "rating": 8.2, "moods": ["Exciting"], "description": "A pragmatic paleontologist touring an almost complete theme park is tasked with protecting two kids." },
    { "id": 25, "title": "The Shawshank Redemption", "year": 1994, "genres": ["Drama"], "language": "English", "rating": 9.3, "moods": ["Emotional", "Thought-Provoking"], "description": "Over the course of several years, two convicts form a friendship, seeking solace and ultimate redemption." },
    { "id": 26, "title": "Forrest Gump", "year": 1994, "genres": ["Drama", "Romance"], "language": "English", "rating": 8.8, "moods": ["Feel Good", "Emotional"], "description": "The history of the United States unfolds through the perspective of an Alabama man with an IQ of 75." },
    { "id": 27, "title": "Gladiator", "year": 2000, "genres": ["Action", "Adventure", "Drama"], "language": "English", "rating": 8.5, "moods": ["Exciting"], "description": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family." },
    { "id": 28, "title": "The Godfather", "year": 1972, "genres": ["Crime", "Drama"], "language": "English", "rating": 9.2, "moods": ["Thought-Provoking"], "description": "The aging patriarch of an organized crime dynasty transfers control of his empire to his reluctant son." },
    { "id": 29, "title": "Avengers: Endgame", "year": 2019, "genres": ["Action", "Sci-Fi", "Adventure"], "language": "English", "rating": 8.4, "moods": ["Exciting", "Futuristic"], "description": "The Avengers assemble once more to reverse Thanos' actions and restore balance to the universe." },
    { "id": 30, "title": "Spider-Man: Into the Spider-Verse", "year": 2018, "genres": ["Animation", "Action", "Adventure"], "language": "English", "rating": 8.4, "moods": ["Exciting", "Feel Good"], "description": "Teen Miles Morales becomes the Spider-Man of his universe and joins spider-powered allies." },

    { "id": 31, "title": "The Lord of the Rings: The Fellowship of the Ring", "year": 2001, "genres": ["Fantasy", "Adventure", "Action"], "language": "English", "rating": 8.8, "moods": ["Exciting"], "description": "A meek Hobbit from the Shire sets out on a journey to destroy the One Ring." },
    { "id": 32, "title": "Alien", "year": 1979, "genres": ["Horror", "Sci-Fi"], "language": "English", "rating": 8.5, "moods": ["Scary", "Futuristic"], "description": "The crew of a commercial spacecraft encounters a deadly alien lifeform." },
    { "id": 33, "title": "The Lion King", "year": 1994, "genres": ["Animation", "Drama", "Musical"], "language": "English", "rating": 8.5, "moods": ["Feel Good", "Emotional"], "description": "Lion prince Simba and his father are targeted by his bitter uncle Scar." },
    { "id": 34, "title": "Coco", "year": 2017, "genres": ["Animation", "Fantasy", "Musical"], "language": "English", "rating": 8.4, "moods": ["Feel Good", "Emotional"], "description": "Aspiring musician Miguel enters the Land of the Dead to find his great-great-grandfather." },
    { "id": 35, "title": "Joker", "year": 2019, "genres": ["Crime", "Drama", "Thriller"], "language": "English", "rating": 8.4, "moods": ["Thought-Provoking", "Emotional"], "description": "A mentally troubled stand-up comedian embarks on a downward spiral that leads to an iconic villain." },
    { "id": 36, "title": "Whiplash", "year": 2014, "genres": ["Drama", "Musical"], "language": "English", "rating": 8.5, "moods": ["Thought-Provoking", "Exciting"], "description": "A promising young drummer enrolls at a cut-throat music conservatory mentored by a ruthless instructor." },
    { "id": 37, "title": "La La Land", "year": 2016, "genres": ["Drama", "Romance", "Musical"], "language": "English", "rating": 8.0, "moods": ["Romantic", "Feel Good"], "description": "While navigating their careers in Los Angeles, a pianist and an actress fall in love." },
    { "id": 38, "title": "The Prestige", "year": 2006, "genres": ["Mystery", "Drama", "Sci-Fi"], "language": "English", "rating": 8.5, "moods": ["Thought-Provoking"], "description": "Two stage magicians in 1890s London engage in a battle to create the ultimate illusion." },
    { "id": 39, "title": "Get Out", "year": 2017, "genres": ["Horror", "Mystery", "Thriller"], "language": "English", "rating": 7.7, "moods": ["Scary", "Thought-Provoking"], "description": "A young African-American visits his white girlfriend's parents, uncovering a disturbing secret." },
    { "id": 40, "title": "Dune", "year": 2021, "genres": ["Sci-Fi", "Adventure", "Drama"], "language": "English", "rating": 8.0, "moods": ["Futuristic", "Thought-Provoking"], "description": "A noble family becomes embroiled in a war for control over the galaxy's most valuable asset." },

    { "id": 41, "title": "Oppenheimer", "year": 2023, "genres": ["Drama", "Documentary"], "language": "English", "rating": 8.9, "moods": ["Thought-Provoking"], "description": "The story of J. Robert Oppenheimer and his role in the development of the atomic bomb." },
    { "id": 42, "title": "Top Gun: Maverick", "year": 2022, "genres": ["Action", "Drama"], "language": "English", "rating": 8.3, "moods": ["Exciting"], "description": "After thirty years, Maverick leads TOPGUN's elite graduates on a dangerous specialized mission." },
    { "id": 43, "title": "Inside Out", "year": 2015, "genres": ["Animation", "Comedy", "Drama"], "language": "English", "rating": 8.1, "moods": ["Feel Good", "Funny"], "description": "After young Riley moves to a new city, her internal emotions conflict on how best to navigate her life." },
    { "id": 44, "title": "WALL-E", "year": 2008, "genres": ["Animation", "Sci-Fi", "Adventure"], "language": "English", "rating": 8.4, "moods": ["Futuristic", "Feel Good"], "description": "In a distant future, a waste-collecting robot embarks on a space journey that decides humanity's fate." },
    { "id": 45, "title": "Super Deluxe", "year": 2019, "genres": ["Crime", "Drama", "Thriller"], "language": "Tamil", "rating": 8.3, "moods": ["Thought-Provoking"], "description": "An unfaithful wife, an estranged father, a transgender woman, and teenagers find their destinies linked." },
    { "id": 46, "title": "Premam", "year": 2015, "genres": ["Romance", "Comedy", "Drama"], "language": "Malayalam", "rating": 8.3, "moods": ["Feel Good", "Romantic"], "description": "A young man's journey through love at three different stages of his life." },
    { "id": 47, "title": "Lagaan", "year": 2001, "genres": ["Drama", "Musical", "War"], "language": "Hindi", "rating": 8.1, "moods": ["Feel Good", "Emotional"], "description": "Villagers in Victorian India stake their future on a game of cricket against ruthless British officers." },
    { "id": 48, "title": "Minari", "year": 2020, "genres": ["Drama"], "language": "Korean", "rating": 7.4, "moods": ["Emotional", "Thought-Provoking"], "description": "A Korean-American family moves to an Arkansas farm in search of their own American dream." },
    { "id": 49, "title": "Free Solo", "year": 2018, "genres": ["Documentary"], "language": "English", "rating": 8.1, "moods": ["Exciting", "Thought-Provoking"], "description": "Follow Alex Honnold as he attempts to perform a free solo climb of El Capitan in Yosemite National Park." },
    { "id": 50, "title": "Schindler's List", "year": 1993, "genres": ["Drama", "War"], "language": "English", "rating": 9.0, "moods": ["Emotional", "Thought-Provoking"], "description": "In German-occupied Poland during WWII, industrialist Oskar Schindler saves his Jewish workforce." }
]


def load_model_artifact(model_path: str = str(MODEL_FILE)) -> Optional[Dict[str, Any]]:
    """Loads saved model artifact containing theta weights and pre-fitted transformers."""
    if not os.path.exists(model_path):
        return None
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def encode_genre_vector(genres_list: List[str]) -> np.ndarray:
    """Multi-hot encodes genres list into 19D vector."""
    vec = np.zeros(len(ALL_GENRES), dtype=float)
    for g in genres_list:
        if g in ALL_GENRES:
            idx = ALL_GENRES.index(g)
            vec[idx] = 1.0
    return vec


def _extract_year_from_movie(row) -> int:
    """Extracts integer release year from release_date or title string."""
    rel_date = str(row.get("release_date", ""))
    if rel_date and len(rel_date) >= 4 and rel_date[-4:].isdigit():
        return int(rel_date[-4:])
    
    title = str(row.get("title", ""))
    import re
    match = re.search(r'\((\d{4})\)', title)
    if match:
        return int(match.group(1))
    return 1995


def process_quantum_recommendations(
    user_prefs: Dict[str, Any],
    mode: str = "RESEARCH_MODE"
) -> Dict[str, Any]:
    """
    Executes the 5-Step Quantum Processing Protocol with full preference parameter integration:
    1. Construct 19D User Preference Vector and candidate Movie Profile.
    2. Build 76D Canonical Interaction Feature Vector.
    3. Apply PRE-FITTED Scaler & PCA to reduce 76D -> 4D mapped to [0, pi].
    4. Execute 4-Qubit VQC to obtain quantum probability S_Q = P(|1>).
    5. Compute Classical Cosine Similarity S_C and output Hybrid Score S_H = alpha * S_Q + (1 - alpha) * S_C.
    6. Return fully-hydrated recommendations with complete metadata and scores.
    """
    # 1. Parse Input Preferences
    selected_genres = user_prefs.get("genres", [])
    if isinstance(selected_genres, str):
        selected_genres = [selected_genres]
    selected_genres_set = set(selected_genres)

    selected_language = user_prefs.get("language") or user_prefs.get("selected_language") or "Any Language"
    any_year = bool(user_prefs.get("any_year") or user_prefs.get("anyYear") or False)
    year_from = int(user_prefs.get("year_from") or user_prefs.get("yearFrom") or 1950)
    year_to = int(user_prefs.get("year_to") or user_prefs.get("yearTo") or 2026)
    min_rating = float(user_prefs.get("min_rating") or user_prefs.get("minRating") or 0)
    fav_movie_id = user_prefs.get("favorite_movie_id") or user_prefs.get("favorite_id") or user_prefs.get("favoriteMovieId")
    if fav_movie_id:
        try:
            fav_movie_id = int(fav_movie_id)
        except Exception:
            fav_movie_id = None

    top_k = int(user_prefs.get("rec_count") or user_prefs.get("recCount") or 5)
    alpha = float(user_prefs.get("alpha", HYBRID_ALPHA))

    # 2. Build Candidate Pool
    candidates = []
    movie_ratings_map = {}

    if mode == "RESEARCH_MODE" and MOVIES_FILE.exists():
        movies_df, ratings_df = download_and_load_data()
        genre_features_df = encode_genres(movies_df)

        # Precompute mean rating per movie scaled to 10-point scale
        if len(ratings_df) > 0:
            mean_ratings = ratings_df.groupby("movieId")["rating"].mean()
            movie_ratings_map = (mean_ratings * 2.0).round(1).to_dict()

        for idx, row in movies_df.iterrows():
            m_genres = [g.strip() for g in str(row["genres"]).split("|") if g.strip()]
            mid = int(row["movieId"])
            m_year = _extract_year_from_movie(row)
            m_rating = movie_ratings_map.get(mid, 7.5)
            candidates.append({
                "id": mid,
                "title": str(row["title"]),
                "year": m_year,
                "genres": m_genres,
                "language": "English",
                "rating": m_rating,
                "description": f"Top-rated {', '.join(m_genres[:2])} classic released in {m_year}."
            })
    else:
        candidates = list(DEMO_BENCHMARK_MOVIES)
        genre_features_df = pd.DataFrame(
            [encode_genre_vector(m["genres"]) for m in candidates],
            index=[m["id"] for m in candidates],
            columns=ALL_GENRES
        )

    # 3. Apply Filtering Rules
    filtered_candidates = []
    for m in candidates:
        if fav_movie_id and m["id"] == fav_movie_id:
            continue
        if selected_genres_set and not any(g in selected_genres_set for g in m["genres"]):
            continue
        if selected_language != "Any Language" and m.get("language") != selected_language:
            continue
        if not any_year:
            if m["year"] < year_from or m["year"] > year_to:
                continue
        if m["rating"] < min_rating:
            continue
        filtered_candidates.append(m)

    # Dynamic Fallback: If strict filters yield zero candidates on MovieLens dataset, check benchmark catalog
    if not filtered_candidates and mode == "RESEARCH_MODE":
        for m in DEMO_BENCHMARK_MOVIES:
            if fav_movie_id and m["id"] == fav_movie_id:
                continue
            if selected_genres_set and not any(g in selected_genres_set for g in m["genres"]):
                continue
            if selected_language != "Any Language" and m.get("language") != selected_language:
                continue
            if not any_year:
                if m["year"] < year_from or m["year"] > year_to:
                    continue
            if m["rating"] < min_rating:
                continue
            filtered_candidates.append(m)

    # Secondary Fallback: If still zero candidates, relax year/rating filters to ensure user receives results
    if not filtered_candidates:
        filtered_candidates = [
            m for m in (candidates + DEMO_BENCHMARK_MOVIES)
            if not selected_genres_set or any(g in selected_genres_set for g in m["genres"])
        ]

    if not filtered_candidates:
        return {
            "status": "empty",
            "message": "No candidate movies match the selected preference criteria."
        }

    # 4. Quantum & Hybrid Recommendation Scoring
    user_vec = encode_genre_vector(selected_genres)

    artifact = load_model_artifact()
    if not artifact:
        return {
            "status": "error",
            "message": "Required trained model artifact (models/vqc_recommender.pkl) not found. Please run 'python train.py' first."
        }

    theta_params = artifact["optimal_theta"]
    scaler = artifact["scaler"]
    pca_model = artifact["pca_model"]
    angle_scaler = artifact["angle_scaler"]
    vqc = VariationalQuantumCircuit(n_qubits=artifact["n_qubits"], n_layers=artifact["n_layers"])

    scored_movies = []
    seen_ids = set()

    for m in filtered_candidates:
        mid = m["id"]
        if mid in seen_ids:
            continue
        seen_ids.add(mid)

        if mid in genre_features_df.index:
            m_prof = genre_features_df.loc[mid].values.astype(float)
        else:
            m_prof = encode_genre_vector(m["genres"])

        raw_feat = build_interaction_features(user_vec, m_prof).reshape(1, -1)
        q_feat = transform_preprocessing_pipeline(raw_feat, scaler, pca_model, angle_scaler)[0]
        s_q = float(vqc.evaluate_preference(q_feat, theta_params))

        u_norm = np.linalg.norm(user_vec)
        m_norm = np.linalg.norm(m_prof)
        if u_norm > 0 and m_norm > 0:
            s_c = float(np.dot(user_vec, m_prof) / (u_norm * m_norm))
        else:
            s_c = 0.5

        s_h = float(alpha * s_q + (1.0 - alpha) * s_c)
        match_pct_int = int(round(s_h * 100))

        # Dynamic Explanation String
        matched_g = [g for g in m["genres"] if g in selected_genres_set]
        exp_parts = []
        if matched_g:
            exp_parts.append(f"Matches your {' and '.join(matched_g)} preference")
        if selected_language != "Any Language" and m.get("language") == selected_language:
            exp_parts.append(f"selected {selected_language} language")
        if not any_year and (year_from <= m["year"] <= year_to):
            exp_parts.append(f"released in target era {m['year']}")

        if exp_parts:
            explanation = exp_parts[0].capitalize() + (", " + ", ".join(exp_parts[1:]) + "." if len(exp_parts) > 1 else ".")
        else:
            explanation = f"Top VQC quantum candidate with rating {m.get('rating', 8.0)}/10."

        scored_movies.append({
            "id": mid,
            "title": m["title"],
            "year": m.get("year", 2015),
            "genres": m.get("genres", []),
            "language": m.get("language", "English"),
            "rating": m.get("rating", 8.0),
            "description": m.get("description", ""),
            "quantum_score": round(s_q, 4),
            "quantum_score_sq": round(s_q, 4),
            "classical_score": round(s_c, 4),
            "classical_score_sc": round(s_c, 4),
            "hybrid_score": round(s_h, 4),
            "hybrid_score_sh": round(s_h, 4),
            "match_percentage": f"{match_pct_int}%",
            "explanation": explanation
        })

    scored_movies.sort(key=lambda x: x["hybrid_score"], reverse=True)
    top_recs = scored_movies[:top_k]

    # Assign explicit ranks
    for idx, rec in enumerate(top_recs):
        rec["rank"] = idx + 1

    return {
        "status": "success",
        "mode": mode,
        "model_artifact_loaded": artifact is not None,
        "hybrid_alpha": alpha,
        "top_match": top_recs[0] if top_recs else None,
        "recommendations": top_recs
    }

