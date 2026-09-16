import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from quantum_model import VariationalQuantumCircuit, AngleEncoding
from config import N_QUBITS, VQC_LAYERS, ALL_GENRES

# 50 Multi-Lingual Benchmark Movies dataset
BENCHMARK_MOVIES = [
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
    { "id": 13, "title": "3 Idiots", "year": 2009, "genres": ["Comedy", "Drama"], "language": "Hindi", "rating": 8.4, "moods": ["Feel Good", "Funny"], "description": "Two friends search for their long lost companion while recollecting their college days and the professor who inspired them." },
    { "id": 14, "title": "Jai Bhim", "year": 2021, "genres": ["Drama", "Crime"], "language": "Tamil", "rating": 8.8, "moods": ["Emotional", "Thought-Provoking"], "description": "When a tribal man is arrested for a case of alleged theft, his wife turns to a brave lawyer for justice." },
    { "id": 15, "title": "Drishyam", "year": 2013, "genres": ["Crime", "Drama", "Thriller"], "language": "Malayalam", "rating": 8.3, "moods": ["Thought-Provoking", "Exciting"], "description": "A desperate man takes extreme measures to protect his family after they commit an accidental crime." },
    { "id": 16, "title": "K.G.F: Chapter 1", "year": 2018, "genres": ["Action", "Crime", "Drama"], "language": "Kannada", "rating": 8.2, "moods": ["Exciting"], "description": "In the 1970s, a fierce rebel rises against brutal oppressors in the gold fields of Kolar." },
    { "id": 17, "title": "Baahubali: The Beginning", "year": 2015, "genres": ["Action", "Drama", "Fantasy"], "language": "Telugu", "rating": 8.0, "moods": ["Exciting"], "description": "A adventurous young man helps his love rescue her former queen from the tyrannical ruler of Mahishmati." },
    { "id": 18, "title": "Your Name", "year": 2016, "genres": ["Animation", "Drama", "Romance"], "language": "Japanese", "rating": 8.4, "moods": ["Romantic", "Emotional"], "description": "Two high school strangers find themselves linked in a bizarre way when they begin swapping bodies." },
    { "id": 19, "title": "Amélie", "year": 2001, "genres": ["Comedy", "Romance"], "language": "French", "rating": 8.3, "moods": ["Feel Good", "Romantic"], "description": "Amélie is an innocent and naive girl in Paris with her own sense of justice who decides to help those around her." },
    { "id": 20, "title": "Pan's Labyrinth", "year": 2006, "genres": ["Fantasy", "Drama", "War"], "language": "Spanish", "rating": 8.2, "moods": ["Thought-Provoking", "Scary"], "description": "In the Falangist Spain of 1944, the bookish young stepdaughter of a sadistic army officer escapes into an eerie fantasy world." },
    { "id": 21, "title": "Titanic", "year": 1997, "genres": ["Drama", "Romance"], "language": "English", "rating": 7.9, "moods": ["Romantic", "Emotional"], "description": "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the ill-fated R.M.S. Titanic." },
    { "id": 22, "title": "Avatar", "year": 2009, "genres": ["Sci-Fi", "Action", "Adventure"], "language": "English", "rating": 7.9, "moods": ["Futuristic", "Exciting"], "description": "A paraplegic Marine dispatched to Pandora becomes torn between following orders and protecting the Na'vi." },
    { "id": 23, "title": "Toy Story", "year": 1995, "genres": ["Animation", "Comedy", "Adventure"], "language": "English", "rating": 8.3, "moods": ["Feel Good", "Funny"], "description": "A cowboy doll is profoundly threatened when a new spaceman action figure replaces him as top toy." },
    { "id": 24, "title": "Jurassic Park", "year": 1993, "genres": ["Sci-Fi", "Adventure", "Thriller"], "language": "English", "rating": 8.2, "moods": ["Exciting"], "description": "A pragmatic paleontologist touring an almost complete theme park is tasked with protecting two kids after a power failure." },
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

def encode_genre_vector(genres_list):
    vec = np.zeros(len(ALL_GENRES), dtype=float)
    for g in genres_list:
        if g in ALL_GENRES:
            idx = ALL_GENRES.index(g)
            vec[idx] = 1.0
    return vec

def run_vqc_pipeline(user_prefs):
    # Candidate filtering
    selected_genres = set(user_prefs.get("genres", []))
    lang = user_prefs.get("language", "Any Language")
    year_from = int(user_prefs.get("year_from", 1950))
    year_to = int(user_prefs.get("year_to", 2026))
    any_year = user_prefs.get("any_year", False)
    min_rating = float(user_prefs.get("min_rating", 0.0))
    top_k = int(user_prefs.get("rec_count", 5))
    moods = user_prefs.get("moods", [])
    fav_id = user_prefs.get("favorite_movie_id", None)

    candidates = []
    for m in BENCHMARK_MOVIES:
        if fav_id and m["id"] == int(fav_id):
            continue
        if lang != "Any Language" and m["language"] != lang:
            continue
        if m["rating"] < min_rating:
            continue
        if not any_year and (m["year"] < year_from or m["year"] > year_to):
            continue
        # Filter candidates matching genres if specified
        if selected_genres:
            if not any(g in selected_genres for g in m["genres"]):
                continue
        candidates.append(m)

    if not candidates:
        return {
            "status": "empty",
            "message": "No exact matches found. Please adjust language, year, or rating filters."
        }

    # Build User Vector (19D)
    user_vec = np.zeros(len(ALL_GENRES), dtype=float)
    if selected_genres:
        for g in selected_genres:
            if g in ALL_GENRES:
                user_vec[ALL_GENRES.index(g)] = 1.0

    # Latent bias adjustment: Moods
    mood_genre_map = {
        "Feel Good": ["Comedy", "Animation", "Romance", "Drama"],
        "Funny": ["Comedy", "Animation"],
        "Exciting": ["Action", "Adventure", "Thriller"],
        "Thought-Provoking": ["Sci-Fi", "Drama", "Mystery", "Documentary"],
        "Romantic": ["Romance", "Drama"],
        "Scary": ["Horror", "Thriller"],
        "Futuristic": ["Sci-Fi"],
        "Emotional": ["Drama", "Romance"]
    }

    if isinstance(moods, str):
        moods = [moods]

    for m_str in moods:
        # Strip emoji if present
        clean_m = m_str.replace("😊 ", "").replace("😂 ", "").replace("🔥 ", "").replace("🧠 ", "").replace("❤️ ", "").replace("👻 ", "").replace("🚀 ", "").replace("🎭 ", "")
        for g in mood_genre_map.get(clean_m, []):
            if g in ALL_GENRES:
                user_vec[ALL_GENRES.index(g)] += 0.25

    # Latent bias adjustment: Favorite Movie
    if fav_id:
        fav_movie = next((m for m in BENCHMARK_MOVIES if m["id"] == int(fav_id)), None)
        if fav_movie:
            fav_vec = encode_genre_vector(fav_movie["genres"])
            user_vec += 0.3 * fav_vec

    # Normalize user vector
    if np.max(user_vec) > 0:
        user_vec = user_vec / np.max(user_vec)

    # 1. Map combined User Vector (19D) and Movie Vector (19D) -> 38D feature space
    X_38D = []
    for m in candidates:
        m_vec = encode_genre_vector(m["genres"])
        combined_38D = np.concatenate([user_vec, m_vec])
        X_38D.append(combined_38D)

    X_38D = np.array(X_38D)

    # 2. Apply PCA to reduce 38D -> 4D
    n_components = min(4, X_38D.shape[0])
    if X_38D.shape[0] >= 4:
        pca = PCA(n_components=4, random_state=42)
        X_4D = pca.fit_transform(X_38D)
    else:
        # Fallback padding for small candidate set
        pca = PCA(n_components=n_components, random_state=42)
        X_reduced = pca.fit_transform(X_38D)
        X_4D = np.zeros((X_38D.shape[0], 4))
        X_4D[:, :n_components] = X_reduced

    # 3. Encode features onto 4 Qubits using Ry Angle Encoding: [0, pi]
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X_angles = scaler.fit_transform(X_4D)

    # 4 & 5. Execute 2-Layer VQC and Extract measurement probability P(|1>)
    vqc = VariationalQuantumCircuit(n_qubits=N_QUBITS, n_layers=VQC_LAYERS)
    np.random.seed(42)
    sample_theta = np.random.uniform(0, 2 * np.pi, vqc.num_params)

    scored_movies = []
    for idx, m in enumerate(candidates):
        angles = X_angles[idx]
        prob_one = vqc.evaluate_preference(angles, sample_theta)
        # Ensure score reflects preference alignment with rating boost
        raw_score = float(prob_one)
        # Smooth scaling for realistic scores
        genre_overlap = sum(1 for g in m["genres"] if g in selected_genres) / max(1, len(selected_genres)) if selected_genres else 0.5
        rating_factor = m["rating"] / 10.0
        final_quantum_score = round(0.4 * raw_score + 0.4 * genre_overlap + 0.2 * rating_factor, 3)
        final_quantum_score = min(max(final_quantum_score, 0.500), 0.999)
        
        scored_movies.append({
            "movie": m,
            "quantum_score": final_quantum_score
        })

    # Sort by quantum score DESC, rating DESC
    scored_movies.sort(key=lambda x: (x["quantum_score"], x["movie"]["rating"]), reverse=True)

    top_recs = scored_movies[:top_k]

    top_match = top_recs[0]["movie"]
    top_score = top_recs[0]["quantum_score"]

    recommendations_list = []
    for rank_idx, item in enumerate(top_recs[1:], start=1):
        m = item["movie"]
        score = item["quantum_score"]
        recommendations_list.append({
            "rank": rank_idx,
            "title": m["title"],
            "year": m["year"],
            "rating": m["rating"],
            "language": m["language"],
            "genres": m["genres"],
            "quantum_score": score,
            "match_percentage": f"{round(score * 100, 1)}%"
        })

    genres_display = list(selected_genres) if selected_genres else ["All"]
    mood_display = moods[0] if len(moods) == 1 else (moods if moods else "None")

    response = {
        "status": "success",
        "preferences_summary": {
            "genres": genres_display,
            "language": lang,
            "year_range": f"{year_from}-{year_to}" if not any_year else "Any Year",
            "min_rating": min_rating,
            "mood": mood_display
        },
        "top_match": {
            "title": top_match["title"],
            "year": top_match["year"],
            "rating": top_match["rating"],
            "language": top_match["language"],
            "genres": top_match["genres"],
            "quantum_score": top_score,
            "match_percentage": f"{round(top_score * 100, 1)}%",
            "description": top_match["description"]
        },
        "recommendations": recommendations_list
    }
    return response

if __name__ == "__main__":
    sample_prefs = {
        "genres": ["Action", "Sci-Fi"],
        "language": "English",
        "year_from": 2010,
        "year_to": 2025,
        "min_rating": 7.0,
        "rec_count": 5,
        "moods": ["Thought-Provoking"]
    }
    import json
    res = run_vqc_pipeline(sample_prefs)
    print(json.dumps(res, indent=2))
