/**
 * QUANTUMFLIX - MOVIE RECOMMENDATION SYSTEM
 * Standalone Vanilla JavaScript Application (ES6+)
 * Zero External Dependencies / Zero APIs
 */

/* ==========================================================================
   1. MOVIE DATASET (50 Multi-Lingual Benchmark Movies)
   ========================================================================== */
const MOVIE_DATABASE = [
    { id: 1, title: "Interstellar", year: 2014, genres: ["Sci-Fi", "Drama"], language: "English", rating: 8.7, moods: ["Thought-Provoking", "Futuristic"], description: "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival." },
    { id: 2, title: "Inception", year: 2010, genres: ["Sci-Fi", "Action", "Thriller"], language: "English", rating: 8.8, moods: ["Thought-Provoking", "Exciting"], description: "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea." },
    { id: 3, title: "Dangal", year: 2016, genres: ["Drama", "Action"], language: "Hindi", rating: 8.3, moods: ["Emotional", "Feel Good"], description: "Former wrestler Mahavir Singh Phogat and his two wrestler daughters struggle towards glory at the Commonwealth Games." },
    { id: 4, title: "Vikram", year: 2022, genres: ["Action", "Thriller", "Crime"], language: "Tamil", rating: 8.3, moods: ["Exciting"], description: "A high-octane action thriller following a special agent investigating a masked vigilante group." },
    { id: 5, title: "Manjummel Boys", year: 2024, genres: ["Adventure", "Thriller", "Drama"], language: "Malayalam", rating: 8.4, moods: ["Exciting", "Emotional"], description: "A group of friends from a small town embark on a vacation to Kodaikanal, where a life-or-death rescue operation unfolds in Guna Caves." },
    { id: 6, title: "RRR", year: 2022, genres: ["Action", "Drama"], language: "Telugu", rating: 7.8, moods: ["Exciting", "Feel Good"], description: "A fearless revolutionary and an officer in the British army forge an unlikely friendship before revealing their true missions." },
    { id: 7, title: "Parasite", year: 2019, genres: ["Thriller", "Drama", "Comedy"], language: "Korean", rating: 8.5, moods: ["Thought-Provoking"], description: "Greed and class discrimination threaten the newly formed symbiotic relationship between two families." },
    { id: 8, title: "Spirited Away", year: 2001, genres: ["Animation", "Adventure", "Fantasy"], language: "Japanese", rating: 8.6, moods: ["Feel Good", "Thought-Provoking"], description: "A young girl wanders into a world ruled by gods, witches and spirits." },
    { id: 9, title: "The Intouchables", year: 2011, genres: ["Comedy", "Drama"], language: "French", rating: 8.5, moods: ["Feel Good", "Funny"], description: "After he becomes a quadriplegic from a paragliding accident, an aristocrat hires a young man from the projects to be his caregiver." },
    { id: 10, title: "Kantara", year: 2022, genres: ["Action", "Adventure", "Fantasy"], language: "Kannada", rating: 8.2, moods: ["Exciting", "Thought-Provoking"], description: "When greed paves the way for betrayal and murder, a young tribal man reluctantly takes up the torch of his ancestors." },
    
    { id: 11, title: "The Matrix", year: 1999, genres: ["Sci-Fi", "Action"], language: "English", rating: 8.7, moods: ["Futuristic", "Exciting"], description: "A computer hacker learns from mysterious rebels about the true nature of his reality." },
    { id: 12, title: "The Dark Knight", year: 2008, genres: ["Action", "Crime", "Drama"], language: "English", rating: 9.0, moods: ["Exciting", "Thought-Provoking"], description: "When the Joker wreaks havoc on Gotham, Batman must accept one of the greatest tests of his ability to fight injustice." },
    { id: 13, title: "3 Idiots", year: 2009, genres: ["Comedy", "Drama"], language: "Hindi", rating: 8.4, moods: ["Feel Good", "Funny"], description: "Two friends search for their long lost companion while recollecting their college days and the professor who inspired them." },
    { id: 14, title: "Jai Bhim", year: 2021, genres: ["Drama", "Crime"], language: "Tamil", rating: 8.8, moods: ["Emotional", "Thought-Provoking"], description: "When a tribal man is arrested for a case of alleged theft, his wife turns to a brave lawyer for justice." },
    { id: 15, title: "Drishyam", year: 2013, genres: ["Crime", "Drama", "Thriller"], language: "Malayalam", rating: 8.3, moods: ["Thought-Provoking", "Exciting"], description: "A desperate man takes extreme measures to protect his family after they commit an accidental crime." },
    { id: 16, title: "K.G.F: Chapter 1", year: 2018, genres: ["Action", "Crime", "Drama"], language: "Kannada", rating: 8.2, moods: ["Exciting"], description: "In the 1970s, a fierce rebel rises against brutal oppressors in the gold fields of Kolar." },
    { id: 17, title: "Baahubali: The Beginning", year: 2015, genres: ["Action", "Drama", "Fantasy"], language: "Telugu", rating: 8.0, moods: ["Exciting"], description: "A adventurous young man helps his love rescue her former queen from the tyrannical ruler of Mahishmati." },
    { id: 18, title: "Your Name", year: 2016, genres: ["Animation", "Drama", "Romance"], language: "Japanese", rating: 8.4, moods: ["Romantic", "Emotional"], description: "Two high school strangers find themselves linked in a bizarre way when they begin swapping bodies." },
    { id: 19, title: "Amélie", year: 2001, genres: ["Comedy", "Romance"], language: "French", rating: 8.3, moods: ["Feel Good", "Romantic"], description: "Amélie is an innocent and naive girl in Paris with her own sense of justice who decides to help those around her." },
    { id: 20, title: "Pan's Labyrinth", year: 2006, genres: ["Fantasy", "Drama", "War"], language: "Spanish", rating: 8.2, moods: ["Thought-Provoking", "Scary"], description: "In the Falangist Spain of 1944, the bookish young stepdaughter of a sadistic army officer escapes into an eerie fantasy world." },

    { id: 21, title: "Titanic", year: 1997, genres: ["Drama", "Romance"], language: "English", rating: 7.9, moods: ["Romantic", "Emotional"], description: "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the ill-fated R.M.S. Titanic." },
    { id: 22, title: "Avatar", year: 2009, genres: ["Sci-Fi", "Action", "Adventure"], language: "English", rating: 7.9, moods: ["Futuristic", "Exciting"], description: "A paraplegic Marine dispatched to Pandora becomes torn between following orders and protecting the Na'vi." },
    { id: 23, title: "Toy Story", year: 1995, genres: ["Animation", "Comedy", "Adventure"], language: "English", rating: 8.3, moods: ["Feel Good", "Funny"], description: "A cowboy doll is profoundly threatened when a new spaceman action figure replaces him as top toy." },
    { id: 24, title: "Jurassic Park", year: 1993, genres: ["Sci-Fi", "Adventure", "Thriller"], language: "English", rating: 8.2, moods: ["Exciting"], description: "A pragmatic paleontologist touring an almost complete theme park is tasked with protecting two kids after a power failure." },
    { id: 25, title: "The Shawshank Redemption", year: 1994, genres: ["Drama"], language: "English", rating: 9.3, moods: ["Emotional", "Thought-Provoking"], description: "Over the course of several years, two convicts form a friendship, seeking solace and ultimate redemption." },
    { id: 26, title: "Forrest Gump", year: 1994, genres: ["Drama", "Romance"], language: "English", rating: 8.8, moods: ["Feel Good", "Emotional"], description: "The history of the United States unfolds through the perspective of an Alabama man with an IQ of 75." },
    { id: 27, title: "Gladiator", year: 2000, genres: ["Action", "Adventure", "Drama"], language: "English", rating: 8.5, moods: ["Exciting"], description: "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family." },
    { id: 28, title: "The Godfather", year: 1972, genres: ["Crime", "Drama"], language: "English", rating: 9.2, moods: ["Thought-Provoking"], description: "The aging patriarch of an organized crime dynasty transfers control of his empire to his reluctant son." },
    { id: 29, title: "Avengers: Endgame", year: 2019, genres: ["Action", "Sci-Fi", "Adventure"], language: "English", rating: 8.4, moods: ["Exciting", "Futuristic"], description: "The Avengers assemble once more to reverse Thanos' actions and restore balance to the universe." },
    { id: 30, title: "Spider-Man: Into the Spider-Verse", year: 2018, genres: ["Animation", "Action", "Adventure"], language: "English", rating: 8.4, moods: ["Exciting", "Feel Good"], description: "Teen Miles Morales becomes the Spider-Man of his universe and joins spider-powered allies." },

    { id: 31, title: "The Lord of the Rings: The Fellowship of the Ring", year: 2001, genres: ["Fantasy", "Adventure", "Action"], language: "English", rating: 8.8, moods: ["Exciting"], description: "A meek Hobbit from the Shire sets out on a journey to destroy the One Ring." },
    { id: 32, title: "Alien", year: 1979, genres: ["Horror", "Sci-Fi"], language: "English", rating: 8.5, moods: ["Scary", "Futuristic"], description: "The crew of a commercial spacecraft encounters a deadly alien lifeform." },
    { id: 33, title: "The Lion King", year: 1994, genres: ["Animation", "Drama", "Musical"], language: "English", rating: 8.5, moods: ["Feel Good", "Emotional"], description: "Lion prince Simba and his father are targeted by his bitter uncle Scar." },
    { id: 34, title: "Coco", year: 2017, genres: ["Animation", "Fantasy", "Musical"], language: "English", rating: 8.4, moods: ["Feel Good", "Emotional"], description: "Aspiring musician Miguel enters the Land of the Dead to find his great-great-grandfather." },
    { id: 35, title: "Joker", year: 2019, genres: ["Crime", "Drama", "Thriller"], language: "English", rating: 8.4, moods: ["Thought-Provoking", "Emotional"], description: "A mentally troubled stand-up comedian embarks on a downward spiral that leads to an iconic villain." },
    { id: 36, title: "Whiplash", year: 2014, genres: ["Drama", "Musical"], language: "English", rating: 8.5, moods: ["Thought-Provoking", "Exciting"], description: "A promising young drummer enrolls at a cut-throat music conservatory mentored by a ruthless instructor." },
    { id: 37, title: "La La Land", year: 2016, genres: ["Drama", "Romance", "Musical"], language: "English", rating: 8.0, moods: ["Romantic", "Feel Good"], description: "While navigating their careers in Los Angeles, a pianist and an actress fall in love." },
    { id: 38, title: "The Prestige", year: 2006, genres: ["Mystery", "Drama", "Sci-Fi"], language: "English", rating: 8.5, moods: ["Thought-Provoking"], description: "Two stage magicians in 1890s London engage in a battle to create the ultimate illusion." },
    { id: 39, title: "Get Out", year: 2017, genres: ["Horror", "Mystery", "Thriller"], language: "English", rating: 7.7, moods: ["Scary", "Thought-Provoking"], description: "A young African-American visits his white girlfriend's parents, uncovering a disturbing secret." },
    { id: 40, title: "Dune", year: 2021, genres: ["Sci-Fi", "Adventure", "Drama"], language: "English", rating: 8.0, moods: ["Futuristic", "Thought-Provoking"], description: "A noble family becomes embroiled in a war for control over the galaxy's most valuable asset." },

    { id: 41, title: "Oppenheimer", year: 2023, genres: ["Drama", "Documentary"], language: "English", rating: 8.9, moods: ["Thought-Provoking"], description: "The story of J. Robert Oppenheimer and his role in the development of the atomic bomb." },
    { id: 42, title: "Top Gun: Maverick", year: 2022, genres: ["Action", "Drama"], language: "English", rating: 8.3, moods: ["Exciting"], description: "After thirty years, Maverick leads TOPGUN's elite graduates on a dangerous specialized mission." },
    { id: 43, title: "Inside Out", year: 2015, genres: ["Animation", "Comedy", "Drama"], language: "English", rating: 8.1, moods: ["Feel Good", "Funny"], description: "After young Riley moves to a new city, her internal emotions conflict on how best to navigate her life." },
    { id: 44, title: "WALL-E", year: 2008, genres: ["Animation", "Sci-Fi", "Adventure"], language: "English", rating: 8.4, moods: ["Futuristic", "Feel Good"], description: "In a distant future, a waste-collecting robot embarks on a space journey that decides humanity's fate." },
    { id: 45, title: "Super Deluxe", year: 2019, genres: ["Crime", "Drama", "Thriller"], language: "Tamil", rating: 8.3, moods: ["Thought-Provoking"], description: "An unfaithful wife, an estranged father, a transgender woman, and teenagers find their destinies linked." },
    { id: 46, title: "Premam", year: 2015, genres: ["Romance", "Comedy", "Drama"], language: "Malayalam", rating: 8.3, moods: ["Feel Good", "Romantic"], description: "A young man's journey through love at three different stages of his life." },
    { id: 47, title: "Lagaan", year: 2001, genres: ["Drama", "Musical", "War"], language: "Hindi", rating: 8.1, moods: ["Feel Good", "Emotional"], description: "Villagers in Victorian India stake their future on a game of cricket against ruthless British officers." },
    { id: 48, title: "Minari", year: 2020, genres: ["Drama"], language: "Korean", rating: 7.4, moods: ["Emotional", "Thought-Provoking"], description: "A Korean-American family moves to an Arkansas farm in search of their own American dream." },
    { id: 49, title: "Free Solo", year: 2018, genres: ["Documentary"], language: "English", rating: 8.1, moods: ["Exciting", "Thought-Provoking"], description: "Follow Alex Honnold as he attempts to perform a free solo climb of El Capitan in Yosemite National Park." },
    { id: 50, title: "Schindler's List", year: 1993, genres: ["Drama", "War"], language: "English", rating: 9.0, moods: ["Emotional", "Thought-Provoking"], description: "In German-occupied Poland during WWII, industrialist Oskar Schindler saves his Jewish workforce." }
];

const ALL_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", 
    "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi", 
    "Thriller", "War", "Musical"
];

const ALL_MOODS = [
    "😊 Feel Good", "😂 Funny", "🔥 Exciting", "🧠 Thought-Provoking",
    "❤️ Romantic", "👻 Scary", "🚀 Futuristic", "🎭 Emotional"
];

/* ==========================================================================
   2. STATE MANAGEMENT
   ========================================================================== */
let userState = {
    selectedGenres: new Set(["Sci-Fi", "Drama"]),
    selectedLanguage: "Any Language",
    yearFrom: 2010,
    yearTo: 2025,
    anyYear: false,
    minRating: 0,
    selectedMoods: new Set(["🧠 Thought-Provoking"]),
    favoriteMovieId: null,
    recCount: 5
};

/* ==========================================================================
   3. INITIALIZATION & NAVIGATION
   ========================================================================== */
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    renderGenreChips();
    renderMoodChips();
    populateFavoriteMovieDropdown();
    renderDiscoverFeatured();
    initYearControls();
    
    // Attach event listeners
    document.getElementById('btn-recommend').addEventListener('click', handleRecommendMovies);
    document.getElementById('btn-reset').addEventListener('click', resetPreferences);
    document.getElementById('btn-start-exploring').addEventListener('click', () => switchTab('tab-recommendations'));
    document.getElementById('btn-relax-filters').addEventListener('click', relaxFiltersAndRetry);
});

function initNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            switchTab(targetTab);
        });
    });
}

function switchTab(tabId) {
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
    });
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.toggle('active', pane.id === tabId);
    });
}

function initYearControls() {
    const anyYearCheckbox = document.getElementById('pref-any-year');
    const rangeContainer = document.getElementById('year-range-container');
    
    anyYearCheckbox.addEventListener('change', () => {
        userState.anyYear = anyYearCheckbox.checked;
        if (anyYearCheckbox.checked) {
            rangeContainer.style.opacity = '0.4';
            rangeContainer.style.pointerEvents = 'none';
        } else {
            rangeContainer.style.opacity = '1';
            rangeContainer.style.pointerEvents = 'auto';
        }
    });
}

/* ==========================================================================
   4. UI COMPONENT RENDERING
   ========================================================================== */
function renderGenreChips() {
    const container = document.getElementById('genre-chips');
    container.innerHTML = '';
    ALL_GENRES.forEach(genre => {
        const chip = document.createElement('div');
        chip.className = `chip ${userState.selectedGenres.has(genre) ? 'selected' : ''}`;
        chip.textContent = genre;
        chip.addEventListener('click', () => {
            if (userState.selectedGenres.has(genre)) {
                userState.selectedGenres.delete(genre);
            } else {
                userState.selectedGenres.add(genre);
            }
            chip.classList.toggle('selected');
        });
        container.appendChild(chip);
    });
}

function renderMoodChips() {
    const container = document.getElementById('mood-chips');
    container.innerHTML = '';
    ALL_MOODS.forEach(mood => {
        const chip = document.createElement('div');
        chip.className = `chip ${userState.selectedMoods.has(mood) ? 'selected' : ''}`;
        chip.textContent = mood;
        chip.addEventListener('click', () => {
            if (userState.selectedMoods.has(mood)) {
                userState.selectedMoods.delete(mood);
            } else {
                userState.selectedMoods.add(mood);
            }
            chip.classList.toggle('selected');
        });
        container.appendChild(chip);
    });
}

function populateFavoriteMovieDropdown() {
    const select = document.getElementById('pref-favorite-movie');
    const sortedMovies = [...MOVIE_DATABASE].sort((a, b) => a.title.localeCompare(b.title));
    sortedMovies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = `${movie.title} (${movie.year}) [${movie.language}]`;
        select.appendChild(option);
    });
}

function renderDiscoverFeatured() {
    const grid = document.getElementById('discover-featured-grid');
    grid.innerHTML = '';
    const featured = MOVIE_DATABASE.slice(0, 6);
    featured.forEach(movie => {
        const card = createMovieCardHTML(movie, null, false);
        grid.appendChild(card);
    });
}

/* ==========================================================================
   5. PREFERENCE COLLECTION & VALIDATION
   ========================================================================== */
function collectPreferences() {
    userState.selectedLanguage = document.getElementById('pref-language').value;
    userState.anyYear = document.getElementById('pref-any-year').checked;
    userState.yearFrom = parseInt(document.getElementById('pref-year-from').value) || 1950;
    userState.yearTo = parseInt(document.getElementById('pref-year-to').value) || 2026;
    userState.minRating = parseFloat(document.getElementById('pref-rating').value) || 0;
    userState.recCount = parseInt(document.getElementById('pref-rec-count').value) || 5;
    
    const favId = document.getElementById('pref-favorite-movie').value;
    userState.favoriteMovieId = favId ? parseInt(favId) : null;
}

function validatePreferences() {
    const errorDiv = document.getElementById('form-error-msg');
    errorDiv.classList.add('hidden');
    errorDiv.textContent = '';

    if (userState.selectedGenres.size === 0) {
        errorDiv.textContent = '⚠️ Please select at least one genre.';
        errorDiv.classList.remove('hidden');
        return false;
    }

    if (!userState.anyYear && userState.yearFrom > userState.yearTo) {
        errorDiv.textContent = '⚠️ Invalid Year Range: "From Year" cannot be greater than "To Year".';
        errorDiv.classList.remove('hidden');
        return false;
    }

    return true;
}

/* ==========================================================================
   6. RECOMMENDATION SCORING ALGORITHM
   Weights:
   - Genre match         = 35%
   - Language match      = 15%
   - Year match          = 10%
   - Rating              = 15%
   - Mood match          = 10%
   - Favorite similarity = 5%
   - Feature score       = 10%
   Total = 100% (Zero Math.random() - Fully Deterministic)
   ========================================================================== */
function computeDeterministicMatchScore(movie, userPrefs) {
    // 1. Genre Score (Weight: 35%)
    const selectedGenresArr = Array.from(userPrefs.selectedGenres);
    const matchedGenres = movie.genres.filter(g => userPrefs.selectedGenres.has(g));
    const genreScore = selectedGenresArr.length > 0 ? (matchedGenres.length / selectedGenresArr.length) : 0.5;

    // 2. Language Match Score (Weight: 15%)
    let languageScore = 1.0;
    if (userPrefs.selectedLanguage !== "Any Language") {
        languageScore = (movie.language === userPrefs.selectedLanguage) ? 1.0 : 0.0;
    }

    // 3. Year Match Score (Weight: 10%)
    let yearScore = 1.0;
    if (!userPrefs.anyYear) {
        if (movie.year < userPrefs.yearFrom || movie.year > userPrefs.yearTo) {
            yearScore = 0.2; // Minor penalty outside target range
        }
    }

    // 4. Rating Score (Weight: 15%)
    const ratingScore = movie.rating / 10.0;

    // 5. Mood Match Score (Weight: 10%)
    const selectedMoodsArr = Array.from(userPrefs.selectedMoods);
    let moodScore = 0.5; // Default neutral if no mood selected
    let matchedMoods = [];
    if (selectedMoodsArr.length > 0) {
        matchedMoods = movie.moods ? movie.moods.filter(m => userPrefs.selectedMoods.has(m)) : [];
        moodScore = matchedMoods.length / selectedMoodsArr.length;
    }

    // 6. Favorite Movie Similarity Score (Weight: 5%)
    let favSimilarityScore = 0.0;
    let favMovieObj = null;
    if (userPrefs.favoriteMovieId) {
        favMovieObj = MOVIE_DATABASE.find(m => m.id === userPrefs.favoriteMovieId);
        if (favMovieObj) {
            const commonGenres = movie.genres.filter(g => favMovieObj.genres.includes(g));
            favSimilarityScore = commonGenres.length / Math.max(favMovieObj.genres.length, 1);
        }
    }

    // 7. Internal Feature Representation Score (Weight: 10%)
    // Deterministic trigonometric feature projection
    const f1 = genreScore * Math.PI;
    const f2 = ratingScore * Math.PI;
    const featureScore = (Math.sin(f1) * 0.5 + Math.cos(f2) * 0.5 + 1) / 2;

    // Combined Weighted Final Score (0% to 100%)
    const finalScoreRaw = (
        genreScore * 0.35 +
        languageScore * 0.15 +
        yearScore * 0.10 +
        ratingScore * 0.15 +
        moodScore * 0.10 +
        favSimilarityScore * 0.05 +
        featureScore * 0.10
    ) * 100;

    const finalMatchScore = Math.min(Math.max(Math.round(finalScoreRaw), 0), 100);

    // Generate Dynamic Explanation
    const explanation = generateDynamicExplanation(movie, matchedGenres, languageScore, yearScore, favMovieObj, userPrefs);

    return {
        finalMatchScore,
        matchedGenres,
        explanation
    };
}

function generateDynamicExplanation(movie, matchedGenres, languageScore, yearScore, favMovieObj, userPrefs) {
    const reasons = [];

    if (matchedGenres.length > 0) {
        reasons.push(`Matches your ${matchedGenres.join(' and ')} preference${matchedGenres.length > 1 ? 's' : ''}`);
    }

    if (userPrefs.selectedLanguage !== "Any Language" && languageScore === 1.0) {
        reasons.push(`matches your selected ${userPrefs.selectedLanguage} language`);
    }

    if (!userPrefs.anyYear && yearScore === 1.0) {
        reasons.push(`released within your ${userPrefs.yearFrom}–${userPrefs.yearTo} era`);
    }

    if (favMovieObj && movie.genres.some(g => favMovieObj.genres.includes(g))) {
        reasons.push(`shares genres with your favorite movie (${favMovieObj.title})`);
    }

    if (reasons.length === 0) {
        return `Top-rated choice with a ${movie.rating} rating.`;
    }

    const firstReason = reasons[0].charAt(0).toUpperCase() + reasons[0].slice(1);
    const rest = reasons.slice(1).join(', ');
    return rest ? `${firstReason}, ${rest}.` : `${firstReason}.`;
}

/* ==========================================================================
   7. RECOMMENDATION GENERATION WORKFLOW
   ========================================================================== */
function handleRecommendMovies() {
    collectPreferences();
    if (!validatePreferences()) return;

    showProcessingModal(() => {
        executeRecommendationPipeline();
    });
}

async function executeRecommendationPipeline() {
    const payload = {
        genres: Array.from(userState.selectedGenres),
        language: userState.selectedLanguage,
        year_from: userState.yearFrom,
        year_to: userState.yearTo,
        any_year: userState.anyYear,
        min_rating: userState.minRating,
        rec_count: userState.recCount,
        moods: Array.from(userState.selectedMoods),
        favorite_movie_id: userState.favoriteMovieId
    };

    try {
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            const data = await response.json();
            if (data.status === "empty") {
                document.getElementById('results-container').classList.add('hidden');
                document.getElementById('preference-summary-box').classList.add('hidden');
                document.getElementById('empty-results-box').classList.remove('hidden');
                return;
            } else if (data.status === "success") {
                document.getElementById('empty-results-box').classList.add('hidden');
                renderPreferenceSummary();
                renderServerRecommendations(data);
                return;
            }
        }
    } catch (e) {
        console.warn("API call failed, running local deterministic recommender:", e);
    }

    // Local Fallback Execution
    let candidates = MOVIE_DATABASE.filter(movie => {
        if (userState.favoriteMovieId && movie.id === userState.favoriteMovieId) return false;
        if (userState.selectedLanguage !== "Any Language" && movie.language !== userState.selectedLanguage) return false;
        if (movie.rating < userState.minRating) return false;
        if (!userState.anyYear) {
            if (movie.year < userState.yearFrom || movie.year > userState.yearTo) return false;
        }
        return true;
    });

    if (candidates.length === 0) {
        document.getElementById('results-container').classList.add('hidden');
        document.getElementById('preference-summary-box').classList.add('hidden');
        document.getElementById('empty-results-box').classList.remove('hidden');
        return;
    }

    document.getElementById('empty-results-box').classList.add('hidden');

    const scoredMovies = candidates.map(movie => {
        const result = computeDeterministicMatchScore(movie, userState);
        return { ...movie, ...result };
    });

    scoredMovies.sort((a, b) => {
        if (b.finalMatchScore !== a.finalMatchScore) return b.finalMatchScore - a.finalMatchScore;
        if (b.rating !== a.rating) return b.rating - a.rating;
        return b.year - a.year;
    });

    const topRecommendations = scoredMovies.slice(0, userState.recCount);

    renderPreferenceSummary();
    renderRecommendations(topRecommendations);
}

function renderServerRecommendations(data) {
    const heroContainer = document.getElementById('top-match-hero');
    const gridContainer = document.getElementById('recommendations-grid');
    heroContainer.innerHTML = '';
    gridContainer.innerHTML = '';

    const top = data.top_match;
    if (top) {
        const matchPct = top.match_percentage || `${Math.round(top.quantum_score * 100)}%`;
        const card = document.createElement('div');
        card.className = 'glass-card hero-card';
        card.innerHTML = `
            <div class="hero-rank">🏆 #1 BEST MATCH (VQC SCORE)</div>
            <div class="hero-details">
                <h3>${top.title}</h3>
                <div class="hero-meta">${top.year} • ${top.genres.join(' • ')} • ${top.language}</div>
                <div class="hero-meta">⭐ ${top.rating} / 10</div>
                <p class="movie-desc">${top.description || ''}</p>
                <div class="why-box">
                    <strong>Quantum VQC Alignment Score:</strong> ${top.quantum_score} (${matchPct})
                </div>
            </div>
            <div class="scores-block">
                <div class="score-row">
                    <span>QUANTUM MATCH SCORE</span>
                    <span class="score-val">${matchPct}</span>
                </div>
                <div class="match-bar-container">
                    <div class="match-bar-fill" style="width: ${matchPct};"></div>
                </div>
            </div>
        `;
        heroContainer.appendChild(card);
    }

    const recs = data.recommendations || [];
    recs.forEach((movie, index) => {
        const rank = movie.rank || (index + 2);
        const matchPct = movie.match_percentage || `${Math.round(movie.quantum_score * 100)}%`;
        const card = document.createElement('div');
        card.className = 'glass-card movie-card';
        card.innerHTML = `
            <div>
                <div class="card-header">
                    <span class="movie-title">${movie.title}</span>
                    <span class="card-rank">#${rank}</span>
                </div>
                <div class="movie-meta">${movie.year} • ${movie.genres.join(', ')} • ${movie.language}</div>
                <div class="movie-meta">⭐ ${movie.rating} / 10</div>
                ${movie.description ? `<p class="movie-desc">${movie.description}</p>` : ''}
            </div>
            <div>
                <div class="scores-block">
                    <div class="score-row">
                        <span>Quantum Score</span>
                        <span class="score-val">${matchPct}</span>
                    </div>
                    <div class="match-bar-container">
                        <div class="match-bar-fill" style="width: ${matchPct};"></div>
                    </div>
                </div>
            </div>
        `;
        gridContainer.appendChild(card);
    });

    document.getElementById('results-container').classList.remove('hidden');
}

function relaxFiltersAndRetry() {
    document.getElementById('pref-language').value = "Any Language";
    document.getElementById('pref-rating').value = "0";
    document.getElementById('pref-any-year').checked = true;
    
    const rangeContainer = document.getElementById('year-range-container');
    rangeContainer.style.opacity = '0.4';
    rangeContainer.style.pointerEvents = 'none';

    collectPreferences();
    executeRecommendationPipeline();
}

/* ==========================================================================
   8. RESULTS RENDERING
   ========================================================================== */
function renderPreferenceSummary() {
    const box = document.getElementById('preference-summary-box');
    const container = document.getElementById('summary-badges');
    container.innerHTML = '';

    const genresList = Array.from(userState.selectedGenres).join(', ');
    addSummaryBadge(container, `🎬 ${genresList}`);
    addSummaryBadge(container, `🌐 ${userState.selectedLanguage}`);
    
    if (userState.anyYear) {
        addSummaryBadge(container, `📅 Any Year`);
    } else {
        addSummaryBadge(container, `📅 ${userState.yearFrom}–${userState.yearTo}`);
    }

    if (userState.minRating > 0) {
        addSummaryBadge(container, `⭐ ${userState.minRating}+`);
    }

    if (userState.selectedMoods.size > 0) {
        addSummaryBadge(container, `🎭 ${Array.from(userState.selectedMoods).join(', ')}`);
    }

    if (userState.favoriteMovieId) {
        const fav = MOVIE_DATABASE.find(m => m.id === userState.favoriteMovieId);
        if (fav) addSummaryBadge(container, `❤️ ${fav.title}`);
    }

    box.classList.remove('hidden');
}

function addSummaryBadge(parent, text) {
    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = text;
    parent.appendChild(badge);
}

function renderRecommendations(movies) {
    const heroContainer = document.getElementById('top-match-hero');
    const gridContainer = document.getElementById('recommendations-grid');
    heroContainer.innerHTML = '';
    gridContainer.innerHTML = '';

    if (movies.length === 0) return;

    // Render #1 Best Match Featured Card
    const topMovie = movies[0];
    heroContainer.appendChild(createHeroCardHTML(topMovie));

    // Render remaining recommendations (#2, #3, etc.)
    movies.slice(1).forEach((movie, index) => {
        const rank = index + 2;
        const card = createMovieCardHTML(movie, rank, true);
        gridContainer.appendChild(card);
    });

    document.getElementById('results-container').classList.remove('hidden');
}

function createHeroCardHTML(movie) {
    const card = document.createElement('div');
    card.className = 'glass-card hero-card';
    card.innerHTML = `
        <div class="hero-rank">🏆 #1 BEST MATCH</div>
        <div class="hero-details">
            <h3>${movie.title}</h3>
            <div class="hero-meta">${movie.year} • ${movie.genres.join(' • ')} • ${movie.language}</div>
            <div class="hero-meta">⭐ ${movie.rating} / 10</div>
            <p class="movie-desc">${movie.description}</p>
            <div class="why-box">
                <strong>Why you'll like it:</strong> ${movie.explanation}
            </div>
        </div>
        <div class="scores-block">
            <div class="score-row">
                <span>MATCH SCORE</span>
                <span class="score-val">${movie.finalMatchScore}%</span>
            </div>
            <div class="match-bar-container">
                <div class="match-bar-fill" style="width: ${movie.finalMatchScore}%;"></div>
            </div>
        </div>
    `;
    return card;
}

function createMovieCardHTML(movie, rank, showScores) {
    const card = document.createElement('div');
    card.className = 'glass-card movie-card';
    card.innerHTML = `
        <div>
            <div class="card-header">
                <span class="movie-title">${movie.title}</span>
                ${rank ? `<span class="card-rank">#${rank}</span>` : ''}
            </div>
            <div class="movie-meta">${movie.year} • ${movie.genres.join(', ')} • ${movie.language}</div>
            <div class="movie-meta">⭐ ${movie.rating} / 10</div>
            <p class="movie-desc">${movie.description}</p>
        </div>
        <div>
            ${showScores ? `
            <div class="scores-block">
                <div class="score-row">
                    <span>Match Score</span>
                    <span class="score-val">${movie.finalMatchScore}%</span>
                </div>
                <div class="match-bar-container">
                    <div class="match-bar-fill" style="width: ${movie.finalMatchScore}%;"></div>
                </div>
            </div>
            <div class="why-box">
                ${movie.explanation}
            </div>
            ` : ''}
        </div>
    `;
    return card;
}

/* ==========================================================================
   9. MODAL ANIMATION & RESET
   ========================================================================== */
function showProcessingModal(onComplete) {
    const modal = document.getElementById('processing-modal');
    modal.classList.remove('hidden');

    const steps = ['pstep-1', 'pstep-2', 'pstep-3'];

    steps.forEach(id => {
        const el = document.getElementById(id);
        el.className = 'step-item';
    });

    let currentStep = 0;
    const interval = setInterval(() => {
        if (currentStep < steps.length) {
            document.getElementById(steps[currentStep]).classList.add('done');
            currentStep++;
        } else {
            clearInterval(interval);
            setTimeout(() => {
                modal.classList.add('hidden');
                if (onComplete) onComplete();
            }, 250);
        }
    }, 300); // 0.9s processing duration
}

function resetPreferences() {
    userState.selectedGenres = new Set(["Sci-Fi", "Drama"]);
    userState.selectedLanguage = "Any Language";
    userState.yearFrom = 2010;
    userState.yearTo = 2025;
    userState.anyYear = false;
    userState.minRating = 0;
    userState.selectedMoods = new Set(["🧠 Thought-Provoking"]);
    userState.favoriteMovieId = null;
    userState.recCount = 5;

    document.getElementById('pref-language').value = "Any Language";
    document.getElementById('pref-year-from').value = 2010;
    document.getElementById('pref-year-to').value = 2025;
    document.getElementById('pref-any-year').checked = false;
    document.getElementById('pref-rating').value = "0";
    document.getElementById('pref-rec-count').value = "5";
    document.getElementById('pref-favorite-movie').value = "";

    const rangeContainer = document.getElementById('year-range-container');
    rangeContainer.style.opacity = '1';
    rangeContainer.style.pointerEvents = 'auto';

    renderGenreChips();
    renderMoodChips();

    document.getElementById('results-container').classList.add('hidden');
    document.getElementById('preference-summary-box').classList.add('hidden');
    document.getElementById('empty-results-box').classList.add('hidden');
    document.getElementById('form-error-msg').classList.add('hidden');
}
