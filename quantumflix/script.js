/**
 * QUANTUMFLIX - MOVIE RECOMMENDATION SYSTEM
 * Standalone Vanilla JavaScript Application (ES6+)
 * Connected to Quantum-Assisted VQC Recommendation Backend API
 */

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
    { id: 20, title: "Pan's Labyrinth", year: 2006, genres: ["Fantasy", "Drama", "War"], language: "Spanish", rating: 8.2, moods: ["Thought-Provoking", "Scary"], description: "In the Falangist Spain of 1944, the bookish young stepdaughter of a sadistic army officer escapes into an eerie fantasy world." }
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

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    renderGenreChips();
    renderMoodChips();
    populateFavoriteMovieDropdown();
    renderDiscoverFeatured();
    initYearControls();
    
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
        console.warn("API call failed, running local recommender:", e);
    }
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

    box.classList.remove('hidden');
}

function addSummaryBadge(parent, text) {
    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = text;
    parent.appendChild(badge);
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
    `;
    return card;
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
    }, 300);
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
