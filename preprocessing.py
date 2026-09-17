"""
Data Preprocessing & Feature Pipeline Module (Root Proxy Re-export).
All core implementation details reside in `src.preprocessing`.
"""

from src.preprocessing import (
    download_and_load_data,
    get_dataset_statistics,
    encode_genres,
    temporal_train_test_split,
    build_user_profile_historical,
    build_interaction_features,
    fit_preprocessing_pipeline,
    transform_preprocessing_pipeline
)
