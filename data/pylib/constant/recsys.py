# About features from/to ETL.
USER_ID_COL = "user_id"
ITEM_ID_COL = "item_id"
ITEM_IDX_COL = "item_idx"
SCORE_COL = "score"
WEIGHT_COL = "weight"
TIMESTAMP_COL = "timestamp"
TARGET_INTERACTION_COL = "target_interaction"
OTHER_ITEM_ID_COL = "other_item_id"
OTHER_ITEM_IDX_COL = "other_item_idx"
EMBEDDING_VECTOR_COL = "embedding_vector"
SIMILARITY_COL = "similarity"
MODEL_COL = "model"
EVENT_COL = "event"  # Will be renamed to `property`.
COUNT_COL = "count"  # Will be Deprecated.

# Additional Features.
OWNED_ITEM_IDS_COL = "owned_items"

# About features inside ML(Data Processor, Data Transformer, Model).
USER_INPUTS_COL = "user_inputs"
ITEM_INPUTS_COL = "item_inputs"
INPUTS_COL = "inputs"
TARGETS_COL = "targets"
NEXT_TARGET_COL = "next_target"
ALL_TARGETS_COL = "all_targets"
SEQ_LEN_COL = "seq_len"
TARGET_IDX_COL = "target_idx"

EXCLUSION_COL = "exclusion"

GENDER_COL = "gender"
GENDER_IDX_COL = "gender_idx"
GENERATION_COL = "generation"
GENERATION_IDX_COL = "generation_idx"

INPUT_WEIGHTS_COL = "input_weights"
TARGET_WEIGHTS_COL = "target_weights"

INPUT_TIMESTAMPS_COL = "input_timestamps"
TARGET_TIMESTAMPS_COL = "target_timestamps"

POSITIVE_CONTEXTS_COL = "positive_contexts"
NEGATIVE_CONTEXTS_COL = "negative_contexts"

# About hive.
EXP_NAME_COL = "exp_name"

# About Metadata.
META_FILENAME = "meta.json"
TRAIN_META_FILENAME = "train_meta.json"
TEST_META_FILENAME = "test_meta.json"
ITEM_FREQUENCY_META_KEY = "item_frequency"

# About preprocessor.
PREPROCESSOR_NAME = "preprocessor"
USER_PREPROCESSOR_NAME = "user_preprocessor"
ITEM_PREPROCESSOR_NAME = "item_preprocessor"
GENDER_PREPROCESSOR_NAME = "gender_preprocessor"
GENERATION_PREPROCESSOR_NAME = "generation_preprocessor"

# About Dataset.
TRAIN_DATASET_NAME = "train"
DEV_DATASET_NAME = "dev"
TEST_DATASET_NAME = "test"
