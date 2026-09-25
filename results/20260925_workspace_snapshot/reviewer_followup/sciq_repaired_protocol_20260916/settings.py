from common import *
ARMS = ['input', 'output', 'hidden_r8', 'small_q']
BASE_MODELS = ['qwen3_06b_base', 'qwen25_15b_base']
LRS = [5e-5, 2e-4, 8e-4]
TUNING_SEED = 5001
CONFIRMATION_SEEDS = list(range(5002, 5007))
MAX_NEW_TOKENS = 128
GPUS = [1, 2, 3, 4, 5, 6]
PRIMARY_FAMILY_SIZE = 16
EFFICIENCY_FAMILY_SIZE = 4
