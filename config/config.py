from yacs.config import CfgNode as CN

_C = CN()
_C.SYSTEM = CN()
_C.SYSTEM.PIN_MEMORY = True
_C.SYSTEM.NUM_WORKERS = 0
_C.SYSTEM.FEATURES_PATH = 'features/'
_C.SYSTEM.LOG_PATH = 'logs/'

_C.DATASET = CN()
_C.DATASET.NAME = 'cifar10'

_C.FEATURE_EXTRACTOR = CN()
_C.FEATURE_EXTRACTOR.NAME = 'esvit_swin_base'
_C.FEATURE_EXTRACTOR.NUM_BLOCKS = 4

_C.EXTRACT = CN()
_C.EXTRACT.BATCH_SIZE = 32

_C.TRAIN = CN()
_C.TRAIN.VERBOSE=0

_C.TRAIN.SGDOCSVM = CN()
_C.TRAIN.SGDOCSVM.HYPERPARAMS = [None]

_C.TRAIN.OCSVM = False
_C.TRAIN.ISOLATION_FOREST = False
_C.TRAIN.LOF = False
_C.TRAIN.METRIC = 'roc_auc'
_C.TRAIN.COST_MODEL = [{'tp':0, 'tn':0, 'fp':0.1, 'fn':1}]





def get_cfg_defaults():
    return _C.clone()

def save_cfg_default(path):
    cfg = get_cfg_defaults()
    with open(path, 'w') as f:
        f.write(cfg.dump())

save_cfg_default('config/default.yaml')