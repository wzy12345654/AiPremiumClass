from models.models import ModelType
from datasets.datasets import DatasetType
from models.optimizers.optimizers import OptimizerType
from models.losses.loss_functions import LossFunctionType


config = dict(
    model = ModelType.MY_TRANSFORMER,
    hidden_size = 128,
    ff_size = 215,
    n_heads = 8,
    encoder_layers = 6,
    decoder_layers = 6,
    dropout = 0.1,
    
    dataset = DatasetType.QA_ENCODER_DECODER,
    batch_size = 4,
    data_root = './邪王真眼/datasets/corpus_p11',
    encoder_max_length = 30,
    decoder_max_length = 30,

    optimizer = OptimizerType.Adam,
    lr = 0.001,
    momentum = 0.9,
    weight_decay = 0.0001,

    loss_function = LossFunctionType.CROSS_ENTROPY,
    
    epochs = 50,
    result_path = './邪王真眼/week09/result'
)
