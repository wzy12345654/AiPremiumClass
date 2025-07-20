from models.models import ModelType
from datasets.datasets import DatasetType
from models.optimizers.optimizers import OptimizerType
from models.losses.loss_functions import LossFunctionType


config = dict(
    model = ModelType.BERT_PREDICTOR,
    num_layers = 6,
    dropout = 0.1,

    dataset = DatasetType.QA,
    batch_size = 64,
    data_root = './邪王真眼/datasets/corpus_p11',
    max_length = 40,

    optimizer = OptimizerType.Adam,
    lr = 0.001,

    loss_function = LossFunctionType.CROSS_ENTROPY,
    
    epochs = 50,
    result_path = './邪王真眼/previous_class11/result'
)
