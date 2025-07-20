from models.models import ModelType
from datasets.datasets import DatasetType
from models.optimizers.optimizers import OptimizerType
from models.losses.loss_functions import LossFunctionType


config = dict(
    model = ModelType.BERT_PREDICTOR,
    num_layers = 6,
    dropout = 0.1,

    dataset = DatasetType.NextWordPrediction,
    batch_size = 64,
    data_root = './邪王真眼/datasets/corpus_p10',
    max_length = 20,

    optimizer = OptimizerType.Adam,
    lr = 0.01,

    loss_function = LossFunctionType.CROSS_ENTROPY,

    epochs = 5,
    result_path = './邪王真眼/previous_class10/result'
)
