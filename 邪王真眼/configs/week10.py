from models.models import ModelType
from datasets.datasets import DatasetType
from models.optimizers.optimizers import OptimizerType
from models.losses.loss_functions import LossFunctionType


config = dict(
    model = ModelType.BERT_CLASSIFIER,

    dataset = DatasetType.JD_COMMENTS,
    split_ratio = 0.7,
    batch_size = 16,
    max_length = 30,
    data_max_num = 5000,
    data_root = './邪王真眼/datasets/jd_comments',

    optimizer = OptimizerType.Adam,
    lr = 0.001,

    loss_function = LossFunctionType.CROSS_ENTROPY,

    epochs = 3,
    result_path = './邪王真眼/week10/result'
)
