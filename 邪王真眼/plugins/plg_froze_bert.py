from .plugins_base import PluginBase
from trainer.trainer import PluginType, TrainContext


class PluginFrozeBert(PluginBase):
    plugin_hooks = {
        PluginType.TRAIN_BEGIN: "froze_bert",
    }

    def froze_bert(self, ctx: TrainContext):
        for param in ctx.model.bert_classifier.bert.parameters():
            param.requires_grad_(False)
