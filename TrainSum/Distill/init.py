from transformers import AutoModel, AutoConfig
import torch.nn.init as init

# 設定を取得してモデルを新規初期化
config = AutoConfig.from_pretrained("../distillLLAMA")
model = AutoModel.from_config(config)

model.save_pretrained("./initialized_distill_model")