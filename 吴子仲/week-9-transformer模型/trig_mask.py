import torch
# 方阵大小
sz = 10
# 生成一个方阵
mask = torch.ones(sz, sz)
# 下三角截取
mask = torch.tril(mask)
# print(mask)

# 条件填充
mask = mask.masked_fill(mask==0, value=float('-inf'))
mask = mask.masked_fill(mask==1, value=float(0.0))
# print(mask)