# model_optimization.py
import torch
import torch.nn.utils.prune as prune
import torch.quantization as quantization

def optimize_model_for_cpu(model, img_size=256):
    """
    优化模型以提升CPU推理性能。包括将模型转换为脚本模式和启用推理优化。
    
    参数:
    - model: 需要优化的PyTorch模型。
    - img_size: 输入图像的尺寸(默认256)，用于设置优化模型的示例输入张量。
    
    返回:
    - optimized_model: 优化后的模型, 适合CPU推理。
    
    过程:
    1. 使用 `torch.jit.trace` 将模型转换为脚本模式，以减少推理开销。
    2. 使用 `torch.jit.optimize_for_inference` 进一步优化脚本模型，以提高推理速度。
    """
    tensor_input = torch.randn(1, 3, img_size, img_size)
    
    scripted_model = torch.jit.trace(model, tensor_input)
    
    optimized_model = torch.jit.optimize_for_inference(scripted_model)
    
    return optimized_model


def prune_model(model, prune_ratio=0.6):
    """
    对模型进行剪枝，减少模型参数，优化内存使用，提升推理性能。
    
    参数:
    - model: 需要剪枝的PyTorch模型。
    - prune_ratio: 剪枝率（默认0.6），即要剪除的参数比例。
    
    返回:
    - pruned_model: 剪枝后的模型。
    
    过程:
    1. 遍历模型的每一层，并在特定层上应用剪枝。剪枝的标准是通过对权重的L1范数排序，去掉指定比例的权重。
    2. `torch.nn.utils.prune.ln_structured` 可以选择对特定维度进行剪枝，如指定在L1范数基础上对每层的特定权重进行剪除。
    3. 这种剪枝方式能够减少模型的参数量，从而提升推理效率。
    """
    for name, module in model.named_modules():
        
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            
            prune.ln_structured(module, name="weight", amount=prune_ratio, n=1, dim=0)
            
    return model


def quantize_model(model):
    """
    对模型进行量化，将浮点模型转换为量化模型，以减少模型的大小和推理时间。
    
    参数:
    - model: 需要量化的PyTorch模型。
    
    返回:
    - quantized_model: 量化后的模型，适合在CPU上推理。
    
    过程:
    1. 对模型进行量化准备，这将插入伪量化模块以模拟量化行为。
    2. 将模型转换为量化模型。量化模型将浮点数据转换为int8格式，从而减少内存使用和计算需求。
    3. 最终得到的量化模型在低性能设备上（如移动设备、CPU）上更高效，推理时间较快。
    """
    model.qconfig = quantization.default_qconfig
    
    quantization.prepare(model, inplace=True)

    quantized_model = quantization.convert(model, inplace=False)
    
    return quantized_model
