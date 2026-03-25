import torch
# 从你的模型包中导入我们写好的 API
from models import build_model, list_models

def test_single_model(model_name, input_shape, num_classes=10):
    print(f"\n{'='*50}")
    print(f"🚀 开始测试模型: {model_name.upper()}")
    print(f"{'='*50}")
    
    try:
        # 1. 使用工厂函数构建模型
        model = build_model(model_name, num_classes=num_classes)
        
        # 打印模型基本信息 (调用 BaseModel 里的通用方法)
        info = model.get_model_info()
        print(f"✅ 模型构建成功！")
        print(f"📊 总参数量: {info['total_params']:,} | 可训练参数: {info['trainable_params']:,}")
        
        # 2. 生成 Dummy Data (假数据)
        # 注意输入形状通常是 (Batch_Size, Channels, Height, Width)
        print(f"📦 生成假数据输入，形状: {input_shape}...")
        dummy_input = torch.randn(*input_shape)
        
        # 3. 执行前向传播
        print("⏳ 正在进行前向传播 (Forward Pass)...")
        
        # 如果你有 GPU 并且想顺便测试 GPU 显存，可以解除下面几行的注释
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        dummy_input = dummy_input.to(device)
        
        output = model(dummy_input)
        
        # 4. 验证输出形状
        print("🎉 前向传播成功！")
        print(f"🎯 输出张量形状: {output.shape}")
        
        # 简单分析一下输出
        if len(output.shape) == 2:
            print("👉 这是一个分类模型 (输出: Batch x Classes)")
        elif len(output.shape) == 4:
            print("👉 这是一个分割/密集预测模型 (输出: Batch x Classes x Height x Width)")
            
    except Exception as e:
        print(f"❌ 测试失败！捕获到异常：")
        print(f"👉 {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    # 检查注册表里有没有东西
    available_models = list_models()
    print(f"📚 当前注册表中的可用模型共有 {len(available_models)} 个: ")
    print(available_models)
    
    if len(available_models) == 0:
        print("⚠️ 警告：没有扫描到任何模型，请检查 models/__init__.py 的导入逻辑！")
    else:
        # 测试计划表
        # 我们挑选几个代表性的模型来测试，输入统一用 1 张 3 通道 224x224 的图片
        test_batch_size = 2
        image_shape = (test_batch_size, 3, 224, 224)
        target_classes = 10
        
        # models_to_test = ["vgg11", "resnet18", "resunet"]
        models_to_test = list_models()
        
        for name in models_to_test:
            if name in available_models:
                test_single_model(name, image_shape, target_classes)
            else:
                print(f"\n⚠️ 跳过 {name}：未在注册表中找到该模型。")
                
    print("\n🏁 所有测试流程执行完毕！")