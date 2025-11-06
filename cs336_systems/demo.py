# check_cuda.py
import torch

def main():
    print(f"PyTorch 版本 : {torch.__version__}")
    print(f"CUDA 可用    : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA 版本    : {torch.version.cuda}")
        print(f"设备名称     : {torch.cuda.get_device_name(0)}")
        print(f"设备数量     : {torch.cuda.device_count()}")

        # 简单申请 1×1 的张量，触发 CUDA 上下文
        x = torch.tensor([1.0], device='cuda')
        print(f"测试张量     : {x}")
        print(f"当前设备     : {x.device}")

        # 打印显存占用（MB）
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved  = torch.cuda.memory_reserved()  / 1024**2
        print(f"已分配显存   : {allocated:.2f} MB")
        print(f"已预留显存   : {reserved:.2f} MB")
    else:
        print("没有找到可用的 CUDA 设备，使用 CPU。")

if __name__ == "__main__":
    main()