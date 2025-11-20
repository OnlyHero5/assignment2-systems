import torch
import numpy as np

def test_fp16_native():

    result = torch.tensor(0.0, dtype=torch.float16)
    for _ in range(10000):
        result += torch.tensor(0.0001, dtype=torch.float16)
    return result.item()

def test_fp32_accumulator():

    result = torch.tensor(0.0, dtype=torch.float32)
    for _ in range(10000):
        result += torch.tensor(0.0001, dtype=torch.float16).float()
    return result.item()

def test_kahan_summation():
    sum_val = torch.tensor(0.0, dtype=torch.float16)
    compensation = torch.tensor(0.0, dtype=torch.float16)

    for _ in range(10000):
        value = torch.tensor(0.0001, dtype=torch.float16)
        
        y = value - compensation
        temp = sum_val + y
        compensation = (temp - sum_val) - y
        sum_val = temp
    
    return sum_val.item()

def test_pairwise_summation():

    values : list = [torch.tensor(0.0001, dtype=torch.float16) for _ in range(10000)]

    def pairwise_sum(arr):
        if len(arr) == 1:
            return arr[0]
        
        mid = len(arr) // 2
        left = pairwise_sum(arr[:mid])
        right = pairwise_sum(arr[mid:])
        return left + right
    
    return pairwise_sum(values).item()



def main():
    print("="*70)
    print("fp16累加数稳定性测试")
    print("="*70)
    print("理论值： 10000 * 0.0001 = 1.0")
    print("="*70)
    print()

    methods = [
        ("朴素 FP16", test_fp16_native),
        ("FP32 累加器", test_fp32_accumulator),
        ("Kahan 求和", test_kahan_summation),
        ("成对求和", test_pairwise_summation)
    ]
    results = {}
    for name, func in methods:
        val = func()
        error = abs(val - 1.0)
        results[name] = error
        print(f"{name}:")
        print(f" 结果: {val:.6f}")
        print(f" 误差: {error:.6f}")
        print()
    best_method = min(results, key=results.get)
    print("="*70)
    print(f"结论: {best_method} 误差最小")
    print("="*70)

if __name__ == "__main__":
    main()