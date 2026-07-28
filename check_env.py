#!/usr/bin/env python3
"""环境检查脚本 - Step 1 依赖验证"""

import sys

def check(name, import_path, min_version=None, attr="__version__"):
    """检查单个库"""
    try:
        mod = __import__(import_path)
        version = getattr(mod, attr, "unknown")
        print(f"  ✅ {name}: {version}")
        if min_version and version != "unknown":
            major = int(str(version).split('.')[0])
            if major < int(min_version.split('.')[0]):
                print(f"  ⚠️  版本过低，需要 >= {min_version}")
                return False
        return True
    except ImportError:
        print(f"  ❌ {name}: 未安装 (需要 >= {min_version})")
        return False
    except Exception as e:
        # 【修复】捕获所有其他错误（如DLL加载失败），不中断检查
        print(f"  ⚠️  {name}: 已安装但加载失败 - {str(e)[:60]}")
        return False

print("=" * 50)
print("Step 1 环境检查报告")
print("=" * 50)

all_ok = True

# Python版本
print(f"\n📌 Python: {sys.version.split()[0]}")
if sys.version_info < (3, 10):
    print("   ⚠️  需要 Python 3.10+")
    all_ok = False

# 核心库（Step 1 必需）
print("\n📦 Step 1 核心依赖:")
all_ok &= check("MuJoCo", "mujoco", "3.1.0")
all_ok &= check("Gymnasium", "gymnasium", "0.29.1")
all_ok &= check("NumPy", "numpy", "1.26.0")
all_ok &= check("Matplotlib", "matplotlib", "3.8.0")
all_ok &= check("tqdm", "tqdm")

# PyTorch（Step 2 才需要，现在不阻塞）
print("\n📦 PyTorch (Step 2 训练时才需要):")
has_torch = check("PyTorch", "torch", "2.2.0")
if not has_torch:
    print("   💡 提示: Step 1 不需要 PyTorch，可稍后解决")

# MuJoCo 功能测试
print("\n🔧 MuJoCo 功能测试:")
try:
    import mujoco
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
    print("  ✅ MuJoCo 基础加载正常")
except Exception as e:
    print(f"  ❌ MuJoCo 加载失败: {e}")
    all_ok = False

# 总结
print("\n" + "=" * 50)
if all_ok:
    print("🎉 Step 1 环境检查通过！可以开始运行代码。")
    if not has_torch:
        print("\n📌 下一步: 把代码文件放入项目目录，运行 collect_seed_data.py")
else:
    print("⚠️  Step 1 环境检查未通过，请按上方 ❌ 提示安装缺失项。")