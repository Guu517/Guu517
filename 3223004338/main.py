#!/usr/bin/env python3
"""
论文查重系统主程序
作者: [您的姓名]
学号: [您的学号]
"""

import sys
import os
from paper_checker import PaperChecker

def main():
    """
    主函数：处理命令行参数并执行查重
    """
    if len(sys.argv) != 4:
        print("错误: 参数数量不正确")
        print("用法: python main.py [原文文件] [抄袭版论文文件] [答案文件]")
        print("示例: python main.py orig.txt orig_add.txt ans.txt")
        sys.exit(1)
    
    orig_file = sys.argv[1]
    copy_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # 验证文件路径
    if not all([orig_file, copy_file, output_file]):
        print("错误: 文件路径参数不能为空")
        sys.exit(1)
    
    checker = PaperChecker()
    
    try:
        print("开始论文查重...")
        print(f"原文文件: {orig_file}")
        print(f"抄袭版文件: {copy_file}")
        print(f"输出文件: {output_file}")
        
        similarity = checker.check_plagiarism(orig_file, copy_file, output_file)
        
        print(f"查重完成！相似度: {similarity:.2%}")
        print(f"结果已保存到: {output_file}")
        
    except FileNotFoundError as e:
        print(f"文件错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()