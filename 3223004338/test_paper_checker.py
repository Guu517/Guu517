"""
    单元测试
"""

import unittest
import os
import tempfile
import sys
from paper_checker import PaperChecker

class TestPaperChecker(unittest.TestCase):
    """论文查重器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.checker = PaperChecker()
        self.test_dir = tempfile.mkdtemp()
        print(f"测试目录: {self.test_dir}")
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_identical_texts(self):
        """测试完全相同文本"""
        text1 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        text2 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        similarity = self.checker.calculate_similarity(text1, text2)
        self.assertAlmostEqual(similarity, 1.0, places=1)
    
    def test_example_case(self):
        """测试题目示例"""
        text1 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        text2 = "今天是周天，天气晴朗，我晚上要去看电影。"
        similarity = self.checker.calculate_similarity(text1, text2)
        self.assertGreater(similarity, 0.6)
        self.assertLess(similarity, 0.9)
    
    def test_completely_different(self):
        """测试完全不同文本"""
        text1 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        text2 = "明天是星期一，天气雨，我打算在家休息。"
        similarity = self.checker.calculate_similarity(text1, text2)
        self.assertLess(similarity, 0.3)
    
    def test_empty_text(self):
        """测试空文本"""
        text1 = "正常文本内容"
        text2 = ""
        similarity = self.checker.calculate_similarity(text1, text2)
        self.assertEqual(similarity, 0.0)
    
    def test_special_characters(self):
        """测试特殊字符处理"""
        text1 = "Hello! 你好！@#$%^&*()"
        text2 = "Hello 你好"
        similarity = self.checker.calculate_similarity(text1, text2)
        self.assertGreater(similarity, 0.5)
    
    def test_long_text(self):
        """测试长文本"""
        base_text = "机器学习是人工智能的重要分支。"
        text1 = base_text * 50
        text2 = base_text * 30 + "深度学习是机器学习的一种方法。" * 20
        similarity = self.checker.calculate_similarity(text1, text2)
        self.assertGreater(similarity, 0.3)
    
    def test_academic_words(self):
        """测试学术词汇识别"""
        text1 = "机器学习算法在数据分析中应用广泛。"
        text2 = "数据分析常用机器学习算法。"
        similarity = self.checker.calculate_similarity(text1, text2)
        self.assertGreater(similarity, 0.6)
    
    def test_file_operations(self):
        """测试文件操作"""
        # 创建测试文件
        orig_file = os.path.join(self.test_dir, "orig.txt")
        copy_file = os.path.join(self.test_dir, "copy.txt")
        output_file = os.path.join(self.test_dir, "result.txt")
        
        with open(orig_file, 'w', encoding='utf-8') as f:
            f.write("原始论文内容。")
        
        with open(copy_file, 'w', encoding='utf-8') as f:
            f.write("抄袭版论文内容。")
        
        similarity = self.checker.check_plagiarism(orig_file, copy_file, output_file)
        
        # 验证输出文件
        self.assertTrue(os.path.exists(output_file))
        with open(output_file, 'r', encoding='utf-8') as f:
            result = float(f.read())
        
        self.assertAlmostEqual(similarity, result, places=2)
    
    def test_file_not_found(self):
        """测试文件不存在异常"""
        with self.assertRaises(FileNotFoundError):
            self.checker.check_plagiarism(
                "nonexistent1.txt", 
                "nonexistent2.txt", 
                "output.txt"
            )
    
    def test_encoding_detection(self):
        """测试编码自动检测"""
        # 创建GBK编码文件
        gbk_file = os.path.join(self.test_dir, "gbk.txt")
        with open(gbk_file, 'w', encoding='gbk') as f:
            f.write("中文内容测试")
        
        # 创建UTF-8编码文件
        utf8_file = os.path.join(self.test_dir, "utf8.txt")
        with open(utf8_file, 'w', encoding='utf-8') as f:
            f.write("中文内容测试")
        
        output_file = os.path.join(self.test_dir, "output.txt")
        
        # 应该能正常处理不同编码
        similarity = self.checker.check_plagiarism(gbk_file, utf8_file, output_file)
        self.assertAlmostEqual(similarity, 1.0, places=1)

def run_performance_test():
    """性能测试函数"""
    checker = PaperChecker()
    
    # 生成测试文本
    base_text = "机器学习是人工智能的重要分支，深度学习是机器学习的一种方法。" * 100
    
    import time
    start_time = time.time()
    
    similarity = checker.calculate_similarity(base_text, base_text)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"性能测试结果:")
    print(f"文本长度: {len(base_text)} 字符")
    print(f"计算时间: {duration:.3f} 秒")
    print(f"相似度: {similarity:.4f}")
    
    # 性能要求：5秒内完成
    assert duration < 5.0, "性能测试失败：计算时间超过5秒"
    assert similarity == 1.0, "性能测试失败：相同文本相似度不为1"

if __name__ == '__main__':
    # 运行单元测试
    unittest.main(verbosity=2)
    
    # 运行性能测试
    print("\n" + "="*50)
    print("性能测试")
    print("="*50)
    run_performance_test()