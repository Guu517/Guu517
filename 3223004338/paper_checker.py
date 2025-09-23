"""
    论文查重核心模块
    实现基于TF-IDF和余弦相似度的查重算法
"""

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from collections import defaultdict
import time

class PaperChecker:
    """
    论文查重器类
    使用TF-IDF向量化和余弦相似度算法
    """
    
    def __init__(self):
        """初始化查重器"""
        # 配置jieba分词
        jieba.setLogLevel('ERROR')  # 关闭jieba日志
        
        # 自定义词典 - 添加学术常用词
        self._add_academic_words()
        
        # 停用词列表
        self.stop_words = self._load_stop_words()
        
        # 初始化TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            max_features=5000  # 限制特征数量提高性能
        )
    
    def _add_academic_words(self):
        """添加学术常用词到分词词典"""
        academic_words = [
            '机器学习', '深度学习', '人工智能', '神经网络', '数据分析',
            '算法', '模型', '训练', '测试', '准确率', '召回率', 'F1分数',
            '预处理', '特征工程', '过拟合', '欠拟合', '交叉验证'
        ]
        for word in academic_words:
            jieba.add_word(word)
    
    def _load_stop_words(self):
        """加载停用词表"""
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', 
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', 
            '没有', '看', '好', '自己', '这个', '那个', '他', '她', '它', '我们',
            '你们', '他们', '这', '那', '哪', '怎么', '什么', '为什么', '因为',
            '所以', '但是', '虽然', '如果', '然后', '可以', '应该', '需要'
        }
        return stop_words
    
    def read_file(self, file_path):
        """
        读取文件内容
        支持多种编码格式，自动检测
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文件内容
            
        Raises:
            FileNotFoundError: 文件不存在
            UnicodeDecodeError: 编码错误
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if os.path.getsize(file_path) == 0:
            return ""
        
        # 尝试多种编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read().strip()
                    if content:
                        return content
                    else:
                        raise ValueError("文件内容为空")
            except UnicodeDecodeError:
                continue
        
        raise UnicodeDecodeError(f"无法解码文件: {file_path}")
    
    def preprocess_text(self, text):
        """
        文本预处理
        包括清洗、分词、去停用词
        
        Args:
            text: 原始文本
            
        Returns:
            list: 处理后的词列表
        """
        if not text or not text.strip():
            return []
        
        # 文本清洗：去除标点符号、数字、特殊字符
        text = self._clean_text(text)
        
        # 使用jieba进行分词
        words = jieba.cut(text)
        
        # 过滤停用词和单字词
        filtered_words = [
            word for word in words 
            if (word.strip() and 
                len(word) > 1 and 
                word not in self.stop_words and
                not word.isspace())
        ]
        
        return filtered_words
    
    def _clean_text(self, text):
        """
        清洗文本
        移除标点、数字等无关字符
        """
        # 移除标点符号
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        # 移除数字
        text = re.sub(r'\d+', '', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def calculate_similarity(self, text1, text2):
        """
        计算两段文本的相似度
        使用TF-IDF + 余弦相似度算法
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            float: 相似度得分 [0, 1]
        """
        start_time = time.time()
        
        # 预处理文本
        words1 = self.preprocess_text(text1)
        words2 = self.preprocess_text(text2)
        
        # 如果任意文本处理后为空，返回0相似度
        if not words1 or not words2:
            return 0.0
        
        # 转换为字符串用于TF-IDF
        text1_processed = ' '.join(words1)
        text2_processed = ' '.join(words2)
        
        try:
            # 计算TF-IDF向量
            tfidf_matrix = self.vectorizer.fit_transform([text1_processed, text2_processed])
            
            # 计算余弦相似度
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # 确保相似度在[0,1]范围内
            similarity = max(0.0, min(1.0, similarity))
            
            end_time = time.time()
            print(f"相似度计算完成，耗时: {(end_time - start_time):.3f}秒")
            
            return round(similarity, 4)
            
        except Exception as e:
            print(f"相似度计算错误: {e}")
            return 0.0
    
    def check_plagiarism(self, orig_file, copy_file, output_file):
        """
        论文查重主函数
        
        Args:
            orig_file: 原文文件路径
            copy_file: 抄袭版文件路径
            output_file: 输出文件路径
            
        Returns:
            float: 相似度得分
        """
        try:
            # 读取文件内容
            print("正在读取文件...")
            orig_text = self.read_file(orig_file)
            copy_text = self.read_file(copy_file)
            
            # 检查文件内容是否为空
            if not orig_text:
                raise ValueError("原文文件内容为空")
            if not copy_text:
                raise ValueError("抄袭版文件内容为空")
            
            print(f"原文长度: {len(orig_text)} 字符")
            print(f"抄袭版长度: {len(copy_text)} 字符")
            
            # 计算相似度
            print("正在计算相似度...")
            similarity = self.calculate_similarity(orig_text, copy_text)
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 写入结果文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"{similarity:.2f}")
            
            return similarity
            
        except FileNotFoundError as e:
            # 文件不存在时输出0.00
            self._write_error_output(output_file, "0.00")
            raise e
        except Exception as e:
            # 其他异常时输出0.00
            self._write_error_output(output_file, "0.00")
            raise e
    
    def _write_error_output(self, output_file, value):
        """错误时写入默认值"""
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(value)
        except Exception as e:
            print(f"写入输出文件错误: {e}")