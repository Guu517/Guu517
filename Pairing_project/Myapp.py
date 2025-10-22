import sys
import random
import fractions
import re
from collections import defaultdict

class Fraction:
    """处理分数的类，支持带分数表示"""

    def __init__(self, numerator=0, denominator=1):
        if denominator == 0:
            raise ValueError("分母不能为零")
        # 确保分母为正
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        # 约分
        common_divisor = self.gcd(abs(numerator), abs(denominator))
        self.numerator = numerator // common_divisor
        self.denominator = denominator // common_divisor

    @staticmethod
    def gcd(a, b):
        """求最大公约数"""
        while b:
            a, b = b, a % b
        return a

    # 四则运算
    def __add__(self, other):
        return Fraction(self.numerator * other.denominator + other.numerator * self.denominator,
                        self.denominator * other.denominator)

    def __sub__(self, other):
        return Fraction(self.numerator * other.denominator - other.numerator * self.denominator,
                        self.denominator * other.denominator)

    def __mul__(self, other):
        return Fraction(self.numerator * other.numerator,
                        self.denominator * other.denominator)

    def __truediv__(self, other):
        if other.numerator == 0:
            raise ZeroDivisionError("分数除以0")
        return Fraction(self.numerator * other.denominator,
                        self.denominator * other.numerator)

    # 比较运算
    def __eq__(self, other):
        return self.numerator == other.numerator and self.denominator == other.denominator

    def __ge__(self, other):
        return self.numerator * other.denominator >= other.numerator * self.denominator

    # 带分数输出
    def to_string(self):
        num, den = self.numerator, self.denominator
        if num == 0:
            return "0"
        sign = "-" if num < 0 else ""
        num = abs(num)
        integer_part = num // den
        remainder = num % den

        if remainder == 0:
            return f"{sign}{integer_part}"
        elif integer_part == 0:
            return f"{sign}{remainder}/{den}"
        else:
            return f"{sign}{integer_part}'{remainder}/{den}"

    @classmethod
    def from_string(cls, s):
        """解析字符串为 Fraction，支持带分数、真分数和整数"""
        s = s.strip()
        sign = -1 if s.startswith('-') else 1
        if s.startswith(('+', '-')):
            s = s[1:]

        if "'" in s:
            integer_part, frac_part = s.split("'")
            numerator, denominator = frac_part.split("/")
            total_num = int(integer_part) * int(denominator) + int(numerator)
            return cls(sign * total_num, int(denominator))
        elif "/" in s:
            numerator, denominator = s.split("/")
            return cls(sign * int(numerator), int(denominator))
        else:
            return cls(sign * int(s), 1)

    def is_positive(self):
        return self.numerator >= 0



def generate_number(range_limit):
    """生成指定范围内的随机数（自然数或真分数）"""
    is_fraction = random.choice([True, False])

    if not is_fraction:
        # 生成自然数
        return Fraction(random.randint(0, range_limit - 1), 1)
    else:
        # 生成真分数
        denominator = random.randint(2, range_limit - 1)
        numerator = random.randint(1, denominator - 1)

        # 50%概率生成带分数
        if random.choice([True, False]):
            integer_part = random.randint(1, range_limit - 1)
            numerator += integer_part * denominator

        return Fraction(numerator, denominator)


def generate_expression(range_limit, op_count=0, max_ops=3):
    """生成表达式及其值"""
    # 有50%概率生成一个简单数字（终结符）
    if op_count >= max_ops or random.random() < 0.5:
        num = generate_number(range_limit)
        return str(num.to_string()), num

    # 否则生成一个包含运算符的表达式
    op_count += 1
    op = random.choice(['+', '-', '×', '÷'])

    # 递归生成左右两个表达式
    left_expr, left_val = generate_expression(range_limit, op_count, max_ops)
    right_expr, right_val = generate_expression(range_limit, op_count, max_ops)

    # 根据运算符检查是否满足条件
    if op == '-':
        if not (left_val >= right_val and (left_val - right_val).is_positive()):
            return generate_expression(range_limit, op_count - 1, max_ops)
        result_val = left_val - right_val
    elif op == '÷':
        # 确保除数不为零且结果为真分数
        if right_val.numerator == 0:
            return generate_expression(range_limit, op_count - 1, max_ops)

        result_val = left_val / right_val
        # 检查结果是否为有效分数（分母在范围内）
        if result_val.denominator >= range_limit:
            return generate_expression(range_limit, op_count - 1, max_ops)
    elif op == '+':
        result_val = left_val + right_val
    else:  # ×
        result_val = left_val * right_val

    # 随机决定是否添加括号
    if random.random() < 0.3:
        expr = f"({left_expr}) {op} ({right_expr})"
    else:
        expr = f"{left_expr} {op} {right_expr}"

    return expr, result_val


def normalize_expression(expr):
    """标准化表达式，用于判断题目是否重复"""
    # 移除所有括号
    expr = re.sub(r'[()]', '', expr)

    # 处理加法和乘法的交换性
    tokens = expr.split()
    normalized = []
    i = 0

    while i < len(tokens):
        if tokens[i] in ('+', '×'):
            # 交换律：对操作数排序
            left = normalized.pop()
            right = tokens[i + 1]
            normalized.append(" ".join(sorted([left, right])) + " " + tokens[i])
            i += 2
        else:
            normalized.append(tokens[i])
            i += 1

    return " ".join(normalized)


def generate_exercises(num, range_limit):
    """生成指定数量的练习题（带序号）"""
    exercises = []
    answers = []
    seen = set()

    while len(exercises) < num:
        # 生成运算符数量为1到3的表达式
        op_count = random.randint(1, 3)
        expr, result = generate_expression(range_limit, 0, op_count)

        # 检查是否重复
        normalized = normalize_expression(expr)
        if normalized in seen:
            continue
        seen.add(normalized)

        # 添加序号（当前题目数量+1，因为从1开始）
        exercises.append(f"{len(exercises) + 1}. {expr} =")
        answers.append(f"{len(answers) + 1}. {result.to_string()}")

    return exercises, answers

import re
import fractions

import re
import fractions

# 替换 myapp.py 中原来的 parse_expression 为下面实现

def parse_expression(expr):
    expr = expr.replace('×', '*').replace('÷', '/')
    token_re = re.compile(r"\d+'\d+/\d+|\d+/\d+|\d+|[()+\-*/]")
    tokens = token_re.findall(expr)
    if not tokens:
        return None

    def token_to_frac(tok):
        return Fraction.from_string(tok)

    # Shunting-Yard 中缀 -> 后缀
    prec = {'+': 1, '-': 1, '*': 2, '/': 2}
    output_queue = []
    op_stack = []

    for tok in tokens:
        if re.fullmatch(r"\d+'\d+/\d+|\d+/\d+|\d+", tok):
            output_queue.append(('num', tok))
        elif tok in prec:
            while op_stack and op_stack[-1] in prec and prec[op_stack[-1]] >= prec[tok]:
                output_queue.append(('op', op_stack.pop()))
            op_stack.append(tok)
        elif tok == '(':
            op_stack.append(tok)
        elif tok == ')':
            while op_stack and op_stack[-1] != '(':
                output_queue.append(('op', op_stack.pop()))
            if not op_stack:
                return None
            op_stack.pop()

    while op_stack:
        if op_stack[-1] in ('(', ')'):
            return None
        output_queue.append(('op', op_stack.pop()))

    # RPN 求值
    val_stack = []
    try:
        for typ, val in output_queue:
            if typ == 'num':
                val_stack.append(token_to_frac(val))
            else:
                b = val_stack.pop()
                a = val_stack.pop()
                if val == '+':
                    val_stack.append(a + b)
                elif val == '-':
                    val_stack.append(a - b)
                elif val == '*':
                    val_stack.append(a * b)
                elif val == '/':
                    val_stack.append(a / b)
        if len(val_stack) != 1:
            return None
        return val_stack[0]
    except:
        return None



def grade_exercises(exercise_file, answer_file):
    """批改练习并生成成绩报告（适配带序号的题目和答案）"""
    def parse_answer_to_fraction(s):
        """支持多种答案格式的解析"""
        s = s.strip()
        # 格式：5'2/3
        if "'" in s:
            integer_part, frac_part = s.split("'")
            num, den = frac_part.split("/")
            total = int(integer_part) * int(den) + int(num)
            return fractions.Fraction(total, int(den))
        # 格式：5 2/3（空格）
        elif " " in s and "/" in s:
            parts = s.split()
            integer_part = int(parts[0])
            num, den = parts[1].split("/")
            total = integer_part * int(den) + int(num)
            return fractions.Fraction(total, int(den))
        # 格式：a/b
        elif "/" in s:
            a, b = s.split("/")
            return fractions.Fraction(int(a), int(b))
        else:
            # 尝试浮点数
            return fractions.Fraction(str(float(s)))

    try:
        with open(exercise_file, 'r', encoding='utf-8') as f:
            exercises = []
            for line in f:
                stripped = line.strip()
                if stripped:
                    expr = re.sub(r'^\d+\. ', '', stripped).replace(' =', '')
                    exercises.append(expr)

        with open(answer_file, 'r', encoding='utf-8') as f:
            answers = []
            for line in f:
                stripped = line.strip()
                if stripped:
                    ans = re.sub(r'^\d+\. ', '', stripped)
                    answers.append(ans)

        if len(exercises) != len(answers):
            raise ValueError("题目数量与答案数量不匹配")

        correct = []
        wrong = []

        for i, (exercise, answer) in enumerate(zip(exercises, answers), 1):
            computed_answer = parse_expression(exercise)
            try:
                expected_answer = parse_answer_to_fraction(answer)
                computed_frac = fractions.Fraction(
                    computed_answer.numerator, computed_answer.denominator
                )
            except:
                wrong.append(i)
                continue

            if computed_frac == expected_answer:
                correct.append(i)
            else:
                wrong.append(i)

        return correct, wrong

    except Exception as e:
        print(f"批改时出错: {str(e)}")
        return [], []


def main():
    """主函数，处理命令行参数并执行相应操作"""
    args = sys.argv[1:]

    # 处理生成题目模式
    if len(args) >= 2 and args[0] == '-n' and args[2] == '-r':
        try:
            num = int(args[1])
            range_limit = int(args[3])

            if num <= 0 or range_limit <= 0:
                print("错误：题目数量和范围必须为正整数")
                return

            print(f"正在生成{num}道题目，数值范围为{range_limit}...")
            exercises, answers = generate_exercises(num, range_limit)

            # 保存题目
            with open('Exercises.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(exercises))

            # 保存答案
            with open('Answers.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(answers))

            print("题目和答案已分别保存到Exercises.txt和Answers.txt")

        except ValueError:
            print("错误：参数必须为整数")
            return

    # 处理批改模式
    elif len(args) == 4 and args[0] == '-e' and args[2] == '-a':
        exercise_file = args[1]
        answer_file = args[3]

        print(f"正在批改 {exercise_file} 和 {answer_file}...")
        correct, wrong = grade_exercises(exercise_file, answer_file)

        # 保存成绩
        with open('Grade.txt', 'w', encoding='utf-8') as f:
            correct_str = f"Correct: {len(correct)} ({', '.join(map(str, correct))})" if correct else f"Correct: 0"
            wrong_str = f"Wrong: {len(wrong)} ({', '.join(map(str, wrong))})" if wrong else f"Wrong: 0"
            f.write(f"{correct_str}\n{wrong_str}")

        print(f"批改完成，结果已保存到Grade.txt")
        print(correct_str)
        print(wrong_str)

    else:
        # 显示帮助信息
        print("小学四则运算题目生成程序")
        print("用法:")
        print("  生成题目:")
        print("    python myapp.py -n <题目数量> -r <数值范围>")
        print("  批改题目:")
        print("    python myapp.py -e <题目文件> -a <答案文件>")
        print("示例:")
        print("  python myapp.py -n 10 -r 10")
        print("  python myapp.py -e Exercises.txt -a Answers.txt")


if __name__ == "__main__":
    main()