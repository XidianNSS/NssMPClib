"""The version of regular expression
"""
import re

name = '_SecLayers'
layers = "Conv2d|MaxPool2d|ReLU|Linear|AdaptiveAvgPool2d|AvgPool2d|BatchNorm2d"


def sec_format(filepath):
    model_file = open(filepath, "r", encoding="UTF-8")
    code = model_file.read()
    code = re.sub(r'(?<!["\'])#.*', '', code)  # delete comments
    code = re.sub(r'(\'\'\'(.*?)\'\'\'|\"\"\"(.*?)\"\"\")', '', code, flags=re.DOTALL)  # multiple comments
    code = re.sub(r'\s*$', '', code, flags=re.M)  # delete blank line
    code = _layer_sec(code)
    code = re.sub(r'(.*?=.*?\.)(view)(\(.*?\))', r'\1reshape\3', code)  # transform view to reshape
    return code


def _layer_sec(code):
    replacements = []
    for item in re.finditer(r'.*Sequential\(', code):
        start = item.span()[0]
        count = 1
        i = item.span()[1]
        while i < len(code) and count:
            if code[i] == '(':
                count += 1
            elif code[i] == ')':
                count -= 1
            i += 1
        segment = code[start:i]
        new_segment = re.sub(rf'(\s*).*\.({layers})', rf'\1{name}.Sec\2', segment)
        replacements.append((start, i, new_segment))
    for start, end, replacement in reversed(replacements):
        code = code[:start] + replacement + code[end:]

    code = re.sub(r'(\s*.*)(=|return)(.*?\.?)(' + layers + r'\(.*\))', rf'\1\2 {name}.Sec\4', code)
    code = re.sub(r'(?!self\.)(.*?)(=|return)(.*?\.?relu)(\(.*?\))', rf'\1\2 {name}._SecReLU\4', code)  # F.relu
    code = f"from application.neural_network.layers import sec_layers as {name}\n" + code
    return code
