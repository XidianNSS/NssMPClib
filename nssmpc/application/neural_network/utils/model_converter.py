"""
The version of regular expression
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import re

name = '_SecLayers'
layers = "Conv2d|MaxPool2d|ReLU|Linear|AdaptiveAvgPool2d|AvgPool2d|BatchNorm2d|Embedding|Dropout|Softmax|Tanh|GELU|LayerNorm"


def _sec_format(filepath):
    """Format and process the code in the given file path.

    This function reads a python file, removes comments (single-line and multi-line),
    removes empty lines, processes specific layer definitions using `_layer_sec`,
    and replaces `.view()` calls with `.reshape()`.

    Args:
        filepath (str): The path of the file to be processed.

    Returns:
        str: Processed code string.

    Examples:
        >>> code = _sec_format('path/to/model.py')
    """
    model_file = open(filepath, "r", encoding="UTF-8")
    code = model_file.read()
    code = re.sub(r'(?<!["\'])#.*', '', code)  # delete comments
    code = re.sub(r'(\'\'\'(.*?)\'\'\'|\"\"\"(.*?)\"\"\")', '', code, flags=re.DOTALL)  # multiple comments
    code = re.sub(r'\s*$', '', code, flags=re.M)  # delete blank line
    code = _layer_sec(code)
    code = re.sub(r'(.*?=.*?\.)(view)(\(.*?\))', r'\1reshape\3', code)  # transform view to reshape
    return code


def _layer_sec(code):
    """Process a given code string to replace standard layers with secure layers.

    This function performs several transformations:
    1. Replaces layers inside `Sequential` blocks with their secure counterparts (e.g., `Layer` -> `SecLayer`).
    2. Replaces layers in assignment and return statements.
    3. Replaces `relu` calls with `SecReLU`.
    4. Adds an import statement for `nssmpc.application.neural_network.layers`.

    Args:
        code (str): The string to be processed.

    Returns:
        str: Processed code string.

    Examples:
        >>> new_code = _layer_sec(original_code_string)
    """
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
    code = re.sub(r'(?!self\.)(.*?)(=|return)(.*?\.?relu)(\(.*?\))', rf'\1\2 {name}.SecReLU()\4', code)  # F.relu
    code = f"import nssmpc.application.neural_network.layers as {name}\n" + code
    return code
