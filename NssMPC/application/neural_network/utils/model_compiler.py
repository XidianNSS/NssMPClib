"""
The version of regular expression
"""
#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import re

name = '_SecLayers'
layers = "Conv2d|MaxPool2d|ReLU|Linear|AdaptiveAvgPool2d|AvgPool2d|BatchNorm2d|Embedding|Dropout|Softmax|Tanh|GELU|LayerNorm"


def sec_format(filepath):
    """
    Format and process the code in the given file path.

    After opening the file in read-only mode:
        1. ``code = re.sub(r'(?<!["\'])#.*', '', code)`` Use the regular expression ``re.sub`` to remove single-line comments starting with **#**.

        .. note::
            **(? <! ["\'])** Make sure that **#** is not preceded by single or double quotes to avoid deleting the comment part of the string.

        2. ``code = re.sub(r'(\'\'\'(.*?)\'\'\'|\"\"\"(.*?)\"\"\")', '', code, flags=re.DOTALL)`` Use regular expressions to remove multi-line comments.

        .. note::
            ``flags=re.DOTALL`` enables. To match newlines and thus capture multiple lines of comments.

        3. ``code = re.sub(r'\s*$', '', code, flags=re.M)`` Delete all empty lines using regular expressions.

        .. note::
            ``flags=re.M`` makes **^** and **$** match the beginning and end of each line

        4. ``code = _layer_sec(code)`` Call the :func:`_layer_sec` function to perform specific processing on the model layer

        5. ``code = re.sub(r'(.*?=.*?\.)(view)(\(.*?\))', r'\1reshape\3', code)`` Replace all calls to ``.view()`` with ``.reshape()``, which is to change the shape of the tensor

    :param filepath: The path of the file to be processed
    :type filepath: str
    :return: Processed code string
    :rtype: str
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
    """
    The purpose of the function is to process a given code string.

    * Find the Sequential layer:
        After extracting each Sequential block, extract the full Sequential paragraph, replace the layer name in the segment with a regular expression, Replace **(.Layername)** with **(.sec.Layername)**, leaving the preceding space.

    * Replace layers in assignment and return statements:
        Replace the layer name in any assignment or return statement with the **(.Sec)** form

    * Replace relu call:
        Replace all relu calls to non-self objects with the SecReLU() method

    * Add import statement:
        At the beginning of the code to add a line of import statements to ``NssMPC.application.neural_network.layers``. The layers module import, and use the name as the alias.

    :param code: The string to be processed
    :type code: str
    :return: Processed code string
    :rtype: str
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
    code = f"import NssMPC.application.neural_network.layers as {name}\n" + code
    return code
