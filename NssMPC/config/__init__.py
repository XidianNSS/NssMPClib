#  This file is part of the NssMPClib project.
#  Copyright (c) 2024 XDU NSS lab,
#  Licensed under the MIT license. See LICENSE in the project root for license information.

import os
from importlib import resources

__data_path = os.path.expanduser('~/.NssMPClib/')
if not os.path.exists(__data_path):
    os.makedirs(__data_path)

if not os.path.exists(__data_path + '/config.json'):
    __resource_path = resources.files('NssMPC') / 'default_config.json'
    try:
        with __resource_path.open('rb') as __source_file:
            __data = __source_file.read()
            with open(__data_path + '/config.json', 'wb') as __target_file:
                __target_file.write(__data)
        print("未找到 config.json  已重新创建.")
    except FileNotFoundError:
        print("未找到 default_config.json.  检查 NssMPC 包是否完整.")

from NssMPC.config.configs import *
