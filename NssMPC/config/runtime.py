class MacBuffer:
    def __init__(self):
        self.x = []
        self.mac = []
        self.key = []

    def add(self, x, mac, key):
        self.x.append(x.clone())
        self.mac.append(mac.clone())
        self.key.append(key.clone())

    def check(self):
        from NssMPC import ReplicatedSecretSharing
        x = ReplicatedSecretSharing.cat(self.x)
        mac = ReplicatedSecretSharing.cat(self.mac)
        key = ReplicatedSecretSharing.cat(self.key)
        from NssMPC.crypto.protocols.replicated_secret_sharing.honest_majority_functional import mac_check
        mac_check(x, mac, key)
        self.__init__()


class Register(object):

    def __init__(self, name):
        self.module_dict = dict()
        self.name = name
        self.ignored_modules = set()

    def ignore(self):
        def _ignore(target):
            self.ignored_modules.add(target.__name__)
            return target

        return _ignore

    def register(self, target):
        self.module_dict[target.__name__] = target
        return target

    def modules(self):
        # 返回未被忽略的模块
        return {name: module for name, module in self.module_dict.items() if name not in self.ignored_modules}


MAC_BUFFER = MacBuffer()
ParameterRegistry = Register('ParameterRegistry')
