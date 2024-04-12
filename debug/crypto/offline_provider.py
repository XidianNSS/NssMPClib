from crypto.primitives.beaver.beaver_triples import BeaverTriples
from crypto.primitives.beaver.msb_triples import MSBTriples
from crypto.protocols.function_secret_sharing import *
from crypto.protocols.arithmetic_secret_sharing.truncate import Wrap

BeaverTriples.gen_and_save(100000, 2, 'TTP')
MSBTriples.gen_and_save(100000, 2, 'TTP')
Wrap.gen_and_save(100000)
PPQCompareKey.gen_and_save(100000)
DICFKey.gen_and_save(100000)
SigmaCompareKey.gen_and_save(100000)
