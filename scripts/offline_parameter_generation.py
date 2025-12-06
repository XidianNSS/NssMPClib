from NssMPC.application.neural_network.layers.activation import GeLUKey
from NssMPC.primitives.secret_sharing.function import VSigmaKey, GrottoDICFKey, DICFKey, SigmaDICFKey
from NssMPC.protocols.honest_majority_3pc.msb_with_os import MACKey
from NssMPC.protocols.honest_majority_3pc.multiplication import RssMulTriples
from NssMPC.protocols.honest_majority_3pc.oblivious_select_dpf import VOSKey
from NssMPC.protocols.semi_honest_3pc.truncate import RssTruncAuxParams
from NssMPC.protocols.semi_honest_2pc import Wrap, ReciprocalSqrtKey
from NssMPC.protocols.semi_honest_2pc.b2a import B2AKey
from NssMPC.protocols.semi_honest_2pc.comparison import BooleanTriples
from NssMPC.protocols.semi_honest_2pc.division import DivKey
from NssMPC.protocols.semi_honest_2pc.multiplication import AssMulTriples
from NssMPC.protocols.semi_honest_2pc.tanh import TanhKey

gen_num = 100

AssMulTriples.gen_and_save(gen_num, num_of_party=2, type_of_generation='TTP')
BooleanTriples.gen_and_save(gen_num, num_of_party=2, type_of_generation='TTP')
Wrap.gen_and_save(gen_num)
GrottoDICFKey.gen_and_save(gen_num)
RssMulTriples.gen_and_save(gen_num)
DICFKey.gen_and_save(gen_num)
SigmaDICFKey.gen_and_save(gen_num)
ReciprocalSqrtKey.gen_and_save(gen_num)
DivKey.gen_and_save(gen_num)
GeLUKey.gen_and_save(gen_num)
RssTruncAuxParams.gen_and_save(gen_num)
B2AKey.gen_and_save(gen_num)
TanhKey.gen_and_save(gen_num)
MACKey.gen_and_save(gen_num)

VOSKey.gen_and_save(gen_num, 'VOSKey_0')
VOSKey.gen_and_save(gen_num, 'VOSKey_1')
VOSKey.gen_and_save(gen_num, 'VOSKey_2')

VSigmaKey.gen_and_save(gen_num, 'VSigmaKey_0')
VSigmaKey.gen_and_save(gen_num, 'VSigmaKey_1')
VSigmaKey.gen_and_save(gen_num, 'VSigmaKey_2')

B2AKey.gen_and_save(gen_num, 'B2AKey_0')
B2AKey.gen_and_save(gen_num, 'B2AKey_1')
B2AKey.gen_and_save(gen_num, 'B2AKey_2')
