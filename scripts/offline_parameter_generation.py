from nssmpc.application.neural_network.layers.activation import GeLUKey
from nssmpc.primitives.secret_sharing.function import VSigmaKey, GrottoDICFKey, DICFKey, SigmaDICFKey
from nssmpc.protocols.honest_majority_3pc.msb_with_os import MACKey
from nssmpc.protocols.honest_majority_3pc.multiplication import RssMulTriples
from nssmpc.protocols.honest_majority_3pc.oblivious_select_dpf import VOSKey
from nssmpc.protocols.honest_majority_3pc.truncation import RssTruncAuxParams
from nssmpc.protocols.semi_honest_2pc.b2a import B2AKey
from nssmpc.protocols.semi_honest_2pc.comparison import BooleanTriples
from nssmpc.protocols.semi_honest_2pc.division import DivKey
from nssmpc.protocols.semi_honest_2pc.multiplication import AssMulTriples
from nssmpc.protocols.semi_honest_2pc.reciprocal_sqrt import ReciprocalSqrtKey
from nssmpc.protocols.semi_honest_2pc.tanh import TanhKey
from nssmpc.protocols.semi_honest_2pc.truncation import Wrap

gen_num = 200

AssMulTriples.gen_and_save(gen_num)
BooleanTriples.gen_and_save(gen_num)
Wrap.gen_and_save(gen_num)
DICFKey.gen_and_save(gen_num)
GrottoDICFKey.gen_and_save(gen_num)
SigmaDICFKey.gen_and_save(gen_num)
ReciprocalSqrtKey.gen_and_save(gen_num)
DivKey.gen_and_save(gen_num)
GeLUKey.gen_and_save(gen_num)
B2AKey.gen_and_save(gen_num)
TanhKey.gen_and_save(gen_num)

MACKey.gen_and_save(gen_num)
RssMulTriples.gen_and_save(gen_num)
RssTruncAuxParams.gen_and_save(gen_num)

VOSKey.gen_and_save(gen_num, tag='0')
VOSKey.gen_and_save(gen_num, tag='1')
VOSKey.gen_and_save(gen_num, tag='2')

VSigmaKey.gen_and_save(gen_num, tag='0')
VSigmaKey.gen_and_save(gen_num, tag='1')
VSigmaKey.gen_and_save(gen_num, tag='2')

B2AKey.gen_and_save(gen_num, tag='0')
B2AKey.gen_and_save(gen_num, tag='1')
B2AKey.gen_and_save(gen_num, tag='2')
