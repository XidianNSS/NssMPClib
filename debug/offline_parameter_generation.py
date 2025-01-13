from NssMPC.crypto.aux_parameter import *
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.vsigma_key import VSigmaKey
from NssMPC.crypto.aux_parameter.select_keys.vos_key import VOSKey
from NssMPC.crypto.aux_parameter.truncation_keys.rss_trunc_aux_param import RssTruncAuxParams

gen_num = 100

AssMulTriples.gen_and_save(gen_num, saved_name='2PCBeaver', num_of_party=2, type_of_generation='TTP')
AssMulTriples.gen_and_save(gen_num, saved_name='3PCBeaver', num_of_party=3, type_of_generation='TTP')
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
