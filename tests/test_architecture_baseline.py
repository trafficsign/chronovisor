from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).parents[1]
BASELINE = ROOT / "docs" / "refactoring" / "architecture-baseline.json"
P2_RETIRED_PRIVATE_EXCEPTION_ID = (
    "arch:97b784f974ebdf78ae0226731f9b421e381ac04bad98dd104435367c618f52e9"
)
P2_RETIRED_SITE_IDS = (
    "arch:0d799ce1e29887c64caf095e2640a148aa9e90326aaef4c45a476ae13c33e85b",
    "arch:7d200838738a191b653ce5829735d820b786746b2a6329bf9cc98610ff57dd32",
    "arch:a36a5c65819a0aab93f8c971dee29537ef65da12069950b9fdab4b270bb9c9d7",
    "arch:c7056bd3c53d85c0dfb27edae2aead5bac51065e169b7b87ef3f21b4363d3ca2",
    "arch:e821d535f1969c514f6f49a2338ec36e382d3a6b82040c593bf28f580d86c97d",
)
P3_RETIRED_CLASSIFICATION_LAB_EDGE_ID = (
    "arch:0f37f016df9c2c328a0edc59fd3b3c4b8039921bde3fdaaede27d59f34be9f60"
)
P3_RETIRED_PROVIDER_SITE_ID = (
    "arch:7923c8117584e014f1a6d93283fb7ce9eb012f2cdb7e6228171c5fffb58aecc1"
)
P4_RETIRED_SEARCH_LAB_PRIVATE_EXCEPTION_ID = (
    "arch:93b9e7819071ee5c3aecca9db43262f5ef56ac9a8b6085178911353e0c0d25b7"
)
P4_RETIRED_SEARCH_LAB_EDGE_ID = (
    "arch:95ae240af520b7872ef42f68074e124a12bc04c5f937d508f8135a6a8a23d03a"
)
P4_RETIRED_SEARCH_LAB_SITE_ID = (
    "arch:d4b968e37807dc285676f8f73e74c3740782f10133aa0fc2e09acf42dfab3636"
)
P4C_RETIRED_EXCEPTION_IDS = (
    "arch:53527ec1ad690243e3d5a03be50728716d34df8f1c222a10e652b40e872e01c0",
    "arch:61a565cf4549c1807a69940ad985ef4c66cce6a47176773d4d10add426b238fd",
    "arch:89e88a117c7d284e8325a0d39a301530316177b1e0054b3eaf0108d82f727f2d",
)
P4C_RETIRED_DECISION_LAB_SITE_IDS = (
    "arch:1bfb36c1612203a1ed27ea9feb74fe67449555be6e6f1663624f13584342514c",
    "arch:260382e328ebb07af32a060e01f277e15c3903d29e58307404e2f21c8e703b0c",
    "arch:67561ef5193c9f26cc02f98c0b2b59ffb78f6d722fb1972227a0ff1778195d7d",
    "arch:6b6b9de176ed950d9f3bc08bd8157fff4affee7a28dd0c2bb58981ee4d7a63e5",
    "arch:c4f90d606485dac03d16fa59b11938455864828332e2ee5f13418a0a7a0c54b3",
    "arch:c70da36211b3b4a165f59ba6281e2a47bc9bcbd134eaa5b155dd29de5b1f5db1",
    "arch:c8f86543c1dee07e9e0bfe5594699726c3f688b1405bcb3570ec8670e37ebe4a",
    "arch:cdf4e056944483b580134c81faa876f2d16dc7d92545a4f5b24ed6bca6a52431",
    "arch:fd60ca9f13016ac01869596b80083d7f7cf3c21ffeff491d104c60643b703003",
)
P4C_RETIRED_MOVED_SITE_IDS = (
    "arch:0e2f4f55e57ae72b55a9ac06a4657a51c188922dd8e2e714e98024f49138251c",
    "arch:1379aba196c611f97d7178a0364f4486f243e9da8b8840e141cb6faae1b6b3ab",
    "arch:1ac58df524d4b41f324f4dff7dd17ddf5fad7635b4213d633cb030ccebabe29e",
    "arch:1b6f386c9960dbb4ea63b3d1ae5284ec9d721aaf22b74790a86197d00f13a090",
    "arch:1dd8a86eae5bc73ea226a6e5a70fabbe49d987ccc6bd203029c84910d2b371c0",
    "arch:3b17a90248ef53b2292ff341faac23a1e5e788bb3ccf457b5214abedbd057892",
    "arch:54d33e971f9381c55ef8197914dac30c05485f7270ab62c9cc8ae368534519cd",
    "arch:59c991a217d25c0c8d89ecb4919405927c090fd27377ce37a12cce437d21aaaf",
    "arch:7225cb98e0b4893dc35179535642999cd467c97f6d18a02e531ce34ae6a765bc",
    "arch:7e6db621af61ead96ab24fde3184087f82cb86d9e80ed4c1a7155ccaf29c2fbc",
    "arch:8734a683868e5bb81a039b9e9332751da5a6e70e7d9161c4574a9c8d86d70a3d",
    "arch:cb3f2a5f7e521eafb107c37f60252650bf07a67300160a2eb4336647adcb87ce",
)
P4A_RETIRED_OPS_LAB_EDGE_ID = (
    "arch:eb2b813a6f4420f60df925a80375c90dea7fc85dcc15240272b743e739b509a7"
)
P4A_RETIRED_OPS_LAB_STATIC_SITE_ID = (
    "arch:7dc55845add1f43bc44723139de24db48ad92ca39be88c436b8d3c6bc5f960b0"
)
P4A_RETIRED_OPS_LAB_DYNAMIC_SITE_ID = (
    "arch:f2679d78c5af6d67a6f7c00c1740b8328bf32cd33253d61369db71f29ee0b8e9"
)
P4A_RETIRED_MOVED_SITE_IDS = (
    "arch:054633d2f397258b2ae27fe2e8771ff11a3b07d2ba2b3237d552518e463a73f7",
    "arch:37ff531c153962a561682fc13ead43a4142dc0d1258701d3feb477586f5b2522",
    "arch:57a4f30b14d566feba89b017d42d0932e701d035baa2e52a81828cd72db17fd1",
    "arch:6f981f4b5cc6d6f3f7d50c5e65d194bf5dcbee6820a647f47030f43f52f76016",
    "arch:dd7c91d6e2347693b73918a9e01e96c08d74031108131b0c44655012ba1a023e",
    "arch:e3e8f26b41ac88d106b2d9a357f71293eba715a14a27cebd717278fa49d66f95",
    "arch:e767a86ee1c2c1ba318ad5fec467355b673266eedea548500ecf197b340eebbc",
)
P5_RETIRED_EXCEPTION_IDS = (
    "arch:171347b3036f3dba27546589a302e02f6d22314fffe9c50e95e25a59b49c7a94",
    "arch:283761a1ce09a52dd6f4edb6f512e0c409fbbaea514131f7a6243d7d8dc53176",
    "arch:396f651d94c790b548203581fc4ed5e9b0bbea9045bab821818b23fea411d4af",
    "arch:e92c8cafb5290b6af5d0a1d279113712d9cc61b81ca067882b50cdcb835a4d7a",
)
P5_RETIRED_SITE_IDS = (
    "arch:01cb4b6b71c55182506795a04ff2f7f6afdbb25b50c73bcdfedcf59311366214",
    "arch:0460fd594ff4b716cbfada7cb5fc442f70073aa662555968c21b7d80f6264ba9",
    "arch:0a7f8dbd9c0820140f0314ba614fde315692fcacd879df9a782af6da2066063d",
    "arch:18631bd52be071b4118f39d7842761c91f59aeee003f788c601d29926d190689",
    "arch:1a32a5682df507ee4cb4533c587f999d508fb5e6616147655a2e05f8e68d7243",
    "arch:24b8a1df86943033712a99f5ebe2ecc3f78001197b7398540bb18c235916614c",
    "arch:24c748ee21fddaa489f0c5d4e5a4c19909f0f0d379a0704b1852471be41187bb",
    "arch:31b34e33d357942860b85fbb2a04099c031c70cf8f7af5700765a18db75516c0",
    "arch:322e6015fd54a49009f3e862509620cd8b819404c1216610750977346901c046",
    "arch:33f58c9af87fda9ba5d418afe4cc74e2216a4022f2dbeb37b52b83ef0097dbf6",
    "arch:3ad1047710ff2624c4db3e26a60198fa91c20fb6bb2bc4781a3b851b89b72e7a",
    "arch:5e9a02e874e2ffe42d46e9d77caca146ff943cb1a9af034efab8b15f64a55c14",
    "arch:639b1bfedebd222a9dd168ac33999f9058fdba166bd54d0208d5b89dfaf55bf3",
    "arch:6f859bfb682d8333a37483c3e76fabb85d5310cf647810e4a7969fd78bf01bea",
    "arch:720a675ffaa72d9b18758180feff7385941e447ceb8e426b6ec26666a01593c1",
    "arch:763463291a58c9fa862528ee05538df38e1cce5820334058ecb2f175fd3f5995",
    "arch:785f95fa09621d520ac9e4647d1eeb482c6bfe90e87a9d00a6b65ca0637f7511",
    "arch:7a7c6d99100b9a3a6014fb5099ffa600f383750baf72f2774418f9994a5b6aea",
    "arch:8411b73350b2f822018630fd1f5215431d91d141e51ffc9ef86e6e52843bece6",
    "arch:8bc1cfd637090afb4291cdd75d002ec8440b96162e852a7a44ab3403ed77c8e5",
    "arch:941cb43f84962cb177f6e8195dc1f14941287500af76c811d2edd678ae05cdd3",
    "arch:944125e390d8dc58c4ffd8f780c07e9b595d0da5cc7e606dbbddf0c0cc88959b",
    "arch:9646b4ffcfcafefa72f16f86d3457d15fde91dc3840ac48cb79060f3ed64507e",
    "arch:973d50db45fcfdffdecc9d4dc326680f63211d5c7f3aa3b46090a26f976ac277",
    "arch:9c0bf8955ee18b265cfc68983104d38c95ff2ff7dcdbf177683cc54c1b5ce716",
    "arch:a34416aa9a7011bf78a5563a1cc091a49e7ca0cc846701fa5c8b5ce2fdf80b61",
    "arch:a366743216a7ce9a3e54c84b3695671afe072f71aeb4c36ed2900e2a87ea95cf",
    "arch:a8a8c2c4a683c87a7d170ea336825ced1bed2c71ed6e92c11da69d3641f7a161",
    "arch:aa807a3c573b0f052ce29cd7222cc1c10d4ca24671f9460c1354d334a627e88e",
    "arch:ad404ebda399c485621c4d0d9647a0bc71a623fe7d1f676aa3d028e3f44a9591",
    "arch:b016c5b1f651885c447f3f1e9e4d0a9e4164f3a6b9164c828936baa9b0eef80f",
    "arch:b68228dd95edfb7820cb517437be0b54f4a8b43d5bccec5a6676bd59d04fa227",
    "arch:baed967eb71e9fbeb06707096af5bc811dac59a32b31189f2a582688df5ab6c6",
    "arch:bc4b9a3ec3d27c54ac23b9d217edf9f75b9c0a1e8a550cb304dd850279464997",
    "arch:bce8001163b7aa8ebd5d95b8db6e1bd782dc153c6deac100e7d73fe61597b88c",
    "arch:c3fe7109db9e6487f7638f1c1c42b5d560b093bd9a63a5e63e4c227d74e03946",
    "arch:cba8a669f8dc249ccd8d3b37f9e1486c463de27b20f280442a1528df2846fc06",
    "arch:d0916a604d0adb6a7ce17b3b222d93e00ac34486242cffff8ea8256d58dd932b",
    "arch:d3197fe29c5799aec77cc2ed8c30953a48096e5c68556beb2ed600899a313c2a",
    "arch:ec49510cceb5e7f9d3a76a6c7069a2b41d19e0b11256fc37b81804a01aa5ba9b",
    "arch:eeb0719681660954298c1a66b356a3e296db86c868fc5b36d2bc3c243753fc37",
    "arch:eecf48345365f12f9c10b90244e2e78fa3b2a25b6c6e77ffa3690ed236796114",
    "arch:efbcf5ca8b62c357f7cc1e77e9fb8828c808866ee76ba3488dae02ade165877c",
    "arch:f585ebdf6516a4beae52be95fdc8df6e4cb6e08fc9e5bc48507fa953cda89745",
    "arch:f7f46f11a378529f43d59ec9abb75f4664879cd942538b1bd237be9a7a2a2f71",
    "arch:f8bbb02b7eec6ff3cfbb0f15ebca32ede6757f230a18a86656ba26e80a006fec",
    "arch:fdd09cc9b8294d7bfd12da71eda5c5d050d6707089e5cdd2213712fd673fcc84",
    "arch:fea1f17ed67a467ea624ed1f272a10f035e9a7edb250cbc9f93784a589faf680",
)
P5_RETIRED_STATIC_SITE_IDS = (
    "arch:0460fd594ff4b716cbfada7cb5fc442f70073aa662555968c21b7d80f6264ba9",
    "arch:720a675ffaa72d9b18758180feff7385941e447ceb8e426b6ec26666a01593c1",
    "arch:d3197fe29c5799aec77cc2ed8c30953a48096e5c68556beb2ed600899a313c2a",
)
P6_RETIRED_EXCEPTION_IDS = (
    "arch:11c80628d958dcc6b21e259ee77d412a97af4cb5511f0429c84a484202a6a596",
    "arch:4e0eb5b52c98d28fdfb160175a39fc24dccd4c1cb9321fbe30aff72e46b8ae90",
    "arch:5844476903cb2fb2dcd084f7a9678a1ddeeaedf6df8dafe8cccc48b487815291",
    "arch:6a8632c30ae66afc94f18ba39d3f1e25d9dd4d6d3b28bd1fff31f5884449c1db",
    "arch:7b141ccbbb17a809019002f5575e6961b49329a7bcaa010927639a3e44030152",
    "arch:7bab544e804fbf0af4eee6323a582bd92c3c743644f85c477bc59bcd85c562b1",
    "arch:7c8d8917ab8f77feb230950beac56d55a66aa6dc44ee32539c9fb9171431eeb7",
    "arch:a04da670e05e0224a5d91fd4a02cc0d0c8d5409dfce78fa6eda301cc91fedafc",
    "arch:bf8303c4fc79c57bafc96d8458d6b39afe5092356234bd623353f23ca09859b4",
    "arch:da499bda6ddd28b2de38c278f9b9a3f21566f7f17ce939dd6947f0dbff701f9a",
    "arch:ee6cf8478aea309360b089c9efd2e03e1c89e3c54771126142c62bde33f45127",
    "arch:f4fd4696f256bd842e26bda1797cec07348534befffe25bcf5c15756d3a7e043",
)
P6_RETIRED_SITE_IDS = (
    "arch:14892c65db81586d6c498c3caba33ac9d0d88bd3ba494d117c31115ff7d6bdf0",
    "arch:2d0a85a686938785c8908d22a51011a3dda18aee717eaca621b72cbb3cb58761",
    "arch:345a1449cf34a4061b4babe16a9f9e456e35720159c9466eb87c1752e2405b7e",
    "arch:9d2b3b531fa72fc18238f09e1b3213bae15a505d1987d595be04a3d11a5782b6",
    "arch:a31c43ef090a778e85c3e5fb5c12d4c52b84f2aa99e88dc02f6476d456eb1e48",
    "arch:a8abccad0ea8e853f423d4d53ac899770efbd37520da28ce6a93bd3cd3d16a88",
    "arch:cd48b5ce268793ec5fd33a0eb4e2cd2ca4686b89e33aa6ef48a9b24ebe8e14aa",
    "arch:d75501787f96455c30118b5a40578f63ea5d26dd79f75676267e19551669dba9",
    "arch:e0041488f7773b3fd6f8a1e5768041c91b4afd7c222787920728c6e6ef8fd0e6",
)
P8_RETIRED_EXCEPTION_IDS = (
    "arch:0dd175ac5f39948a2ea0105bd57085bc0ad73146211af3bb2fd4145ce3e85500",
    "arch:28c3a88a93f1de01512b7f754d6750ed14917c793cf8dd1469180be7e10d1b19",
    "arch:338b445c2207aafffd487bcb16cd0a25071ef34a3860d4a7be4f472326527990",
    "arch:41f00727fc440966e09b85f6fbf82b0ddb3cebfd242ce258c8b322705d627986",
    "arch:4378d25de1fb1ddbe365e869d32cb5356c149f66771ab10c2e498266c0bfc0f3",
    "arch:590fa4fd0924687c1a913b2af7f182f92143de6fbbf0e541f64953d16a179408",
    "arch:648a9b9e052d10909f44d4b79186d0793a5de7ce64b918b3f98482561d55818c",
    "arch:6abce68a5a1e666ebff3595b7b7cb404ef0a97215372636843e1ec041630493a",
    "arch:6cc04644754e482ed6ae7e9859ad0fac7ed50e8b4835de515d0b1ad38b824d79",
    "arch:7154180a53adc001e2ce4811641f166678933f36f4d85168227508aecf661719",
    "arch:72a718cc63232593dfbd7ffc828f783ab5af79e501158637e58604e8219e82ca",
    "arch:7d4e47bd194181a9b13b16420474fbc3403a2059f771569ef26617fa14c08ef2",
    "arch:9187b1564e97702b2c73709ba1ebae7a4b04280dd12f78d0d4bed66c46c5b794",
    "arch:95bdc07367c91b215296553d55fd82c5953fbfef4629d534980c6da0c9cd222f",
    "arch:9891f454d950d68fd2895bfe17e55ca371eb284e7abf0041b91299ce393b1a1a",
    "arch:9b54312069c9a4d226d034ccd821763d8227c003cc5b31d29243e76d465123e1",
    "arch:9be0fa9078776c58c283d3dc92b97af31c7828b7cd0bb496b167981eac0f35ed",
    "arch:9c938719975bdadaa3c4946e8622bf2a4a66d26c5a34eff815b2eb958c2bf95f",
    "arch:a2ca34a4cc432e3fb966ab693529f5f1a4ef36f4de07655c0cfc29ab1e319039",
    "arch:b85bca5b566b069e65c44d23689a6fc707768712de8b789913734814b8ecf640",
    "arch:d75e577e883bebe790d233a96cf5dde988790e705c776ab677dd4f96ec3e691c",
    "arch:e1e7224a039e1e222df0c954dc473c9829c6102f7279927927c48c9e63b6f2a2",
    "arch:e9fa9bce4c0a06aa0146a84a4ba56ec0137c485109ed9064c9109699917c0988",
    "arch:fbdec81173e15632af15abe3ca6495603c199d519fcafaf1891b52e3902c4e17",
)
P8_RETIRED_SITE_IDS = (
    "arch:05a5644c1c8f4dd7536dfa0ff41043eeb47e345093018b6794bed6f5808711df",
    "arch:17f858bf2bfddb6e63415b822ff82724119dc57418147be329a08c242f155a4e",
    "arch:3eb89b4f98050387999ec5390c5bed0764626ed8a258efb28e2887dc8c9d9576",
    "arch:41a537e2dd25f011407cd49f335a1409af848d1a7dfa0c67f994a944c59a9310",
    "arch:4588bb1ce077357947e20e066c6093ae4fd37a0da12115ea4d4eb50dfcb506ca",
    "arch:4d60610f1891ba0fce296751c45342fba1e11422a1154372dd68ef9fb209cee1",
    "arch:54594cf5b153251205c0f1a64f2802f2c8434e3470238ffc4077e54c913d215c",
    "arch:5c50cd737a1946fbee84ca62d05f871f8b0501fdaaf195887d0015e1369bd47a",
    "arch:5c6f34e998db473f557450427911060f10d2104a08edcc5ce9d571df6a9e30b4",
    "arch:704b12abe8163940b25486bdad5a611795f5fefcb41181979a145d6535e23a09",
    "arch:7732b045d37b3b0908200e20b8363574942fea9697833942639ef630b1f85fc8",
    "arch:866bbdab4f7d3bc8bff91b6be7a381959a2869e0ea745089d075418637bef7b2",
    "arch:93d16b885ad11b7bb1c80e7828a3f11c4360cc4f1a7ee6d23582513ed9c020ad",
    "arch:9c72014099f660aa1a4c423d8f320d8ca3e184208e518e8251c6533d818a67bb",
    "arch:a0657ac2ea78d3f4b9f89d408926ea25114c6db8618a0493c4e0c05bb1da5f56",
    "arch:a86be5811e6fe125b2c97401a36498267f440fa9beec2590bfc1ed59e62b9cde",
    "arch:a95706f14c430049687c0c78ad294625b60d2d12e171d5823912b0fbd63eceed",
    "arch:aeedfb954b8e0397f834e7a90459f6df0f2e9b8beb128c7739f8646c84fdd0f4",
    "arch:b2e6e19cdf61d0a42cb4f53676b94eb3abcebfadd941922cbc49418b8fa352df",
    "arch:d7babd64ddbfa4c8fa1284a37e088fb938e78a1cfae8f887c57cc5a129c4ee90",
    "arch:de6ae4bbda036951b03e1ed2118299d645683c0ef9a8ca5ae58178631cae2afa",
    "arch:e025071ee401c58f20d9ec33d033bae0f9e5e63048d9a4c40988927c3c13cc2a",
    "arch:e0f682cdefa0a5826d0fd863f8567461073a1ff7d7b2a81ddf69368cf4fc9f37",
    "arch:ffe37b3f777b7be66e4a055426d26f1a7750aa4654290db9a8e7e824e94f643b",
)
Q_RETIRED_EXCEPTION_IDS = (
    "arch:09562a56e206b46c00dc6162c8a25877c5be0922d341d59a27b2760461794009",
    "arch:35d25704d99867a9d6ddc85775fa35dd9a8c0c348ff81a4c77b63ba45394d27f",
    "arch:65a25466b54a85c7fac4b98ae2385decda3a097626a4e25032f76b20bd78d123",
    "arch:722e44de84dc21d1453258284767bd62da53acdda4a5aa2714f7cc3966279cf2",
    "arch:80497e420c350de929b19520f4b5e582f7aeef43a7e4fd7fcc83ce0f2e6e2400",
    "arch:c607c67a56e1bf3727164569f5762c28562c14de6c2dd65118037798e1bc5422",
    "arch:e6c724f0b06379d55660232c4ceb93edbead8217dfcc5b93377edff9077c7948",
    "arch:f8662161b0b513d566a2c1371d78ffc02e39c4ee310c04f34598a275f0c16169",
)
Q_RETIRED_CODEX_STORE_SITE_ID = (
    "arch:078f19253b4b2590505ced5257392c0718443b64a691cccefb6e19c20f7b9b94"
)
Q_RETIRED_CLAUDE_CODE_STORE_SITE_ID = (
    "arch:8e44976bee95c4f649cf36c24aaa1f3aad8f83d70eacd937cfe6dfa7d63d35f2"
)
R_RETIRED_EXCEPTION_IDS = (
    "arch:0a0ecf52ab0c563bda61e75cce2798f692b54d14d6f3b49e7e0751fb44730beb",
    "arch:399d92dc378092ffa44dab2d26f9f424864e0ac5bfabebac84547350033450cf",
    "arch:5174f3ba292ee22c6d157040e15c0a79b16043c72f725e5512a8c31da4e8f4e7",
    "arch:60a5b5c668b37aa3ee7d70a4e44bdad889568df6d841871b2ef6749d64eefde0",
    "arch:60cfc7ccff32195d69cf1ff12e387cc4f0d078047822ff8705ff6bcbec1cb7d4",
    "arch:76453fde7eec05f39031c86510e5016812fabf38593116a4adecc42f1976a652",
    "arch:788ec9ed6692f1a4670d5aea47cd4022e0b28007a34d792b7159c49bef286088",
    "arch:a5d4ad8bde7f5dfaa9f357b83e7201f6e89095519c5c8a54071dc9e43324d859",
    "arch:a781da4d245f5995f5df1879d052252ad9b6bd5ebcd0dfc76e0194fd5ca1f1d3",
    "arch:a856d00f2db0a1f2faa9f184c07d656628c69257272735f151e06261534daeb0",
    "arch:b71370ce6a32319293a1e43fe04b9d9a15c2865165e04a23b53047fdfb5a94cd",
    "arch:c29d98a39da94b2c3cce80e59e0f4176265401935709254ea5cea9c0650fe6fe",
    "arch:dc1051ce8daa96c7d4657ffcc600992231da8aff2421b5e72e9f00a0a3a798d0",
    "arch:dfa983ef75c1cb9dc5780530a6f4d36ea41242e4a9de1903a1f9818ea057c3c8",
    "arch:e1077a94466a205536c6e25a117027d5f77c06c0cc55482f925787412c9513ea",
)
R_RETIRED_PUBLICATION_SITE_IDS = (
    "arch:014cf5f6c0d82c4f638f4780ad77fe541e1a0092c3eb306e661449f82c8f2dbd",
    "arch:5b59c553e5646f532d691a4109ab356d242a1cb16931883a1db7d73c3ad6d75e",
    "arch:71a72909b9612fe9617c499e0c9f4c393f22e434f69855245b80599bc44d975b",
    "arch:842b5bc82d3ca9fc358a2998a88eb870a77835519e3c9d65f47c320b9c9de591",
    "arch:aab4417a8f69291dc742ca422e9ed3b0cff932c11ecff5a9a689b0e0386e7956",
    "arch:cc1fb68ebe2ed455c391ed4334afea731abf630c2645500a04b9f2f80ed0bf5b",
    "arch:e5f77c7061275206472263c4622936b746460ad02da0ace3355a6d12d902b935",
)
R3_RETIRED_READ_BACK_SITE_IDS = (
    "arch:3da3a455be15a5691ae0e5c6ee440ccc87748962a6709344209139af5a040fc7",
    "arch:7a2d5165f7eea9c3e3e8ed6eb54f6b5e54edcf3918c772855809ca81d336d13a",
    "arch:9ca182f482eb669a841a1d7963941f0db3fd0ce8ce40a6bb504ae808a94c7359",
    "arch:fb1c4ba11f65e136b046580b21bfa48ecbaf284c3127c14bafdbbeafe73c04f2",
)
R4_RETIRED_ROUTINE_REVIEW_SITE_IDS = (
    "arch:21249548ec49a61a193676dcd02608d2f3ed5484e2afe94139d14d8007d4464c",
    "arch:2b0ea53e6a2688814cf77568739df0a8fd5b2e45c4cc8febe56219beade13da6",
    "arch:396a2038ad4bc5fc265d4396c9e3faf0be363b297ba6115c133eb6d4628a35f3",
    "arch:66f538a6432ca0eeae8f75f61356f3fde522871a47352401b229e15a3a768c89",
    "arch:76a317ad927e3d5f5b571d22c7d88ab5e31e6f5ac164e03bab83c6e19069ca97",
    "arch:78cccbf0b14f6d2a62dd8d0dd6ea0395630cb3ee07f189abd7d93d25474582b6",
    "arch:848ba8e26d052a1a02586472f39059e49d7c2107988c596389e241580be989d1",
    "arch:8582ce2f763f23faf24cf6caccfdab7c250da0710fa0cb0541ea05ffbae36f0f",
    "arch:919b6a8be06d3c778c5258a9e4f0b2fd82de7c44b21fc0c8bfd8497c62bcd068",
    "arch:a02fe27ee53b83fb3a8aae6d01ea0383eb33851ed3f9b51c48b3713b8716e45a",
    "arch:a7d6c2e592b64acb06b900bbac5ca5712577639eafa7bc95ac9429130032fff1",
    "arch:b588ec35b758ad0dcba3f625d67293812e2e5bef7eeb558f9d362241e09871b8",
    "arch:bc7ca22c6d4590c243dee1ac25dd39cc2c87d789686cb7751fd0bc19eeebaab0",
    "arch:cf411151eae4a30c69ea344e63655577d1cce073c0f24a8c246a2b19acadcd53",
    "arch:cfdbfcff38abdc49e8631e2c6af6acae0b1d89958c98b3c8194fe97853f3be1b",
    "arch:db3e20561f2a5aac64bc0f1f1c37e07aa1509639a76d36489ad1e3f9da09980e",
    "arch:ddeaa9d96c4795a9a58c1973fb89ed5483865a641e9733ed76a806a3a9eca999",
    "arch:ef6f56ffd387d3538cc9e04112ef37bf646ff2f5fb35e4f76a19f6a58d8d5940",
    "arch:fb2f2abe1096e870dad5228173f4a70708e242d0f6f19853d4525c1e7c42080e",
)
S_RETIRED_PREFETCH_SITE_IDS = (
    "arch:18029d733218880572bb07bad67caf20d8bf45f52efebbd637b6030f7f874d81",
    "arch:3627981bd6863ed029059ed65b070ba8398e4924777780899adf5a3ccdcd9e8b",
    "arch:69f9252bd7630104d82e93f126e4658b598aaa7346629c8eff81ce2130c93e7a",
    "arch:aa400f09e2db4b388521b3792e5e6fc883d12f28310984567254487e0166f088",
    "arch:c5854d44b491487581356c43000abf8b3ff484c76a7461b8d224e3cb1fe38969",
)
RETIREMENT_HISTORY = {
    "exception_semantic_ids": tuple(
        sorted(
            (
                P3_RETIRED_CLASSIFICATION_LAB_EDGE_ID,
                *P4C_RETIRED_EXCEPTION_IDS,
                P4_RETIRED_SEARCH_LAB_PRIVATE_EXCEPTION_ID,
                P4_RETIRED_SEARCH_LAB_EDGE_ID,
                P2_RETIRED_PRIVATE_EXCEPTION_ID,
                P4A_RETIRED_OPS_LAB_EDGE_ID,
                P4A_RETIRED_OPS_LAB_DYNAMIC_SITE_ID,
                *P5_RETIRED_EXCEPTION_IDS,
                *P6_RETIRED_EXCEPTION_IDS,
                *P8_RETIRED_EXCEPTION_IDS,
                *Q_RETIRED_EXCEPTION_IDS,
                *R_RETIRED_EXCEPTION_IDS,
            )
        )
    ),
    "cross_domain_site_semantic_ids": tuple(
        sorted(
            (
                *P2_RETIRED_SITE_IDS,
                P3_RETIRED_PROVIDER_SITE_ID,
                *P4C_RETIRED_DECISION_LAB_SITE_IDS,
                *P4C_RETIRED_MOVED_SITE_IDS,
                P4A_RETIRED_OPS_LAB_STATIC_SITE_ID,
                *P4A_RETIRED_MOVED_SITE_IDS,
                P4_RETIRED_SEARCH_LAB_SITE_ID,
                *P5_RETIRED_SITE_IDS,
                *P6_RETIRED_SITE_IDS,
                *P8_RETIRED_SITE_IDS,
                Q_RETIRED_CODEX_STORE_SITE_ID,
                Q_RETIRED_CLAUDE_CODE_STORE_SITE_ID,
                *R_RETIRED_PUBLICATION_SITE_IDS,
                *R3_RETIRED_READ_BACK_SITE_IDS,
                *R4_RETIRED_ROUTINE_REVIEW_SITE_IDS,
                *S_RETIRED_PREFETCH_SITE_IDS,
            )
        )
    ),
    "production_to_lab_edge_semantic_ids": tuple(
        sorted(
            (
                P3_RETIRED_CLASSIFICATION_LAB_EDGE_ID,
                P4C_RETIRED_EXCEPTION_IDS[1],
                P4_RETIRED_SEARCH_LAB_EDGE_ID,
                P4A_RETIRED_OPS_LAB_EDGE_ID,
                P5_RETIRED_EXCEPTION_IDS[2],
            )
        )
    ),
    "production_to_lab_static_site_semantic_ids": tuple(
        sorted(
            (
                *P2_RETIRED_SITE_IDS,
                P3_RETIRED_PROVIDER_SITE_ID,
                *P4C_RETIRED_DECISION_LAB_SITE_IDS,
                P4A_RETIRED_OPS_LAB_STATIC_SITE_ID,
                P4_RETIRED_SEARCH_LAB_SITE_ID,
                *P5_RETIRED_STATIC_SITE_IDS,
            )
        )
    ),
    "production_to_lab_dynamic_site_semantic_ids": (
        P4A_RETIRED_OPS_LAB_DYNAMIC_SITE_ID,
    ),
    "compatibility_semantic_ids": (),
}

DiagnosticPath = tuple[str | int, ...]


def _diagnostic_line_entries(
    value: Any, path: DiagnosticPath = ()
) -> list[tuple[DiagnosticPath, Any]]:
    if isinstance(value, dict):
        entries: list[tuple[DiagnosticPath, Any]] = []
        for key, item in value.items():
            item_path = (*path, key)
            if key == "line":
                entries.append((item_path, item))
            else:
                entries.extend(_diagnostic_line_entries(item, item_path))
        return entries
    if isinstance(value, list):
        return [
            entry
            for index, item in enumerate(value)
            for entry in _diagnostic_line_entries(item, (*path, index))
        ]
    return []


def _without_diagnostic_lines(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_diagnostic_lines(item)
            for key, item in value.items()
            if key != "line"
        }
    if isinstance(value, list):
        return [_without_diagnostic_lines(item) for item in value]
    return value


def _assert_diagnostic_line_contract(recorded: Any, built: Any) -> None:
    recorded_entries = _diagnostic_line_entries(recorded)
    built_entries = _diagnostic_line_entries(built)
    assert recorded_entries
    assert built_entries
    assert all(
        type(line) is int
        for _path, line in (*recorded_entries, *built_entries)
    )
    recorded_paths = [path for path, _line in recorded_entries]
    built_paths = [path for path, _line in built_entries]
    assert len(recorded_paths) == len(built_paths)
    assert set(recorded_paths) == set(built_paths)
    assert _without_diagnostic_lines(built) == _without_diagnostic_lines(recorded)


EMPTY_EXCEPTION_VIOLATIONS = {
    "ledger_load_error": "",
    "ledger_schema_version": [],
    "ledger_source_baseline_head_drift": [],
    "ledger_baseline_sha256_drift": [],
    "new_exception_ids": [],
    "unrecorded_exception_ids": [],
    "stale_exception_ids": [],
    "baseline_semantic_id_non_subset": [],
    "exception_identity_mismatches": [],
    "exception_content_mismatches": [],
    "exception_metadata_missing": [],
    "duplicate_exception_ids": [],
    "new_cross_domain_site_ids": [],
    "unrecorded_cross_domain_site_ids": [],
    "stale_cross_domain_site_ids": [],
    "baseline_site_semantic_id_non_subset": [],
    "site_identity_mismatches": [],
    "site_content_mismatches": [],
    "duplicate_site_ids": [],
    "site_count_drift": {},
    "ledger_count_drift": {},
    "production_to_lab_edge_growth": [],
    "production_to_lab_static_site_growth": [],
    "production_to_lab_dynamic_site_growth": [],
    "compatibility_contract_drift": {},
    "compatibility_metadata_missing": [],
    "duplicate_compatibility_ids": [],
    "seed_load_error": "",
    "seed_schema_version": [],
    "seed_source_baseline_head_drift": [],
    "previous_seed_source_baseline_head_drift": [],
    "seed_source_baseline_head_history_drift": {},
    "seed_structure_errors": [],
    "seed_universe_drift": {},
    "seed_active_retired_overlap": {},
    "seed_current_drift": {},
    "retired_id_reintroductions": {},
    "seed_retired_regressions": {},
    "seed_active_growth": {},
    "duplicate_seed_ids": {},
    "seed_count_drift": {},
    "previous_seed_load_error": "",
    "previous_seed_schema_version": [],
}


def _load_script() -> ModuleType:
    path = ROOT / "scripts" / "architecture_baseline.py"
    spec = importlib.util.spec_from_file_location("architecture_baseline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def architecture() -> ModuleType:
    return _load_script()


@pytest.fixture(scope="module")
def baseline() -> dict[str, Any]:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current(architecture: ModuleType) -> dict[str, Any]:
    return architecture.scan_repository(ROOT, captured_at="verification")


def _exception_inputs(
    current: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    return (
        copy.deepcopy(current["worktree_source"]),
        copy.deepcopy(current["architecture_exception_ledger"]),
        copy.deepcopy(current["compatibility_contracts"]),
        copy.deepcopy(current["architecture_exception_baseline"]),
        copy.deepcopy(current["frozen_architecture_exception_reference"]),
        copy.deepcopy(current["previous_architecture_exception_baseline"]),
    )


def _exception_violations(
    architecture: ModuleType,
    source: dict[str, Any],
    ledger: dict[str, Any],
    compatibility: list[dict[str, Any]],
    seed: dict[str, Any],
    frozen: dict[str, Any],
    previous_seed: dict[str, Any],
) -> dict[str, Any]:
    return architecture._architecture_exception_violations(
        source,
        ledger,
        compatibility,
        seed,
        frozen,
        previous_seed,
    )


def _sync_seed_and_ledger_counts(
    architecture: ModuleType,
    source: dict[str, Any],
    ledger: dict[str, Any],
    compatibility: list[dict[str, Any]],
    seed: dict[str, Any],
) -> None:
    retired = {
        field: architecture._seed_ids(seed, field, "retired")
        for field in architecture.EXCEPTION_BASELINE_ID_FIELDS
    }
    counts = architecture._exception_counts(source, compatibility, retired)
    seed["counts"] = counts
    ledger["counts"] = counts["active"]
    ledger["baseline_sha256"] = architecture._canonical_sha256(seed)


def _move_seed_id(
    seed: dict[str, Any],
    field: str,
    semantic_id: str,
    *,
    target: str,
) -> None:
    source = "retired" if target == "active" else "active"
    seed[field][source].remove(semantic_id)
    seed[field][target].append(semantic_id)
    seed[field][target].sort()


def _synthetic_dynamic_import_site(
    architecture: ModuleType,
    template: dict[str, Any],
    *,
    source_module: str,
    target_package: str = "ops",
    target_module: str = "chronovisor.ops.synthetic_target",
) -> dict[str, Any]:
    row = {
        **template,
        "category": "dynamic_import",
        "source_package": "ops",
        "source_module": source_module,
        "scope": "<module>",
        "scope_kind": "module",
        "statement_kind": "__import__",
        "target_package": target_package,
        "target_module": target_module,
        "symbols": [],
        "occurrence": 1,
        "line": 1,
    }
    row["semantic_id"] = architecture._semantic_id(row)
    return row


def _assert_exact_retirement_history(
    architecture: ModuleType,
    seed: dict[str, Any],
) -> None:
    assert set(RETIREMENT_HISTORY) == set(
        architecture.EXCEPTION_BASELINE_ID_FIELDS
    )
    assert {
        field: tuple(seed[field]["retired"])
        for field in architecture.EXCEPTION_BASELINE_ID_FIELDS
    } == RETIREMENT_HISTORY
    assert seed["counts"]["retired"] == {
        field: len(retired_ids)
        for field, retired_ids in RETIREMENT_HISTORY.items()
    }
    assert all(
        set(seed[field]["active"]).isdisjoint(seed[field]["retired"])
        for field in architecture.EXCEPTION_BASELINE_ID_FIELDS
    )


def _without_persisted_retirement_history(
    architecture: ModuleType,
    seed: dict[str, Any],
) -> dict[str, Any]:
    normalized = copy.deepcopy(seed)
    _assert_exact_retirement_history(architecture, normalized)
    for field, retired_ids in RETIREMENT_HISTORY.items():
        normalized[field]["active"] = sorted(
            (*normalized[field]["active"], *retired_ids)
        )
        normalized[field]["retired"] = []
    normalized["counts"]["retired"] = {
        field: 0 for field in RETIREMENT_HISTORY
    }
    active_counts = normalized["counts"]["active"]
    active_counts["exceptions"] += len(RETIREMENT_HISTORY["exception_semantic_ids"])
    active_counts["by_category"]["cross_domain_edge"] += 5
    active_counts["by_category"]["dynamic_import"] = (
        1 + len(Q_RETIRED_EXCEPTION_IDS) + len(R_RETIRED_EXCEPTION_IDS)
    )
    active_counts["by_category"]["private_symbol_import"] = 31
    active_counts["by_category"]["schema_manifest_implementation_import"] = len(
        P6_RETIRED_EXCEPTION_IDS
    )
    active_counts["schema_manifest_implementation"] = {
        "background_decision_schemas": {"statements": 1, "symbols": 5},
        "production_decision_schemas": {"statements": 11, "symbols": 13},
    }
    active_counts["cross_domain_sites"] += len(
        RETIREMENT_HISTORY["cross_domain_site_semantic_ids"]
    )
    active_counts["production_to_lab_edges"] += len(
        RETIREMENT_HISTORY["production_to_lab_edge_semantic_ids"]
    )
    active_counts["production_to_lab_static_sites"] += len(
        RETIREMENT_HISTORY["production_to_lab_static_site_semantic_ids"]
    )
    active_counts["production_to_lab_dynamic_sites"] += len(
        RETIREMENT_HISTORY["production_to_lab_dynamic_site_semantic_ids"]
    )
    return normalized


def test_baseline_records_complete_pre_campaign_inventory(
    baseline: dict[str, Any],
) -> None:
    assert baseline["schema_version"] == 1
    assert baseline["campaign"] == "O"
    assert baseline["campaign_started_at"] == "2026-08-06T12:44:00+09:00"
    assert baseline["captured_at"] != baseline["campaign_started_at"]
    assert (
        "before the authoritative isolated full suite"
        in baseline["captured_at_semantics"]
    )
    assert baseline["repository"]["head_at_capture"] == (
        "d341d575f56c1f3217840e20a0dd144799244a89"
    )
    assert baseline["repository"]["worktree"]["capture_phase"] == (
        "Campaign O frozen pre-full, pre-commit implementation worktree"
    )
    assert baseline["repository"]["pre_campaign_source_head"] == (
        "a17b8704e2a69e1df1dc3466e956edee77fec870"
    )
    assert baseline["source"]["totals"] == {
        "modules": 281,
        "lines": 200634,
        "functions": 4743,
    }
    assert baseline["source"]["package_count"] == 13
    assert "knowledge_graph" in baseline["source"]["packages"]
    assert baseline["source"]["namespace_packages"] == []
    assert len(baseline["source"]["modules"]) == 281
    assert len(baseline["source"]["module_hotspots"]) == 25
    assert len(baseline["source"]["function_hotspots"]) == 50
    assert len(baseline["source"]["python_source_bytes_sha256"]) == 64
    assert baseline["architecture"]["edge_count"] == 95
    assert len(baseline["architecture"]["strongly_connected_components"][0]) == 12
    assert baseline["architecture"]["strongly_connected_components"][1] == ["core"]
    assert len(baseline["console_entrypoints"]) == 51
    assert len(baseline["tracked_launchd_plists"]) == 7
    assets = baseline["source"]["tracked_non_python_assets"]
    assert assets["file_count"] > 0
    assert assets["total_bytes"] > 0
    assert len(assets["manifest_sha256"]) == 64
    assert len(assets["files"]) == assets["file_count"]
    assert all(
        row["path"] and row["bytes"] >= 0 and len(row["sha256"]) == 64
        for row in assets["files"]
    )
    assert assets["frontend_totals"] == {"file_count": 11, "lines": 16979}
    assert assets["asset_hotspots"][0]["path"] == (
        "src/chronovisor/dashboard_static/cortex.js"
    )
    assert assets["asset_hotspots"][0]["lines"] == 5606


def test_baseline_labels_repository_contract_hash_semantics(
    baseline: dict[str, Any], current: dict[str, Any]
) -> None:
    hashes = baseline["contract_hashes"]
    authority = hashes["decision_authority"]
    schema = hashes["production_schema_manifest"]
    signature = hashes["production_signature_manifest"]
    assert authority["lane_contract_case_manifest_sha256"] == (
        "a3a8b84e249b4a6bf36ba3f3584bd6fae45ac4fa521c83c34637879e9b2473eb"
    )
    assert schema["canonical_mapping_sha256"]["sha256"] == (
        "1541981873a0669f5ef7234c9b4490fe3c3f00872d1a584b182a3a33799fbea2"
    )
    assert schema["artifact_validator_sorted_rows_sha256"]["sha256"] == (
        "299b9e5c7c1b5f0195e6437890c111c82cbf63545333eebab83e7b42a870ed58"
    )
    assert signature["sha256"] == (
        "057a9edf3c0d88f579bef8c0836535714aefba73fdba6a15b9b9072f46540f05"
    )
    assert hashes == current["contract_hashes"]


def test_baseline_keeps_live_evidence_out_of_repository_gates(
    baseline: dict[str, Any],
) -> None:
    exclusions = {row["evidence"] for row in baseline["live_only_exclusions"]}
    assert exclusions == {
        "production_runtime_archives_and_running_processes",
        "production_authority_artifact_identity",
        "recall_save_ingest_and_repair_live_behavior",
        "dashboard_cortex_dom_latency_frame_and_memory",
    }


def test_current_architecture_does_not_weaken_baseline(
    architecture: ModuleType,
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> None:
    report = architecture.architecture_fitness(baseline, current)

    assert report["passed"] is True
    assert report["violations"] == {
        "new_edges": [],
        "scc_regressions": [],
        "namespace_packages": [],
        "entrypoint_drift": {},
        "launchd_drift": {},
        "contract_hash_drift": {},
        "architecture_contract_drift": {},
        **EMPTY_EXCEPTION_VIOLATIONS,
    }


def test_architecture_fitness_rejects_new_edge_and_scc_growth(
    architecture: ModuleType,
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> None:
    regressed = copy.deepcopy(current)
    worktree = regressed["worktree_architecture"]
    worktree["edges"].append(["core", "ops"])
    worktree["strongly_connected_components"] = (
        architecture._strongly_connected_components(
            regressed["worktree_source"]["packages"], worktree["edges"]
        )
    )

    report = architecture.architecture_fitness(baseline, regressed)

    assert report["passed"] is False
    assert report["violations"]["new_edges"] == [["core", "ops"]]
    assert len(report["violations"]["scc_regressions"][0]) == 12


def test_namespace_package_is_inventoried_and_rejected(
    architecture: ModuleType,
    baseline: dict[str, Any],
    current: dict[str, Any],
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "src" / "chronovisor"
    (package_root / "core").mkdir(parents=True)
    (package_root / "core" / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "shadow").mkdir()
    (package_root / "shadow" / "worker.py").write_text(
        "from chronovisor import core\n", encoding="utf-8"
    )

    source, edges = architecture._source_inventory(package_root)

    assert source["packages"] == ["core", "shadow"]
    assert source["namespace_packages"] == ["shadow"]
    assert edges == [["shadow", "core"]]

    regressed = copy.deepcopy(current)
    regressed["worktree_source"]["packages"].append("shadow")
    regressed["worktree_source"]["namespace_packages"] = ["shadow"]
    regressed["worktree_architecture"]["strongly_connected_components"].append(
        ["shadow"]
    )
    report = architecture.architecture_fitness(baseline, regressed)
    assert report["passed"] is False
    assert report["violations"]["namespace_packages"] == ["shadow"]


def test_python_source_digest_distinguishes_crlf_from_lf(
    architecture: ModuleType, tmp_path: Path
) -> None:
    package_root = tmp_path / "src" / "chronovisor"
    package = package_root / "core"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    module = package / "sample.py"
    module.write_bytes(b"def value():\r\n    return 1\r\n")
    crlf, _edges = architecture._source_inventory(package_root)
    module.write_bytes(b"def value():\n    return 1\n")
    lf, _edges = architecture._source_inventory(package_root)

    assert crlf["totals"] == lf["totals"]
    assert crlf["python_source_bytes_sha256"] != lf["python_source_bytes_sha256"]


def test_entrypoint_and_launchd_surfaces_still_match_baseline(
    baseline: dict[str, Any], current: dict[str, Any]
) -> None:
    assert current["console_entrypoints"] == baseline["console_entrypoints"]
    assert current["tracked_launchd_plists"] == baseline["tracked_launchd_plists"]
    core_contract = next(
        contract
        for contract in baseline["architecture"]["contracts"]
        if contract["name"] == "Core cannot depend on domain or outer layers"
    )
    assert "chronovisor.knowledge_graph" in core_contract["forbidden_modules"]


def test_architecture_fitness_allows_and_reports_package_edge_removal(
    architecture: ModuleType,
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> None:
    reduced = copy.deepcopy(current)
    reduced["worktree_source"]["packages"].remove("knowledge_graph")
    worktree = reduced["worktree_architecture"]
    worktree["edges"] = [
        edge for edge in worktree["edges"] if "knowledge_graph" not in edge
    ]
    worktree["strongly_connected_components"] = (
        architecture._strongly_connected_components(
            reduced["worktree_source"]["packages"], worktree["edges"]
        )
    )

    report = architecture.architecture_fitness(baseline, reduced)

    assert report["passed"] is True
    assert report["observations"]["missing_packages"] == ["knowledge_graph"]
    assert report["violations"]["new_edges"] == []
    assert report["violations"]["scc_regressions"] == []
    assert report["violations"]["namespace_packages"] == []


def test_import_scanner_covers_function_scope_root_and_relative_imports(
    architecture: ModuleType, tmp_path: Path
) -> None:
    package_root = tmp_path / "src" / "chronovisor"
    for package in ("core", "ops"):
        path = package_root / package
        path.mkdir(parents=True)
        (path / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "core" / "sample.py").write_text(
        "def load():\n    from chronovisor import ops\n",
        encoding="utf-8",
    )
    (package_root / "ops" / "sample.py").write_text(
        "def load():\n    from .. import core\n",
        encoding="utf-8",
    )

    _source, edges = architecture._source_inventory(package_root)

    assert edges == [["core", "ops"], ["ops", "core"]]


def test_statement_inventory_tracks_registry_to_implementation_direction(
    architecture: ModuleType, tmp_path: Path
) -> None:
    package_root = tmp_path / "src" / "chronovisor"
    for package in ("decision", "lab", "ops"):
        path = package_root / package
        path.mkdir(parents=True)
        (path / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "decision" / "decision_schema_manifest.py").write_text(
        "def production_decision_schemas():\n"
        "    from chronovisor.ops.schemas import ALPHA_SCHEMA, BETA_SCHEMA\n"
        "\n"
        "def background_decision_schemas():\n"
        "    from chronovisor.decision.graph_decisions import BACKGROUND_SCHEMA\n",
        encoding="utf-8",
    )
    (package_root / "ops" / "sample.py").write_text(
        "import importlib\n"
        "from chronovisor.lab._private import _value\n"
        "from chronovisor.decision.decision_schema_manifest import "
        "NON_DECISION_FIELDS\n"
        "\n"
        "def load():\n"
        "    importlib.import_module('chronovisor.lab.worker')\n"
        "    return __import__('chronovisor.lab.plugin')\n",
        encoding="utf-8",
    )

    source, edges = architecture._source_inventory(package_root)
    rows = source["import_sites"]
    schema_rows = [
        row
        for row in rows
        if row["category"] == "schema_manifest_implementation_import"
    ]

    assert edges == [
        ["decision", "ops"],
        ["ops", "decision"],
        ["ops", "lab"],
    ]
    assert len(schema_rows) == 2
    assert {row["source_module"] for row in schema_rows} == {
        "chronovisor.decision.decision_schema_manifest"
    }
    assert {row["scope"] for row in schema_rows} == {
        "background_decision_schemas",
        "production_decision_schemas",
    }
    assert sum(len(row["symbols"]) for row in schema_rows) == 3
    assert all(
        row["target_module"] != "chronovisor.decision.decision_schema_manifest"
        for row in schema_rows
    )
    assert len({row["semantic_id"] for row in rows}) == len(rows)


def test_schema_registry_detects_all_uppercase_same_package_constants(
    architecture: ModuleType, tmp_path: Path
) -> None:
    package_root = tmp_path / "src" / "chronovisor"
    decision = package_root / "decision"
    decision.mkdir(parents=True)
    (decision / "__init__.py").write_text("", encoding="utf-8")
    (decision / "decision_schema_manifest.py").write_text(
        "def production_decision_schemas():\n"
        "    from chronovisor.decision.schemas import (\n"
        "        SCHEMA, FOO_SCHEMA_VERSION, FOO_SCHEMA_V2, lower_name\n"
        "    )\n"
        "\n"
        "def background_decision_schemas():\n"
        "    from chronovisor.decision.schemas import (\n"
        "        BACKGROUND_SCHEMA_VERSION, mixed_Name\n"
        "    )\n",
        encoding="utf-8",
    )

    source, edges = architecture._source_inventory(package_root)
    schema_rows = [
        row
        for row in source["import_sites"]
        if row["category"] == "schema_manifest_implementation_import"
    ]

    assert edges == []
    assert len(schema_rows) == 2
    assert {row["target_module"] for row in schema_rows} == {
        "chronovisor.decision.schemas"
    }
    assert {row["scope"]: row["symbols"] for row in schema_rows} == {
        "production_decision_schemas": [
            "FOO_SCHEMA_V2",
            "FOO_SCHEMA_VERSION",
            "SCHEMA",
        ],
        "background_decision_schemas": ["BACKGROUND_SCHEMA_VERSION"],
    }


def test_statement_semantic_identity_and_content_ignore_line_moves(
    architecture: ModuleType, tmp_path: Path
) -> None:
    package_root = tmp_path / "src" / "chronovisor"
    for package in ("core", "ops"):
        path = package_root / package
        path.mkdir(parents=True)
        (path / "__init__.py").write_text("", encoding="utf-8")
    module = package_root / "core" / "sample.py"
    module.write_text(
        "from chronovisor.ops import public_api\n",
        encoding="utf-8",
    )
    before, _edges = architecture._source_inventory(package_root)
    module.write_text(
        "\n\n\nfrom chronovisor.ops import public_api\n",
        encoding="utf-8",
    )
    after, _edges = architecture._source_inventory(package_root)

    before_site = before["import_sites"][0]
    after_site = after["import_sites"][0]
    assert before_site["line"] == 1
    assert after_site["line"] == 4
    assert before_site["semantic_id"] == after_site["semantic_id"]
    assert before_site["content_sha256"] == after_site["content_sha256"]
    assert (
        architecture._architecture_exception_rows(before)[0]["semantic_id"]
        == architecture._architecture_exception_rows(after)[0]["semantic_id"]
    )


def test_current_exception_ledger_seed_and_schema_inventory_are_exact(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility, seed, frozen, previous = _exception_inputs(current)
    previous = copy.deepcopy(seed)
    detected = architecture._architecture_exception_rows(source)
    detected_ids = {row["semantic_id"] for row in detected}
    ledger_ids = {row["semantic_id"] for row in ledger["exceptions"]}
    edge_rows = [
        row for row in ledger["exceptions"] if row["category"] == "cross_domain_edge"
    ]
    raw_cross_sites = [
        row
        for row in source["import_sites"]
        if row["category"] == "cross_domain_import"
    ]
    counts = seed["counts"]["active"]

    assert detected_ids == ledger_ids == set(seed["exception_semantic_ids"]["active"])
    _assert_exact_retirement_history(architecture, seed)
    assert len(edge_rows) == current["worktree_architecture"]["edge_count"] == 90
    assert sum(len(row["sites"]) for row in edge_rows) == len(raw_cross_sites) == 1256
    assert {
        field: counts[field]
        for field in (
            "exceptions",
            "cross_domain_sites",
            "production_to_lab_edges",
            "production_to_lab_static_sites",
            "production_to_lab_dynamic_sites",
            "compatibility_contracts",
        )
    } == {
        "exceptions": 90,
        "cross_domain_sites": 1256,
        "production_to_lab_edges": 0,
        "production_to_lab_static_sites": 0,
        "production_to_lab_dynamic_sites": 0,
        "compatibility_contracts": 289,
    }
    assert counts["by_category"] == {
        "cross_domain_edge": 90,
    }
    assert counts["compatibility_by_kind"] == {
        "console_entrypoint": 51,
        "lab_dispatch": 15,
        "module_string": 223,
    }
    assert counts["schema_manifest_implementation"] == {
        "background_decision_schemas": {"statements": 0, "symbols": 0},
        "production_decision_schemas": {"statements": 0, "symbols": 0},
    }
    assert (
        _exception_violations(
            architecture,
            source,
            ledger,
            compatibility,
            seed,
            frozen,
            previous,
        )
        == EMPTY_EXCEPTION_VIOLATIONS
    )


@pytest.mark.parametrize("mutation", ["extra", "wrong"])
def test_retirement_history_rejects_extra_or_wrong_id(
    architecture: ModuleType,
    current: dict[str, Any],
    mutation: str,
) -> None:
    seed = copy.deepcopy(current["architecture_exception_baseline"])
    if mutation == "extra":
        seed["exception_semantic_ids"]["retired"].append("arch:" + "0" * 64)
    else:
        seed["cross_domain_site_semantic_ids"]["retired"][0] = "arch:" + "f" * 64

    with pytest.raises(AssertionError):
        _assert_exact_retirement_history(architecture, seed)


def test_actual_schema_registry_has_no_implementation_imports(
    current: dict[str, Any],
) -> None:
    rows = [
        row
        for row in current["architecture_exception_ledger"]["exceptions"]
        if row["category"] == "schema_manifest_implementation_import"
    ]
    assert rows == []


def test_new_sensitive_exception_cannot_self_authorize_in_ledger_and_seed(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility, seed, frozen, previous = _exception_inputs(current)
    previous = copy.deepcopy(seed)
    template = next(
        row for row in source["import_sites"] if row["category"] == "cross_domain_import"
    )
    import_site = _synthetic_dynamic_import_site(
        architecture,
        template,
        source_module="chronovisor.ops.new_dynamic_site",
    )
    source["import_sites"].append(import_site)
    ledger["exceptions"].append(
        {
            **import_site,
            "owner": "chronovisor.ops",
            "deadline": "2026-12-31",
            "removal_campaign": "S",
            "rationale": "Synthetic forbidden exception.",
        }
    )
    seed["exception_semantic_ids"]["active"].append(import_site["semantic_id"])
    seed["exception_semantic_ids"]["active"].sort()
    _sync_seed_and_ledger_counts(architecture, source, ledger, compatibility, seed)

    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
        seed,
        frozen,
        previous,
    )

    assert violations["new_exception_ids"] == []
    assert violations["unrecorded_exception_ids"] == []
    assert violations["seed_universe_drift"]["exception_semantic_ids"]["added"] == [
        import_site["semantic_id"]
    ]
    assert violations["seed_active_growth"]["exception_semantic_ids"] == [
        import_site["semantic_id"]
    ]


def test_existing_edge_replacement_site_can_be_seeded_with_exact_ledger_match(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility, seed, frozen, previous = _exception_inputs(current)
    previous = copy.deepcopy(seed)
    edge = next(
        row
        for row in ledger["exceptions"]
        if row["category"] == "cross_domain_edge"
        and row["target_package"] != "lab"
        and len(row["sites"]) > 1
    )
    template = next(
        row
        for row in source["import_sites"]
        if row["category"] == "cross_domain_import"
        and row["source_package"] == edge["source_package"]
        and row["target_package"] == edge["target_package"]
    )
    import_site = {
        **template,
        "source_module": f"chronovisor.{edge['source_package']}.new_public_site",
        "scope": "<module>",
        "scope_kind": "module",
        "occurrence": 1,
        "line": 1,
        "content_sha256": "a" * 64,
    }
    import_site["semantic_id"] = architecture._semantic_id(import_site)
    source["import_sites"].append(import_site)
    replacement = next(
        row
        for row in architecture._architecture_exception_rows(source)
        if row["category"] == "cross_domain_edge"
        and row["source_package"] == edge["source_package"]
        and row["target_package"] == edge["target_package"]
    )
    replacement.update(
        {field: edge[field] for field in architecture.EXCEPTION_METADATA_FIELDS}
    )
    ledger["exceptions"] = [
        replacement if row["semantic_id"] == edge["semantic_id"] else row
        for row in ledger["exceptions"]
    ]
    seed["cross_domain_site_semantic_ids"]["active"].append(import_site["semantic_id"])
    seed["cross_domain_site_semantic_ids"]["active"].sort()
    _sync_seed_and_ledger_counts(architecture, source, ledger, compatibility, seed)

    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
        seed,
        frozen,
        previous,
    )

    assert violations == EMPTY_EXCEPTION_VIOLATIONS


def test_site_gate_rejects_unrecorded_stale_duplicate_identity_content_and_count(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    inputs = _exception_inputs(current)
    source, ledger, compatibility, seed, frozen, previous = inputs
    edge = next(
        row
        for row in ledger["exceptions"]
        if row["category"] == "cross_domain_edge" and len(row["sites"]) > 1
    )
    site = edge["sites"][0]
    edge["sites"].remove(site)
    violations = _exception_violations(
        architecture, source, ledger, compatibility, seed, frozen, previous
    )
    assert violations["unrecorded_cross_domain_site_ids"] == [site["semantic_id"]]
    assert violations["baseline_site_semantic_id_non_subset"] == [site["semantic_id"]]
    assert violations["site_count_drift"]

    source, ledger, compatibility, seed, frozen, previous = _exception_inputs(current)
    edge = next(
        row
        for row in ledger["exceptions"]
        if row["category"] == "cross_domain_edge" and len(row["sites"]) > 1
    )
    original = edge["sites"][0]
    stale = {
        **original,
        "source_module": f"{original['source_module']}.stale",
        "occurrence": 1,
    }
    stale_identity = {
        "category": "cross_domain_import",
        "source_package": edge["source_package"],
        "target_package": edge["target_package"],
        **stale,
    }
    stale["semantic_id"] = architecture._semantic_id(stale_identity)
    edge["sites"].append(stale)
    edge["sites"].append(copy.deepcopy(original))
    for recorded in edge["sites"]:
        if recorded["semantic_id"] == original["semantic_id"]:
            recorded["content_sha256"] = "0" * 64
    edge["sites"][1]["target_module"] += ".tampered"
    violations = _exception_violations(
        architecture, source, ledger, compatibility, seed, frozen, previous
    )
    assert stale["semantic_id"] in violations["stale_cross_domain_site_ids"]
    assert original["semantic_id"] in violations["duplicate_site_ids"]
    assert original["semantic_id"] in violations["site_content_mismatches"]
    assert violations["site_identity_mismatches"]
    assert violations["site_count_drift"]


def test_exception_rows_reject_unrecorded_stale_duplicate_content_and_metadata(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility, seed, frozen, previous = _exception_inputs(current)
    template = next(
        row for row in source["import_sites"] if row["category"] == "cross_domain_import"
    )
    dynamic = _synthetic_dynamic_import_site(
        architecture,
        template,
        source_module="chronovisor.ops.synthetic_dynamic",
    )
    source["import_sites"].append(dynamic)
    seed["exception_semantic_ids"]["active"].append(dynamic["semantic_id"])
    seed["exception_semantic_ids"]["active"].sort()
    _sync_seed_and_ledger_counts(architecture, source, ledger, compatibility, seed)
    violations = _exception_violations(
        architecture, source, ledger, compatibility, seed, frozen, previous
    )
    assert violations["unrecorded_exception_ids"] == [dynamic["semantic_id"]]
    assert violations["baseline_semantic_id_non_subset"] == [dynamic["semantic_id"]]

    source, ledger, compatibility, seed, frozen, previous = _exception_inputs(current)
    dynamic_rows = [
        _synthetic_dynamic_import_site(
            architecture,
            row,
            source_module=f"chronovisor.ops.synthetic_dynamic_{index}",
            target_module=f"chronovisor.ops.synthetic_target_{index}",
        )
        for index, row in enumerate(
            row
            for row in source["import_sites"]
            if row["category"] == "cross_domain_import"
        )
    ][:2]
    dynamic, identity_row = dynamic_rows
    source["import_sites"].extend(dynamic_rows)
    ledger["exceptions"].extend(
        {**row, **architecture._exception_metadata(row)} for row in dynamic_rows
    )
    seed["exception_semantic_ids"]["active"].extend(
        row["semantic_id"] for row in dynamic_rows
    )
    seed["exception_semantic_ids"]["active"].sort()
    _sync_seed_and_ledger_counts(architecture, source, ledger, compatibility, seed)
    stale = {
        **dynamic,
        "source_module": f"{dynamic['source_module']}.stale",
        "occurrence": 1,
        **architecture._exception_metadata(dynamic),
    }
    stale["semantic_id"] = architecture._semantic_id(stale)
    ledger["exceptions"].append(stale)
    duplicate = {
        **copy.deepcopy(dynamic),
        **architecture._exception_metadata(dynamic),
    }
    ledger["exceptions"].append(duplicate)
    for row in ledger["exceptions"]:
        if row["semantic_id"] == dynamic["semantic_id"]:
            row["content_sha256"] = "0" * 64
    identity_row = next(
        row
        for row in ledger["exceptions"]
        if row["category"] == "dynamic_import"
        and row["semantic_id"] != dynamic["semantic_id"]
    )
    identity_row["target_module"] += ".tampered"
    identity_row.pop("owner")
    ledger["counts"]["exceptions"] += 1
    violations = _exception_violations(
        architecture, source, ledger, compatibility, seed, frozen, previous
    )
    assert stale["semantic_id"] in violations["stale_exception_ids"]
    assert dynamic["semantic_id"] in violations["duplicate_exception_ids"]
    assert dynamic["semantic_id"] in violations["exception_content_mismatches"]
    assert identity_row["semantic_id"] in violations["exception_identity_mismatches"]
    assert {
        "semantic_id": identity_row["semantic_id"],
        "missing": ["owner"],
    } in violations["exception_metadata_missing"]
    assert violations["ledger_count_drift"]


def test_site_deletion_retires_monotonically_and_reintroduction_is_rejected(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility, seed, frozen, _previous = _exception_inputs(current)
    previous_seed = copy.deepcopy(seed)
    edge = next(
        row
        for row in ledger["exceptions"]
        if row["category"] == "cross_domain_edge"
        and row["target_package"] != "lab"
        and len(row["sites"]) > 1
    )
    site = edge["sites"][0]
    source["import_sites"] = [
        row
        for row in source["import_sites"]
        if row["semantic_id"] != site["semantic_id"]
    ]
    edge["sites"] = [
        row for row in edge["sites"] if row["semantic_id"] != site["semantic_id"]
    ]
    _move_seed_id(
        seed,
        "cross_domain_site_semantic_ids",
        site["semantic_id"],
        target="retired",
    )
    _sync_seed_and_ledger_counts(architecture, source, ledger, compatibility, seed)

    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
        seed,
        frozen,
        previous_seed,
    )
    assert violations == EMPTY_EXCEPTION_VIOLATIONS

    retired_seed = copy.deepcopy(seed)
    source, ledger, compatibility, _initial_seed, frozen, _previous = _exception_inputs(
        current
    )
    seed = copy.deepcopy(retired_seed)
    _move_seed_id(
        seed,
        "cross_domain_site_semantic_ids",
        site["semantic_id"],
        target="active",
    )
    _sync_seed_and_ledger_counts(architecture, source, ledger, compatibility, seed)
    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
        seed,
        frozen,
        retired_seed,
    )
    assert violations["seed_retired_regressions"]["cross_domain_site_semantic_ids"] == [
        site["semantic_id"]
    ]
    assert violations["seed_active_growth"] == {}


def test_production_to_lab_static_and_dynamic_site_growth_are_explicit(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility, seed, frozen, previous = _exception_inputs(current)
    static_template = next(
        row
        for row in source["import_sites"]
        if row["category"] == "cross_domain_import"
    )
    static_site = {
        **static_template,
        "source_package": "classification",
        "source_module": "chronovisor.classification.new_lab_static",
        "target_package": "lab",
        "target_module": "chronovisor.lab.new_static",
        "occurrence": 1,
        "line": 1,
    }
    static_site["semantic_id"] = architecture._semantic_id(static_site)
    dynamic_site = _synthetic_dynamic_import_site(
        architecture,
        static_template,
        source_module="chronovisor.ops.new_lab_dynamic",
        target_package="lab",
        target_module="chronovisor.lab.new_dynamic",
    )
    source["import_sites"].extend((static_site, dynamic_site))

    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
        seed,
        frozen,
        previous,
    )

    assert violations["production_to_lab_static_site_growth"] == [
        static_site["semantic_id"]
    ]
    assert violations["production_to_lab_dynamic_site_growth"] == [
        dynamic_site["semantic_id"]
    ]


def test_exception_metadata_routes_to_real_owner_and_removal_campaign(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    rows = current["architecture_exception_ledger"]["exceptions"]
    by_key = {
        (row["category"], row["source_package"], row["target_package"]): row
        for row in rows
    }

    assert ("cross_domain_edge", "classification", "lab") not in by_key
    assert ("private_symbol_import", "classification", "lab") not in by_key
    assert ("cross_domain_edge", "search", "lab") not in by_key
    assert ("private_symbol_import", "search", "lab") not in by_key
    assert ("cross_domain_edge", "decision", "lab") not in by_key
    assert ("private_symbol_import", "decision", "lab") not in by_key
    assert ("cross_domain_edge", "ops", "lab") not in by_key
    assert ("cross_domain_edge", "librarian", "lab") not in by_key
    assert P2_RETIRED_PRIVATE_EXCEPTION_ID not in {
        row["semantic_id"] for row in rows
    }
    assert current["architecture_exception_baseline"]["exception_semantic_ids"][
        "retired"
    ] == list(RETIREMENT_HISTORY["exception_semantic_ids"])
    assert current["architecture_exception_baseline"][
        "cross_domain_site_semantic_ids"
    ]["retired"] == list(RETIREMENT_HISTORY["cross_domain_site_semantic_ids"])
    assert current["architecture_exception_baseline"][
        "production_to_lab_edge_semantic_ids"
    ]["retired"] == list(RETIREMENT_HISTORY["production_to_lab_edge_semantic_ids"])
    assert current["architecture_exception_baseline"][
        "production_to_lab_static_site_semantic_ids"
    ]["retired"] == list(
        RETIREMENT_HISTORY["production_to_lab_static_site_semantic_ids"]
    )
    assert current["architecture_exception_baseline"][
        "production_to_lab_dynamic_site_semantic_ids"
    ]["retired"] == [P4A_RETIRED_OPS_LAB_DYNAMIC_SITE_ID]
    assert P2_RETIRED_PRIVATE_EXCEPTION_ID not in current[
        "architecture_exception_baseline"
    ]["exception_semantic_ids"]["active"]
    assert P3_RETIRED_CLASSIFICATION_LAB_EDGE_ID not in current[
        "architecture_exception_baseline"
    ]["exception_semantic_ids"]["active"]
    assert P3_RETIRED_PROVIDER_SITE_ID not in current[
        "architecture_exception_baseline"
    ]["cross_domain_site_semantic_ids"]["active"]
    assert {
        row["removal_campaign"]
        for row in rows
        if row["category"] == "schema_manifest_implementation_import"
    } == set()
    assert {
        row["removal_campaign"]
        for row in rows
        if row["category"] == "private_symbol_import" and row["target_package"] != "lab"
    } == set()
    assert {row["removal_campaign"] for row in rows} == {"S"}
    assert all(row["owner"].startswith("chronovisor.") for row in rows)
    assert all(row["owner"] != "chronovisor-architecture" for row in rows)
    assert all(
        row["deadline"] == architecture.CAMPAIGN_DEADLINES[row["removal_campaign"]]
        for row in rows
    )
    custom = {
        "owner": "chronovisor.custom.owner",
        "deadline": "2099-01-01",
        "removal_campaign": "S",
        "rationale": "Keep this reviewed metadata.",
    }
    assert (
        architecture._preserved_metadata(
            custom,
            {field: "fallback" for field in architecture.EXCEPTION_METADATA_FIELDS},
            legacy_owner="chronovisor-architecture",
        )
        == custom
    )


def test_lab_dispatch_compatibility_and_drift_are_protected(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility, seed, frozen, previous = _exception_inputs(current)
    dispatch = [row for row in compatibility if row["kind"] == "lab_dispatch"]
    assert len(dispatch) == 15
    assert {row["name"] for row in dispatch} == {
        "adoption-corpus",
        "classification-annif",
        "classification-calibrate",
        "classification-library-pilot",
        "classification-migrate",
        "classification-pilot",
        "classification-pilot-v2",
        "classification-profile-pilot",
        "classification-query2doc-pilot",
        "classification-query2doc-unseen",
        "librarian-burn",
        "local-model-eval",
        "model",
        "recall-challengers",
        "research-eval",
    }
    previous_id = dispatch[0]["semantic_id"]
    dispatch[0]["target"] += ".moved"
    dispatch[0]["semantic_id"] = architecture._compatibility_semantic_id(dispatch[0])

    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
        seed,
        frozen,
        previous,
    )
    assert violations["compatibility_contract_drift"] == {
        "unrecorded": [dispatch[0]["semantic_id"]],
        "stale": [previous_id],
        "identity_mismatches": [],
    }

    source, ledger, compatibility, seed, frozen, previous = _exception_inputs(current)
    duplicate = copy.deepcopy(ledger["compatibility_contracts"][0])
    ledger["compatibility_contracts"].append(duplicate)
    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
        seed,
        frozen,
        previous,
    )
    assert violations["duplicate_compatibility_ids"] == [duplicate["semantic_id"]]


def test_exception_artifacts_are_fresh_and_head_independent(
    architecture: ModuleType,
) -> None:
    recorded_seed = json.loads(
        (
            ROOT / "docs" / "refactoring" / "architecture-exception-baseline.json"
        ).read_text(encoding="utf-8")
    )
    recorded_ledger = json.loads(
        (ROOT / "docs" / "refactoring" / "architecture-exceptions.json").read_text(
            encoding="utf-8"
        )
    )
    built_ledger = architecture.build_architecture_exception_ledger(ROOT)
    built_seed = architecture.build_architecture_exception_baseline(ROOT)

    normalized_seed = _without_persisted_retirement_history(
        architecture,
        recorded_seed,
    )
    normalized_seed["cross_domain_site_semantic_ids"] = built_seed[
        "cross_domain_site_semantic_ids"
    ]
    normalized_seed["counts"]["active"]["cross_domain_sites"] = built_seed["counts"][
        "active"
    ]["cross_domain_sites"]
    assert built_seed == normalized_seed
    _assert_diagnostic_line_contract(recorded_ledger, built_ledger)
    assert "captured_from_head" not in recorded_seed
    assert "captured_from_head" not in recorded_ledger
    assert (
        architecture.FROZEN_EXCEPTION_SOURCE_HEAD
        == "d404a6b20d00e3bcd1d4cdb89edfa5a718c51833"
    )
    assert (
        recorded_seed["source_baseline_head"]
        == architecture.FROZEN_EXCEPTION_SOURCE_HEAD
    )
    assert recorded_ledger["source_baseline_head"] == (
        architecture.FROZEN_EXCEPTION_SOURCE_HEAD
    )


def test_diagnostic_line_contract_rejects_bool_and_path_drift() -> None:
    with pytest.raises(AssertionError):
        _assert_diagnostic_line_contract(
            {"items": [{"line": 1}]},
            {"items": [{"line": True}]},
        )
    with pytest.raises(AssertionError):
        _assert_diagnostic_line_contract(
            {"items": [{"line": 1}, {}]},
            {"items": [{}, {"line": 99}]},
        )


def test_frozen_source_head_rejects_coordinated_current_artifact_drift(
    architecture: ModuleType,
    current: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, ledger, compatibility, seed, frozen, _previous = _exception_inputs(current)
    previous = copy.deepcopy(seed)
    original_head = seed["source_baseline_head"]
    replacement_head = "0" * 40
    monkeypatch.setattr(
        architecture,
        "FROZEN_EXCEPTION_SOURCE_HEAD",
        replacement_head,
    )
    seed["source_baseline_head"] = replacement_head
    frozen["source_baseline_head"] = replacement_head
    ledger["source_baseline_head"] = replacement_head
    ledger["baseline_sha256"] = architecture._canonical_sha256(seed)

    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
        seed,
        frozen,
        previous,
    )

    assert violations["ledger_source_baseline_head_drift"] == []
    assert violations["seed_source_baseline_head_drift"] == []
    assert violations["previous_seed_source_baseline_head_drift"] == [original_head]
    assert violations["seed_source_baseline_head_history_drift"] == {
        "previous": original_head,
        "current": replacement_head,
    }


def test_compatibility_policy_requires_mixed_version_observation_and_rollback() -> None:
    adr = (
        ROOT
        / "docs"
        / "architecture"
        / "adr"
        / "0001-layering-dependency-and-compatibility.md"
    ).read_text(encoding="utf-8")
    policy = adr.lower()
    assert "mixed-version" in policy
    assert "observation" in policy
    assert "rollback" in policy
