# 从 NVD 读取吧


import json
import os
from tqdm import tqdm


nvd_root_path = '../../NVD'
with open('./data/cve2desc.json', 'r') as f:
    cve2desc = json.load(f)  # load existing cve2desc

for f in os.listdir(nvd_root_path):
    if f.endswith('.json'):
        with open(os.path.join(nvd_root_path, f), 'r') as fin:
            data = json.load(fin)
            # print(data)
        # import pdb; pdb.set_trace()

        for item in tqdm(data['CVE_Items'], desc=f'loading {f}', total=len(data['CVE_Items'])):
            cve_id = item['cve']['CVE_data_meta']['ID']
            if cve_id in cve2desc:
                continue
            desc = item['cve']['description']['description_data'][0]['value']
            cve2desc[cve_id] = desc

    print(f'{f} loaded, {len(cve2desc)} cves')
    with open('./data/cve2desc.json', 'w') as fout:
        json.dump(cve2desc, fout, indent=4)



