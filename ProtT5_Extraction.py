import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import numpy as np
import pandas as pd

def extraction_pLMs(seqfile):
    df = pd.read_csv(seqfile, header=None)
    names=df[df.columns[0]].values.tolist()
    proteins = df[df.columns[1]].values.tolist()
    proteins = [s[0:2500] for s in proteins] #if gpu memory is not enough ,the sequence should be truncated
    proteins = [' '.join(s.replace(' ', '')) for s in proteins]
    proteins = [re.sub(r"[UZOB]", "X", sequence) for sequence in proteins]

    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    i=0
    dlist=zip(names,proteins)
    for name,item in dlist:
        print(i)
        print(len(item))
        i+=1
        ids = tokenizer.batch_encode_plus([item], add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)

        embedding = embedding.last_hidden_state.cpu().numpy()

        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]

            with open("midData/ProtTran/"+name, 'w') as f:
                np.savetxt(f, seq_emd, delimiter=',', fmt='%s')
            #print(seq_emd.shape)

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    extraction_pLMs('Dataset/Non-AFP9493.seq') #AFP920.seq Non-AFP9493.seq
