import torch
import os

# new addidtion 
import numpy as np
import pandas as pd



m = torch.load("saved_models/pruned_tiles_quantized/The_model_complet")
torch.set_printoptions(profile="full") # Pour afficher toutes les valeurs
#print(m)

#print(m.ctc)

#m.encoder.encoders[1].self_attn.linear_q.weight()

for name, module in m.named_modules():
        #if "coders." in name:
        #if "feed" in name:
        if isinstance(module, torch.nn.Linear):
                if not "ctc" in name and not "embed" in name:

                        print(name)
                        #           print('test/'+name)
                        nomfich="SavedModules_" + name
                        #           if name != " ":
                        #                            print("kk",name,"ll")
                        #                     nomfich=os.path.join("test/",name)
                        print(nomfich)
                        torch.save(module,nomfich)

                        #npmod=module.numpy()
                        #df=pd.DataFrame(npmod)
                        #df.to_csv(nomfich,index=False)
                

                        #print("module name:   ", module)
                        #print("*******************")

                        s=torch.load(nomfich)
                        v=s.weight
                        #print(v)
                        # not working ? 
                        # intversion=torch.int_repr(v)
                        
                        # instead 
                        nomfich2="SavedModules_" + name + ".txt"

                        np.savetxt(nomfich2,v.numpy(force=True))

                        #np.savetxt(nomfich,intversion.numpy())
                        #np.savetxt(nomfich,v.numpy())



# this works but save in int8 format
#s=torch.load("saved_models/pruned_tiles_quantized/SavedModules_encoder.encoders.4.self_attn.linear_k")
#v=s.weight()
#intversion=torch.int_repr(v)
#np.savetxt("t",intversion.numpy())