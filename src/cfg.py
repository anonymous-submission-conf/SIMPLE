
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig

class CFG_class():
    def __init__(self):
        self.batch_size = 64
        self.nonGC_lr = 1e-4
        self.GC_lr = 1e-3
        self.lr = 3e-5
        self.num_epochs = 20
        self.eval_epoch = 0
        self.seed = 0
        self.weight_decay = 1e-8
        self.max_len = 128 #Difference in 128 to 256 is around 1GB 
        self.debug = False
        self.name = "exp_test"
#         self.contrastive_param = 0.1
        
        self.generation_cfg = {
                "max_new_tokens": 128,
#                 "min_new_tokens": 5,
#                 "num_beams": 8,
#                 "num_beam_groups": 4,
#                 "diversity_penalty": 1.0,
# #                 "top_p": 0.9,
                "top_k": 30,
                "penalty_alpha": 0.6
# #                 "temperature": 0.7,
#                 "do_sample": False,
#                 "repetition_penalty": 1.03,
#                 "no_repeat_ngram_size": 3,
        }

        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=False,
            bnb_4bit_quant_type="nf4",
#             bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
#             init_lora_weights="gaussian",
            task_type="CAUSAL_LM"
        )
    
    def log_config(self, log):
        log.info("Config details:")
        d = vars(self)
        for attr in d:
            if type(d[attr]) is dict:
                log.info(f"{attr}:")
                for subattr in d[attr]:
                    log.info(f"  {subattr}: {d[attr][subattr]}")
            else: 
                log.info(f"{attr}: {d[attr]}")
        
        
CFG = CFG_class()
