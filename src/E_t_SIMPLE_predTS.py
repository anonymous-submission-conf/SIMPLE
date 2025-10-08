#Importing Libraries
import os
import time
import inspect
import sys
from tqdm.auto import tqdm, trange # Use tqdm.auto for best console/notebook compatibility
import torch
from torch.optim import AdamW
import torch.nn as nn
from transformers.optimization import get_linear_schedule_with_warmup
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from E_m_SIMPLE_predTS import SIMPLE, TARGEN
from torchvision import transforms
import E_m_SIMPLE_predTS as model_file
from torch.utils.data import DataLoader
from  E_d_SIMPLE_predTS import GoldDataset
from utils import setup_seed, send_to_device
from transformers import BartTokenizer
import logging
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
from rouge_score import rouge_scorer
from cfg import CFG
from transformers import logging as t_logging
t_logging.set_verbosity_error()
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

def prep_ref_files (test_dataloader, tkr):
    out_file = open('<output_filename>', 'w', encoding='utf-8')

    for test_data in tqdm(test_dataloader, desc="Preparing Reference Files"):
        for i in range(len(test_data['input_ids'])):
            #Sarcasm Explanation
            out_pad = test_data['target_ids'][i].masked_fill(test_data['target_ids'][i] == -100, 0)
            out = tkr.decode(out_pad, skip_special_tokens=True)

            #Writing to files
            out_file.write(out + '\n')

    #Flush and close files
    for txt in [out_file]:
        txt.flush()
        txt.close()

#Functions to evaluate text similarity metrics
def eval_metrics(gen_text_file, ref_text_file):
    # calculate metrices
    with open(ref_text_file, 'r', encoding='utf-8') as f:
        ref_list = f.readlines()
    with open(gen_text_file, 'r', encoding='utf-8') as f:
        gen_list = f.readlines()

    ref_dict = {}
    gen_dict = {}
    
    for i in trange(len(gen_list), desc="Processing for Metrics"):
        ref = ' '.join(nltk.word_tokenize(ref_list[i].lower()))
        gen = ' '.join(nltk.word_tokenize(gen_list[i].lower()))

        ref_dict[i] = [ref]
        gen_dict[i] = [gen]

    scores = eval_metrics_helper(gen_dict, ref_dict)

    for k in scores:
        scores[k] = f"{scores[k] * 100:.3f}"
    return scores

def eval_metrics_helper(gen_dict, ref_dict):
    gen_list=[]
    ref_list=[]
    print("Hypothesis list:")
    for i in gen_dict.values():
        gen_list.append(i)
        print(i, end=" ")
    for i in ref_dict.values():
        ref_list.append(i)
    
    scores_dict = {}
    
    b = Bleu()
    score, _ = b.compute_score(gts=ref_dict, res=gen_dict)
    b1, b2, b3, b4 = score

    r = Rouge()
    score, _ = r.compute_score(gts=ref_dict, res=gen_dict)

    rl = score
    count=0
    rouge1=0
    rouge2=0
    meteor=0
    for reference, hypothesis in zip(ref_list, gen_list):
        reference=str(reference).strip('[]\'')
        hypothesis=str(hypothesis).strip('[]\'')

        count += 1
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        meteor += nltk.translate.meteor_score.meteor_score([nltk.word_tokenize(reference)], nltk.word_tokenize(hypothesis))
        scores = scorer.score(reference, hypothesis)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
    rouge1 = rouge1 / count
    rouge2 = rouge2 / count
    meteor = meteor / count


    scores_dict['Bleu_1'] = b1
    scores_dict['Bleu_2'] = b2
    scores_dict['Bleu_3'] = b3
    scores_dict['Bleu_4'] = b4
    scores_dict['Rouge_L'] = rl
    scores_dict['Rouge1'] = rouge1
    scores_dict['Rouge2'] = rouge2
    scores_dict['METEOR'] = meteor
    return scores_dict

#Function to create and return dataloaders
def load_data(tkr):    
    
    #Defining transform to apply to images
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    #Train
    train_dataset = GoldDataset (
        data_file= 'train_data.tsv', 
        D_file='D_train.pkl', 
        O_file = 'O_train.pkl', 
        path_to_images = 'images', 
        tokenizer = tkr, 
        image_transform = image_transform, 
        cfg=CFG
    )
    
    #Val
    val_dataset = GoldDataset (
        data_file= 'val_data.tsv', 
        D_file='D_val.pkl', 
        O_file = 'O_val.pkl', 
        path_to_images = 'images', 
        tokenizer = tkr, 
        image_transform = image_transform, 
        cfg=CFG
    )
    
    #Test
    test_dataset = GoldDataset (
        data_file= 'test_data.tsv',
        D_file='D_test.pkl', 
        O_file = 'O_test.pkl', 
        path_to_images = 'images', 
        tokenizer = tkr, 
        image_transform = image_transform, 
        cfg=CFG
    )

    train_dataloader = DataLoader(train_dataset, batch_size=CFG.batch_size, num_workers=8, shuffle=True)
    eval_dataloader = DataLoader(val_dataset, batch_size=CFG.batch_size, num_workers=8, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=CFG.batch_size, num_workers=8, shuffle=False)

    return train_dataloader, eval_dataloader, test_dataloader

#Function to run for validation loop
def eval_net(model, loader, device):
    total_ppl = 0.0
    model.eval() #Only inference

    with torch.no_grad():
        # Create a tqdm progress bar for the validation loader
        eval_pbar = tqdm(loader, desc="Validating", leave=False)
        for idx, batch in enumerate(eval_pbar):
            batch = send_to_device(batch, device)
            _, ppl = model(**batch,mode='eval')
            
            # Update the running total for PPL
            total_ppl += ppl.item()
            
            # Update the progress bar with the running average PPL
            eval_pbar.set_postfix(avg_ppl=f"{(total_ppl / (idx + 1)):.4f}")

    # Final calculation of the mean PPL
    ppl_mean = total_ppl / len(loader)

    return ppl_mean

def target_incorporation(caption, target_of_sarcasm, tkr, device):
        concat = [i + ' </s> ' + j for i, j in zip(caption, target_of_sarcasm)]

        encoded_dict = tkr(
            concat,
            max_length=CFG.max_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
            add_prefix_space=True
        )

        concat_ids = encoded_dict['input_ids'].to(device)
        concat_attn_mask = encoded_dict['attention_mask'].to(device)

        return concat_ids, concat_attn_mask

#Function to generate explanations on test set
def gen_net(ep, model, model_ts, loader, device):
    
    #All explanations are written (in order) to gen_file_name
    gen_file_name = "<gen_output_filename.txt>"
    gen_file = open(gen_file_name, 'w', encoding='utf-8') 
    
    model.eval()
    with torch.no_grad():
        gen_pbar = tqdm(loader, desc=f"Generating (Epoch {ep+1})", leave=False)
        for batch in gen_pbar:
            batch = send_to_device(batch, device)
            
            #Generate TS
            _, ts = model_ts(**batch)
                
            #Incorporate generated TS with caption
            input_ids, attention_mask = target_incorporation(batch["caption"], ts, model.tkr, device)
            
            #Generate explanation and backprop
            _, exp = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                target_ids = batch["target_ids"],
                input_image = batch["input_image"],
#                 is_sarcastic = batch["is_sarcastic"],
                mode="gen"
            )
           
            gen_file.write('\n'.join(exp) + '\n')

    gen_file.flush()
    gen_file.close()
    
    #Evaluate metrics on generated explanations
    scores = eval_metrics(gen_file_name, '<output_filename.txt>')

    return scores


def load_model(model, epoch):
    state_dict = torch.load('state_dict_path', weights_only=True)
    model.load_state_dict(state_dict)
    return model


def save_model(model, epoch):
    torch.save(model.state_dict(), "<save_dir_path>")

def run_stage(model, model_ts, lr_sche, opt,
              train_loader, eval_loader, test_loader,
              device, log):
    max_epoch = int(CFG.num_epochs)
    scores = []
    imp_vals = []

    epoch_pbar = trange(max_epoch, desc="Training Progress")
    for epoch in epoch_pbar:
        model.train()
        
        total_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epoch} [Train]", leave=False)
        for batch in train_pbar:
            opt.zero_grad()

            batch = send_to_device(batch, device)
            
            #Generate TS
            with torch.no_grad():
                _, ts = model_ts(**batch)
                
            #Incorporate generated TS with caption
            input_ids, attention_mask = target_incorporation(batch["caption"], ts, model.tkr, device)
            
            #Generate explanation and backprop
            contrastive_loss, ce_loss = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                target_ids = batch["target_ids"],
                input_image = batch["input_image"],
#                 is_sarcastic = batch["is_sarcastic"],
            )
            
            contrastive_param = min(0.5, (epoch+1)/20.0)
            loss = ce_loss + (contrastive_param * contrastive_loss)
            
            loss.backward()
            opt.step()
            lr_sche.step() # Step scheduler after each batch

            # Update running loss total
            total_loss += loss.item()
            
            # Update the progress bar with current loss info
            train_pbar.set_postfix(
                loss=f"{loss.item():.4f}", 
                avg_loss=f"{total_loss / (train_pbar.n + 1):.4f}"
            )

        # Log epoch-level stats after the inner loop is complete
        avg_train_loss = total_loss / len(train_loader)
        log.info(f"Epoch {epoch + 1} finished. Average Training Loss: {avg_train_loss:.4f}")
        log.info(f'Scores for epoch {epoch + 1}: {score}')
        scores.append(score)

        imp_vals.append({})
    return scores, imp_vals

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_seed(int(CFG.seed))
   
    #Set up logging
    logging.basicConfig(filename="<log_filename>", filemode = "a",level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    global log
    log = logging.getLogger()
    
    to_log = "\n\n\n\n\n\n\n[EVENMORE] Caption + predicted TS (epoch 18 in E_SIMPLE_TARGEN) -> Encoder -> FULL SF -> Decoder -> Exp, Checkpoints saved (TS set to 'No target present' for NS samples);"
    log.info(to_log)

    tkr = BartTokenizer.from_pretrained("facebook/bart-base")
    
    model = SIMPLE(tkr, device).to(device)
    model_ts = TARGEN(tkr, device).to(device)
    torch.compile(model)

    train_dataloader, eval_dataloader, test_dataloader = load_data(tkr)
    
    optimizer = AdamW(model.parameters(), lr=3e-5)

    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=int(CFG.num_epochs) * len(train_dataloader))
    
    prep_ref_files(test_dataloader, tkr)

    #Log source code of model file
    log.info(inspect.getsource(model_file))

    log.info("Starting training...")

    #Log device
    log.info(f'Found device: {device}')
    
    #Log config
    CFG.log_config(log)

    #Log sizes of dataloaders
    log.info(f"train data: {CFG.batch_size * len(train_dataloader)}")
    log.info(f"eval data: {CFG.batch_size * len(eval_dataloader)}")
    log.info(f"test data: {CFG.batch_size * len(test_dataloader)}")

    scores, imp_vals = run_stage(model=model, model_ts=model_ts, lr_sche=lr_scheduler, opt=optimizer,
                       train_loader=train_dataloader, eval_loader=eval_dataloader, 
                       test_loader=test_dataloader,device=device, log=log)

    #Logging all scores and other important values for reference
    log.info(to_log)
    for idx, score in enumerate(scores):
        log.info(f"{'-' * 20} Epoch {idx + 1} {'-' * 20}")
        
        for key in score.keys():
            log.info(f'{key}: {score[key]}')
        
        for key in imp_vals[idx].keys():
            log.info(f'{key}: {imp_vals[idx][key]}')


if __name__ == '__main__':
    main()