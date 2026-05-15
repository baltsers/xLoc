# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup, 
                          T5ForConditionalGeneration, RobertaTokenizer)
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import datasets
from sklearn.model_selection import train_test_split
import pickle
import json
from fuzzywuzzy import fuzz

cpu_cont = 16
logger = logging.getLogger(__name__)

PAD, UNK, EOS, SOS = '<PAD>', '<UNK>', '<EOS>', '<SOS>'
class Vocab(object):
    def __init__(self, args):
        self.args = args
        self.vocab = dict()
        self.re_vocab = dict()
        self.__special__()

    def find(self, sub_token):
        return self.vocab.get(sub_token, self.unk_index)

    def has_token(self, token):
        if token in self.vocab:
            return True
        else:
            return False

    def has_idx(self, idx):
        if idx in self.re_vocab:
            return True
        else:
            return False

    def __special__(self):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.vocab[PAD] = self.pad_index
        self.vocab[UNK] = self.unk_index
        self.vocab[EOS] = self.eos_index
        self.vocab[SOS] = self.sos_index
        self.special_index = [self.pad_index, self.eos_index, self.sos_index, self.unk_index]

    def re_find(self, idx):
        return self.re_vocab.get(idx, UNK)

    def __len__(self):
        return len(self.vocab)

class CTTextVocab(Vocab):
    def __init__(self, args):
        '''
        Code Transformer Vocab
        '''
        super().__init__(args)
        self.type = 'CT'
        print('Get CT Vocab')
        #self.dataset_dir = os.path.join('./data', args.dataset)
        self.dataset_dir = './'
        with open(os.path.join(self.dataset_dir, 'ct_vocab.json'), 'r') as f:
            vocab_dict = json.load(f)
        for key, _ in vocab_dict.items():
            self.vocab[key] = len(self.vocab)
        print('{} Vocab length == {}'.format(self.type, len(self.vocab)))
        for key, value in self.vocab.items():
            self.re_vocab[value] = key


def content_process(content, vocab, args):
    max_code_length = args.encoder_block_size
    content_ = [vocab.find(token) for token in content]
    content_ = content_[:max_code_length] + [vocab.pad_index] * abs(max_code_length - len(content))
    content_ = content_[:max_code_length]
    # content_: max_code_length
    content_mask_ = [1 for _ in content][:max_code_length] + [0] * abs(
        max_code_length - len(content))
    return torch.LongTensor([content_]), torch.LongTensor([content_mask_])

def target_process(content, vocab, args):
    max_code_length = args.decoder_block_size
    content_ = [vocab.find(token) for token in content]
    content_ = content_[:max_code_length] + [vocab.pad_index] * abs(max_code_length - len(content))
    content_ = content_[:max_code_length]
    # content_: max_code_length
    content_mask_ = [1 for _ in content][:max_code_length] + [0] * abs(
        max_code_length - len(content))
    return torch.LongTensor([content_]), torch.LongTensor([content_mask_])


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,file_name,input_ids,label,decoder_input_ids,relative_call_position):
        self.file_name=file_name
        self.input_ids = input_ids
        self.label=label
        self.decoder_input_ids = decoder_input_ids
        self.relative_call_position = relative_call_position
        

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, train_data=None, val_data=None, file_type="train"):
        if file_type == "train":
            file_path = args.train_data_file
        #    sources = train_data["source"].tolist()
        #    labels = train_data["target"].tolist()
        elif file_type == "eval":
            file_path = args.eval_data_file
        #    sources = val_data["source"].tolist()
        #    labels = val_data["target"].tolist()
        elif file_type == "test":
            file_path = args.test_data_file
        #    data = datasets.load_dataset("MickyMike/cvefixes_bigvul", split="test")
        #    sources = data["source"]
        #    labels = data["target"]
        
        
        self.examples = []
        
        #df = pd.read_csv(file_path)
        #sources = df["source"].tolist()
        #labels = df["target"].tolist()
        print(f"Start loading {file_path}")
        f=open(file_path,'rb')
        objs=pickle.load(f)
        print("Loading success")
        import random
        random.seed(1000)
        random.shuffle(objs)
        objs = objs[:]
        f.close()
        file_names=[]
        source_tokens=[]
        target_tokens=[]
        relative_call_locations=[]
        ab_locations=[]
        vocab = {}
        print(f"load {file_path} done")
        for i,obj in enumerate(objs):
            file_names.append(obj['file_name'])
            source_tokens.append(obj['source_tokens'])
            for token in obj['source_tokens']:
                if not token in vocab and len(vocab)<10000:
                    vocab[token] = len(vocab)
            target_tokens.append(obj['target_tokens'])
            for j,token in enumerate(obj['target_tokens']):
                objs[i]['target_tokens'][j]=str(objs[i]['target_tokens'][j])
                if not token in vocab:
                    vocab[token] = len(vocab)
            try:
                if args.rela_hyper == 1.0:
                    relative_call_locations.append(obj['relative_call_position'])
                else:
                    multiplied_dict = {k: v * args.rela_hyper for k, v in obj['relative_call_position'].items()}
                    relative_call_locations.append(multiplied_dict)
                
            except:
                relative_call_locations.append({})
            
            try:
                ab_dict={}
                for kkk,token in enumerate(obj['source_tokens']):
                    token_line = obj['source_token_lines'][kkk]
                    if token_line > 0:
                        ab_dict[(0,kkk)] = -obj['absolute_position'][token_line]
                    else:
                        ab_dict[(0,kkk)] = 0
            
            
                ab_locations.append(ab_dict)
            except:
                ab_locations.append({})
                
        print("rela and abs position done")
        if os.path.exists('ct_vocab.json'):
            f=open('ct_vocab.json','r')
            vocab=json.load(f)
            f.close()
        else:
            f=open('ct_vocab.json','w')
            json.dump(vocab,f)
            f.close()
        self.vocab = CTTextVocab(args)    
        self.source_tokens = source_tokens
        self.target_tokens = target_tokens
        self.relative_call_locations = relative_call_locations
        self.ab_locations = ab_locations
        self.args=args
        self.file_names=file_names
        '''
        for i in tqdm(range(len(source_tokens))):
            #try:
            source_ids, source_mask = content_process(source_tokens[i], vocab, args)
            target_ids, target_mask = target_process(target_tokens[i], vocab, args)
            relative_call_position = torch.ones(args.encoder_block_size,args.encoder_block_size, dtype=torch.long)
            relative_call_position = relative_call_position * (-args.encoder_block_size)
            for position in relative_call_locations[i]:
                try:
                    relative_call_position[position[0]][position[1]] = relative_call_locations[i][position]
                except:
                    break
            self.examples.append(InputFeatures(file_names[i], source_ids, target_ids, target_ids,relative_call_position))
            #except:
            #    pass
        '''
        
        if file_type == "train":
            self.source_tokens = self.source_tokens[:int(0.9*len(self.source_tokens))]
            self.target_tokens = self.target_tokens[:int(0.9*len(self.target_tokens))]
            self.relative_call_locations = self.relative_call_locations[:int(0.9*len(self.relative_call_locations))]
            self.ab_locations = self.ab_locations[:int(0.9*len(self.ab_locations))]
            self.file_names = self.file_names[:int(0.9*len(self.file_names))]

            '''
            for example in self.examples[:3]:
                logger.info("*** Example ***")
                logger.info("label: {}".format(example.label))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                logger.info("decoder_input_ids: {}".format(' '.join(map(str, example.decoder_input_ids))))
            '''
        elif file_type == "eval":
            #self.examples = self.examples[int(0.9*len(self.examples)):]
            pass
            '''
            self.source_tokens = self.source_tokens[int(0.9*len(self.source_tokens)):]
            self.target_tokens = self.target_tokens[int(0.9*len(self.target_tokens)):]
            self.relative_call_locations = self.relative_call_locations[int(0.9*len(self.relative_call_locations)):]
            self.file_names = self.file_names[int(0.9*len(self.file_names)):]
            '''
        
        print(len(self.source_tokens))
        
    def __len__(self):
        return len(self.source_tokens)

    def __getitem__(self, i):
        #import pdb
        #pdb.set_trace()
        source_ids, source_mask = content_process(self.source_tokens[i], self.vocab, self.args)
        target_ids, target_mask = target_process(self.target_tokens[i], self.vocab, self.args)
        relative_call_position = torch.ones(self.args.encoder_block_size,self.args.encoder_block_size, dtype=torch.long)
        relative_call_position = relative_call_position * (-self.args.encoder_block_size)
        for position in self.relative_call_locations[i]:
            #import pdb
            #pdb.set_trace()
            try:
                relative_call_position[position[0]+1][position[1]] = self.relative_call_locations[i][position]
            except:
                break
        
        for position in self.ab_locations[i]:
            try:
                '''
                if relative_call_position[position[0]][position[1]] == -1024:
                    relative_call_position[position[0]][position[1]] = self.ab_locations[i][position]
                else:
                    relative_call_position[position[0]][position[1]]+= self.ab_locations[i][position]
                '''
                
                relative_call_position[position[0]][position[1]]= self.ab_locations[i][position]
            except:
                break
        this_example = InputFeatures(self.file_names[i], source_ids, target_ids, target_ids,relative_call_position)
        return this_example.file_name, this_example.input_ids, this_example.input_ids.ne(0), this_example.label, this_example.decoder_input_ids, this_example.relative_call_position
        
        #return self.examples[i].file_name, self.examples[i].input_ids, self.examples[i].input_ids.ne(0), self.examples[i].label, self.examples[i].decoder_input_ids, self.examples[i].relative_call_position


def convert_examples_to_features(source, label, tokenizer, args):
    # encode - subword tokenize
    source_ids = tokenizer.encode(source, truncation=True, max_length=args.encoder_block_size, padding='max_length', return_tensors='pt')
    decoder_input_ids = tokenizer.encode(label, truncation=True, max_length=args.decoder_block_size, padding='max_length', return_tensors='pt')
    label = tokenizer.encode(label, truncation=True, max_length=args.decoder_block_size, padding='max_length', return_tensors='pt')
    return InputFeatures(source_ids, label, decoder_input_ids)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)
    
    args.max_steps = args.epochs * len(train_dataloader)

    # evaluate model per epoch
    args.save_steps = len(train_dataloader) * 1
   
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    
    # multi-gpu training
    #if args.n_gpu > 1:
    #    model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_loss = 100

    writer_path = "tb/codet5_training_loss"
    writer = SummaryWriter(writer_path)

    model.zero_grad()

    for idx in range(args.epochs):
    #for idx in range(1):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0

        for step, batch in enumerate(bar):
            (file_names, input_ids, attention_mask, labels, decoder_input_ids, relative_call_position) = [x for x in batch]
            input_ids = input_ids.squeeze(1).to(args.device)
            attention_mask = attention_mask.squeeze(1).to(args.device)
            labels = labels.squeeze(1).to(args.device)
            decoder_input_ids = decoder_input_ids.squeeze(1).to(args.device)
            relative_call_position = relative_call_position.squeeze(1).to(args.device)
            model.train()
            # the forward function automatically creates the correct decoder_input_ids
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, relative_call_position=relative_call_position).loss
            #loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if global_step % args.save_steps == 0:

                    # placeholder of evaluation
                    #eval_loss = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)    
                    # Save model checkpoint
                    #if eval_loss < best_loss:
                        #best_loss = eval_loss
                        #logger.info("  "+"*"*20)  
                        #logger.info("  Best Loss:%s",round(best_loss,4))
                        #logger.info("  "+"*"*20)                          
                    checkpoint_prefix = 'checkpoint-best-loss'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)                        
                    model_to_save = model.module if hasattr(model,'module') else model
                    output_dir = os.path.join(output_dir, '{}'.format(args.model_name)) 
                    torch.save(model_to_save.state_dict(), output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)

def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens

def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    
    eval_loss, num = 0, 0
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch in bar:
        #(input_ids, attention_mask, labels, decoder_input_ids) = [x.squeeze(1).to(args.device) for x in batch]
        (file_names, input_ids, attention_mask, labels, decoder_input_ids, relative_call_position) = [x for x in batch]
        input_ids = input_ids.squeeze(1).to(args.device)
        attention_mask = attention_mask.squeeze(1).to(args.device)
        labels = labels.squeeze(1).to(args.device)
        decoder_input_ids = decoder_input_ids.squeeze(1).to(args.device)
        relative_call_position = relative_call_position.squeeze(1).to(args.device)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, relative_call_position=relative_call_position).loss
        #loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        if args.n_gpu > 1:
            loss = loss.mean()
        eval_loss += loss.item()
        num += 1
    eval_loss = round(eval_loss/num,5)
    model.train()
    logger.info("***** Eval results *****")
    logger.info(f"Evaluation Loss: {str(eval_loss)}")
    return eval_loss

def test(args, model, tokenizer, test_dataset, best_threshold=0.5):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Test!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    model.eval()
    file_namess = []
    accuracy = []
    raw_predictions = []
    ground_truths = []
    correct_prediction = ""
    vocab = CTTextVocab(args)
    bar = tqdm(test_dataloader, total=len(test_dataloader))
    iii=0
    for batch in bar:
        #iii+=1
        #if iii>10:
        #    break
        correct_pred = False
        #(input_ids, attention_mask, labels, decoder_input_ids)=[x.squeeze(1).to(args.device) for x in batch]
        (file_names, input_ids, attention_mask, labels, decoder_input_ids, relative_call_position) = [x for x in batch]
        file_namess.extend(file_names)
        input_ids = input_ids.squeeze(1).to(args.device)
        attention_mask = attention_mask.squeeze(1).to(args.device)
        labels = labels.squeeze(1).to(args.device)
        decoder_input_ids = decoder_input_ids.squeeze(1).to(args.device)
        relative_call_position = relative_call_position.squeeze(1).to(args.device)
        with torch.no_grad():
            beam_outputs = model.generate(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          do_sample=False, # disable sampling to test if batching affects output
                                          num_beams=args.num_beams,
                                          num_return_sequences=args.num_beams,
                                          max_length=args.decoder_block_size,relative_call_position=relative_call_position)
        beam_outputs = beam_outputs.detach().cpu().tolist()
        decoder_input_ids = decoder_input_ids.detach().cpu().tolist()
        
        for single_output in beam_outputs:
            # pred
            prediction = ""
            for kkk,idx in enumerate(single_output):
                if idx>3:
                    prediction = prediction + vocab.re_find(idx)+" "
                '''
                breakflag = True
                for nnn in range(kkk,len(single_output)):
                    if single_output[nnn]!=0:
                        breakflag = False
                        break
                if breakflag:
                    break
                '''
            #prediction = tokenizer.decode(single_output, skip_special_tokens=False)
            #prediction = clean_tokens(prediction)
            # truth
            ground_truth = ""
            for kkk,idx in enumerate(decoder_input_ids[0]):
                if idx>3:
                    ground_truth = ground_truth + vocab.re_find(idx)+" "
                '''
                breakflag = True
                for nnn in range(kkk,len(single_output)):            
                    if single_output[nnn]!=0:
                        breakflag = False
                        break
                if breakflag:
                    break
                '''
            #ground_truth = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=False)
            #ground_truth = clean_tokens(ground_truth)
            #if prediction in ground_truth or ground_truth in prediction:
            print("pred: "+prediction)
            if fuzz.partial_ratio(prediction,ground_truth)>90:
                correct_prediction = prediction
                correct_pred = True
                break
        if correct_pred:
            raw_predictions.append(correct_prediction)
            accuracy.append(1)
        else:
            # if not correct, use the first output in the beam as the raw prediction
            raw_pred = tokenizer.decode(beam_outputs[0], skip_special_tokens=False)
            raw_pred = clean_tokens(raw_pred)
            raw_predictions.append(prediction)
            accuracy.append(0)
        ground_truths.append(ground_truth)
        #print('\n'+ground_truth+"\n"+prediction)
        print("gth: "+ground_truth)
        #print(round(sum(accuracy) / len(accuracy), 4))
        print("!@#$====$#@!")
        nb_eval_steps += 1
    # calculate accuracy
    test_result = round(sum(accuracy) / len(accuracy), 4)
    logger.info("***** Test results *****")
    logger.info(f"Test Accuracy: {str(test_result)}")

    # write prediction to file
    df = pd.DataFrame({"file_names": [], "raw_predictions": [], "ground_truth": [], "correctly_predicted": []})
    df["file_names"] = file_namess
    df["raw_predictions"] = raw_predictions
    df["correctly_predicted"] = accuracy
    df["ground_truth"] = ground_truths
    df.to_csv("./raw_predictions/VulRepair_raw_preds_call_position_beam"+str(len(beam_outputs))+".csv")

def main():
    parser = argparse.ArgumentParser()
    # Params
    parser.add_argument("--train_data_file", default="lang_loc_train_bn.pkl", type=str, required=False,
                                    help="The input training data file (a csv file).")#lang_loc_train_bn.pkl
    parser.add_argument("--eval_data_file", default="lang_loc_test_bn2.pkl", type=str,
                                    help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default="lang_loc_test_bn2.pkl", type=str,
                                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--output_dir", default="./saved_models", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="t5", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--encoder_block_size", default=1024, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--decoder_block_size", default=256, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--num_beams", default=3, type=int,
                        help="Beam size to use when decoding.")                          
    parser.add_argument("--model_name", default="model_call_position2_fc_as.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--checkpoint_model_name", default="non_domain_model.bin", type=str,
                            help="Checkpoint model name.")
    parser.add_argument("--model_name_or_path", default="Salesforce/codet5-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="Salesforce/codet5-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=20,
                        help="training epochs")
    parser.add_argument('--rela_hyper', type=float, default=1.0,
                                    help="hyperparameter for the cross-language api distance")
    args = parser.parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)
    # Set seed
    set_seed(args)

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.add_tokens(["<S2SV_StartBug>", "<S2SV_EndBug>", "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>"])
    if(args.model_name_or_path=="Salesforce/codet5-base"):
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path) 
    else:
        model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
    model.resize_token_embeddings(len(tokenizer))

    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        if args.model_name_or_path !="Salesforce/codet5-base":
            checkpoint_prefix = f'checkpoint-best-loss/{args.model_name_or_path}'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        #train_data_whole = datasets.load_dataset("MickyMike/cvefixes_bigvul", split="train")
        #df = pd.DataFrame({"source": train_data_whole["source"], "target": train_data_whole["target"]})
        #train_data, val_data = train_test_split(df, test_size=0.1238, random_state=42)
        #train_dataset = TextDataset(tokenizer, args, train_data, val_data, file_type='train')
        #eval_dataset = TextDataset(tokenizer, args, train_data, val_data, file_type='eval')
        eval_dataset = TextDataset(tokenizer, args, file_type='eval')
        '''
        for epo in range(args.epochs):
            logger.info("=============================")
            logger.info("  Current epoch = %d", epo)
            logger.info("=============================")
            files=os.listdir('./assig_train/')
            current_file = 0
            total_files = len(files)
            for file in files:
                current_file+=1
                logger.info("  Current epoch = %d, file: %d/%d", epo,current_file,total_files)
                args.train_data_file = os.path.join('./assig_train/', file)
                train_dataset = TextDataset(tokenizer, args, file_type='train')
                train(args, train_dataset, model, tokenizer, eval_dataset)
        '''
        train_dataset = TextDataset(tokenizer, args, file_type='train')
        train(args, train_dataset, model, tokenizer, eval_dataset)
    # Evaluation
    results = {}  
    if args.do_test:
        if args.model_name_or_path != "MickyMike/VulRepair":
            checkpoint_prefix = f'checkpoint-best-loss/{args.model_name}'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        test(args, model, tokenizer, test_dataset, best_threshold=0.5)
    return results

if __name__ == "__main__":
    main()
