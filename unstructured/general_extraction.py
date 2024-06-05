import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'utils')
import os
from unstructured.partition.auto import partition

import time
import numpy as np
from evaluation import *
from preprocess import *

#some warning in scikit learn
#import warnings
#warnings.filterwarnings('ignore')


#intitialize variables
fin_tp,lr_tp,gt_tp,m_tp,sa_tp,p_tp,o_tp,fin_fp,lr_fp,gt_fp,m_fp,sa_fp,p_fp,o_fp = (0,0,0,0,0,0,0,0,0,0,0,0,0,0)
fin_m,lr_m,gt_m,m_m,sa_m,p_m,o_m,fin_n,lr_n,gt_n,m_n,sa_n,p_n,o_n = (0,0,0,0,0,0,0,0,0,0,0,0,0,0)
fin_c,lr_c,gt_c,m_c,sa_c,p_c,o_c = (0,0,0,0,0,0,0) # for counting files
undetected_pdfs = 0
empty_json = 0
gt_collated=''
et_collated=''
t_parse = []

#directory of files
json_list = os.listdir("train/annotations")


for file in json_list:
  filepath = os.path.join("train/annotations",file)
  #load ground truth to df
  df1,doc_type = json_to_df(filepath) #type of document-finanacial,scientific etc
  
  if doc_type == 0:
    empty_json +=1
    continue
  #get a dataframe in separate token format
  df2 =  separate_token_format_gt(df1)

   ## PDF Parsing
  t1 = time.time()
  
  filepath = "train/pdfs/" + os.path.splitext(file)[0] + '.pdf'
  text = ''
  #unstructured
  try:
    elements = partition(filepath)
    text = "\n\n".join([str(el) for el in elements])
  except:
    undetected_pdfs +=1
    continue
  
  
  
  t2 = time.time()
  t_parse.append(t2-t1)
  
  if text.split() == []:
    undetected_pdfs +=1
    continue  
  
  #Extracted text as a dataframe in separate token format
  df3 = separate_token_format_et(text)

## Token wise evaluation
#-----------------------------#

# levenshtein similarity matrix
  
  matrix=similarity_matrix(df3['tokens_et'],df2['tokens_gt'])

# get tp,fp,m,n
  tp,fp,m,n = compute_tpfp(matrix)
  
  #assinging values based on doc type
  if doc_type == 'financial_reports':
    fin_tp +=tp
    fin_fp +=fp
    fin_m +=m
    fin_n +=n
    fin_c +=1
  elif doc_type == 'government_tenders':
    gt_tp +=tp
    gt_fp +=fp
    gt_m +=m
    gt_n +=n
    gt_c +=1
  elif doc_type == 'laws_and_regulations':
    lr_tp +=tp
    lr_fp +=fp
    lr_m +=m
    lr_n +=n
    lr_c +=1
    
  elif doc_type ==  'manuals':
    m_tp +=tp
    m_fp +=fp
    m_m +=m
    m_n +=n
    m_c +=1
  
  elif doc_type ==  'scientific_articles':
    sa_tp +=tp
    sa_fp +=fp
    sa_m +=m
    sa_n +=n
    sa_c +=1
    
  else:                 # patents
    p_tp +=tp
    p_fp +=fp
    p_m +=m
    p_n +=n
    p_c +=1    
  #overall tp,fp,m,n
  o_tp +=tp
  o_fp +=fp
  o_m +=m
  o_n +=n
  o_c +=1
  #get ground truth in collated format
  gt_text = collated_text(df1)
  
  #save collated text
  gt_collated += gt_text + '\n'
  et_collated += text +'\n'  
    


score_fin = calculate_scores(fin_tp,fin_m,fin_n)
score_gt = calculate_scores(gt_tp,gt_m,gt_n)
score_lr = calculate_scores(lr_tp,lr_m,lr_n)
score_sa = calculate_scores(sa_tp,sa_m,sa_n)
score_m = calculate_scores(m_tp,m_m,m_n)
score_o = calculate_scores(o_tp,o_m,o_n)
score_p = calculate_scores(p_tp,p_m,p_n)

with open('unstructured/separate_token_score.txt', 'w') as f:
  f.write('Financial Documents'+'\n' + score_fin + '\n' + 'Number of documents:' + repr(fin_c) + '\n\n')
  f.write('Government Tenders' + '\n' + score_gt + '\n' + 'Number of documents:' + repr(gt_c) + '\n\n')
  f.write('Law and Regulations' + '\n' + score_lr + '\n' +  'Number of documents:' + repr(lr_c) + '\n\n')
  f.write('Scientific Articles' + '\n' +score_sa + '\n' + 'Number of documents:' + repr(sa_c) + '\n\n')
  f.write('Manuals' + '\n' + (score_m) + '\n' + 'Number of documents:' + repr(m_c) + '\n\n')
  f.write('Patents' + '\n' + (score_p) + '\n' + 'Number of documents:' + repr(p_c) + '\n\n')
  f.write('Overall Performance' + '\n' + (score_o) + '\n' + 'Number of documents:' + repr(o_c) + '\n\n'
          +'undetected documents:' + repr(undetected_pdfs)+'\n\n')
  f.write('Average time taken to parse 1 file: '+ repr(np.mean(t_parse)))



##Collated format evaluation
#--------------------------#

score_coll = calculate_score_collated(et_collated,gt_collated)
with open('unstructured/collated_token_score.txt', 'w') as f:
  f.write(score_coll)