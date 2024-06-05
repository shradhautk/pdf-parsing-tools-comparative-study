import numpy as np
import rapidfuzz
import textdistance


def levenshtein_score(s1,s2):
  return rapidfuzz.distance.Levenshtein.normalized_similarity(s1,s2)

def similarity_matrix(token_et,token_gt):
  """
  Args:
    token_et: dataframe containing token_et
    token_gt: dataframe containing token_gt
  Return:
    matrix_lv: similarity matrix(extracted x ground)
  """
  t_et = token_et.to_numpy()
  t_gt = token_gt.to_numpy()
  matrix_lv = np.vectorize(levenshtein_score)(t_et[:, np.newaxis], t_gt[np.newaxis, :])
  #matrix_jaccard
  return matrix_lv
  
def compute_tpfp(matrix):
    """
    This function considers Extracted token as Ground-truth token when its Levenshteins 
    similarity index is > 0.7. Otherwise it is non-gt token.
    :param matrix: Similarity Matrix
    :return: Number of GT in ET, Number of Non GT
        m:no. of tokens in et
        n:no. of tokens in gt
    """
    tp=0
    fp=0
    rows=matrix.shape[0]
    cols=matrix.shape[1]
    for x in range(0,rows):
        for y in range(0,cols):
            if matrix[x,y] > 0.7:
                flag=True
                break
            else:
                flag=False
        if flag is True:
            tp=tp+1
        else:
            fp=fp+1
    m,n = matrix.shape
    return tp,fp,m,n
  
def calculate_scores(tp,m,n):
    """
    Args:
      tp: True positive-Number of GT in ET
      fp: False Positive - Number of Non-GT in ET
      m: no. of tokens in et
      n:  no. of tokens in gt
    :return: Precision, Recall and F1 Score
    """
    prec=tp/m
    recall= tp/n
    if prec==0 and recall==0:
        return 0,0,0
    else:
        f1_score= (2 * prec * recall)/ (prec + recall)
        return f'f1_score: {f1_score} \nPrecision: {prec} \nRecall: {recall}' 
    #f1_score, prec, recall
    

def calculate_score_collated(et_collated,gt_collated):
  """
  Args:
    df: dataframe containing text ground truth text
    text: String containing parsed text from PDF parser
    
  Return: 
    Accuracy Score
  """
  jac_index=textdistance.jaccard.normalized_similarity(et_collated,gt_collated)
  lcs_sim = rapidfuzz.distance.LCSseq.normalized_similarity(et_collated,gt_collated)
  #gestalt_match =textdistance.ratcliff_obershelp.normalized_similarity(et_collated,gt_collated)
  return f'Jaccard Index: {jac_index} \nLongest Common Subsequence Similarity: {lcs_sim}'# \nGestalt pattern Score {gestalt_match}'
#rapidfuzz.distance.LCSseq.normalized_similarity(gt_collated,et_collated)