import json
import pandas as pd

def json_to_df(filepath):
  """
  Args:
    filepath: path of the json file

  Return:
    dataframe of the json file with columns text and category
    type of document
  """
  
  with open(filepath,'r') as f:
      data = json.load(f)
  df = pd.DataFrame(data['form'])
  
  if data['form'] == []:
    return (0,0)
  else: 
    return df[['text','category']],data['metadata']['doc_category']



def separate_token_format_gt(df):
  """
  Args:
    df: dataframe
  Return:
    df_final: dataframe with new column where each row is a tokenized sentence
  """
  df['tokens_gt']=df['text'].apply(lambda x: x.split())

  df_final = df.explode(['tokens_gt']).reset_index(drop=True)
  return df_final


def separate_token_format_et(text):
  """
  Args:
    text: parsed string from PDF parser
  Return:
    df_final: dataframe where each row is a tokenized sentence
  """
  df_final = pd.DataFrame(text.split()).rename(columns ={0:'tokens_et'})

  return df_final


def collated_text(df):
  """
  Args:
    df: dataframe column containing text
  Return:
    collated text
  """
  collated_list = df['text'].to_list()
  return ' '.join(collated_list)