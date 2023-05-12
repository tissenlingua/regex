import re
import math
import jieba
import numpy as np
import pandas as pd
from collections import Counter

# 設定
file_path = 'abt/'
jieba.set_dictionary('ref/dict.txt')
stopwords = "，。！？【】（）－：、,.!?[]()-:\n"

# ------------------------------------------------

# 正確範本
def getAns(file):
    ans = pd.read_excel(file).rename(columns={'Q':'q','正確範本':'ans'})
    ans['noPunc'] = ans['ans'].apply(lambda x: ''.join(re.compile(r"\w+").findall(x)))
    return ans

# STT 文字結果
def getPoc(file):
    poc = pd.read_excel(file,sheet_name=1)
    poc.columns = ['role','start','end','sentence']
    return poc

# 將 STT 文字結果業務員部分的問題 concat 在一起
def pocGrouping(poc):
    pocConcat = ['']
    i = 0
    for r,s in poc[['role','sentence']].values:
        if r=='A':
            pocConcat[i] = pocConcat[i] + s
        else:
            pocConcat.append('')
            i += 1
    pocConcat = [s for s in pocConcat if s!='']; del i
    pocConcat = pd.DataFrame(pocConcat,columns=['q'])
    pocConcat['noPunc'] = pocConcat.q.apply(lambda x: ''.join(re.compile(r"\w+").findall(x)))
    return pocConcat

# ------------------------------------------------

# CKIP 斷詞套件
def getCKIP():
    from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
    try:
        # 可以先下載下來存起來
        ws_driver = pd.read_pickle('model/ws_driver_albert_base.pi')
        pos_driver = pd.read_pickle('model/pos_driver_albert_base.pi')
        ner_driver = pd.read_pickle('model/ner_driver_albert_base.pi')
    except:
        # 要連網下載
        ws_driver  = CkipWordSegmenter(model="albert-base", device=-1)
        pos_driver = CkipPosTagger(model="albert-base", device=-1)
        ner_driver = CkipNerChunker(model="albert-base", device=-1)
    return ws_driver, pos_driver, ner_driver

# 斷詞 WS (word segmentation)
def lcutJeiba(lists_of_setences):
    return [' '.join([s for s in jieba.lcut(l) if s not in stopwords]) for l in lists_of_setences]
def lcutCKIP(lists_of_setences):
    return pd.Series(ws_driver(lists_of_setences)).apply(lambda x: ' '.join(x))

# 命名實體辨識 NER (named entity recognition)
def nerCKIP(lists_of_setences):
    return ner_driver(lists_of_setences)

# ------------------------------------------------

# 文字轉向量
def text_to_vector(text):
    words = re.compile(r"\w+").findall(text)
    return Counter(words)

# 計算 cosine similarity
def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

# 計算相似度矩陣
def compare(pocSentences, ansSentences):
    vec1 = pocSentences.apply(text_to_vector)
    vec2 = ansSentences.apply(text_to_vector)
    matrix = [ get_cosine(v1, v2) for v1 in vec1 for v2 in vec2]
    matrix = np.reshape(matrix, (len(vec1), len(vec2)))
    return pd.DataFrame(matrix)

# ------------------------------------------------

if __name__ == "__main__":
    
    # 初始化 CKIP 模型
    ws_driver, pos_driver, ner_driver = getCKIP()
    
    for f in os.listdir(file_path):
        
        # 取檔
        ans = getAns(file_path+f)
        poc = getPoc(file_path+f)
        
        # 將 STT 文字結果業務員部分的問題 concat 在一起
        pocConcat = pocGrouping(poc)

        # 確認 STT 後的問題數與範本的問題數
        print(f[:-5], '- poc: ', len(pocConcat), 'ans:', len(ans))

        # WS & NER
        ansJ = ans.copy();pocJ = pocConcat.copy()
        ansC = ans.copy();pocC = pocConcat.copy()
        ansJ['lcut'] = lcutJeiba(ansJ['noPunc'])
        pocJ['lcut'] = lcutJeiba(pocJ['noPunc'])
        ansC['lcut'] = lcutCKIP(ansC['noPunc'])
        pocC['lcut'] = lcutCKIP(pocC['noPunc'])
        ansC['ner'] = nerCKIP(ansC['noPunc'])
        pocC['ner'] = nerCKIP(pocC['noPunc'])

        # 計算相似度矩陣
        similarityMatrixJ = compare(pocJ['lcut'], ansJ['lcut'])
        similarityMatrixC = compare(pocC['lcut'], ansC['lcut'])

        # 對每個業務員對話取最接近的題項 (以Jieba斷詞)
        similarityMatrixJ = pd.concat([
            pocJ[['q','lcut']],
            ansJ.loc[similarityMatrixJ.idxmax(axis=1),['q','ans','lcut']].reset_index(drop=True),
            similarityMatrixJ.max(axis=1)
        ], axis=1)

        # 對每個業務員對話取最接近的題項 (以CKIP斷詞)
        similarityMatrixC = pd.concat([
            pocC[['q','lcut','ner']],
            ansC.loc[similarityMatrixC.idxmax(axis=1),['q','ans','lcut','ner']].reset_index(drop=True),
            similarityMatrixC.max(axis=1)
        ], axis=1)

        # 調整欄位名稱
        similarityMatrixJ.columns = ['poc_ans', 'poc_lcut', 'predict', 'original_template', 'template_lcut', 'similarity']
        similarityMatrixC.columns = ['poc_ans', 'poc_lcut', 'poc_ner', 'predict', 'original_template', 'template_lcut', 'template_ner', 'similarity']

        # 匯出結果
        with pd.ExcelWriter(f'output/compare_{f[:-5]}.xlsx') as writer:
            similarityMatrixJ.to_excel(writer, sheet_name="mapping_Jieba")
            similarityMatrixC.to_excel(writer, sheet_name="mapping_CKIP")
            pd.read_excel(file_path+f).to_excel(writer, sheet_name="ori_template", index=None)
        
        # 結束
        print('all complete.')