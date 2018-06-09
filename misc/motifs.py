"""
SCRIPT TO MAKE MEMES. this was from an old version of the code, so it might require some fixes to get working.

"""
from dataloaders.visual_genome import VG
# import matplotlib
# # matplotlib.use('Agg')
from tqdm import tqdm
import seaborn as sns
import numpy as np
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from collections import defaultdict
train, val, test = VG.splits(filter_non_overlap=False, num_val_im=2000)

count_threshold = 50
pmi_threshold = 10

o_type = []
f = open("object_types.txt")
for line in f.readlines():
  tabs = line.strip().split("\t")
  t = tabs[1].split("_")[0]
  o_type.append(t)

r_type = []
f = open("relation_types.txt")
for line in f.readlines():
  tabs = line.strip().split("\t")
  t = tabs[1].split("_")[0]
  r_type.append(t) 

max_id = 0

memes_id_id = {}

memes_id = {}
id_memes = {}

id_key = {}
key_id = {}
#go through and assign keys
dataset = []
for i in range(0, len(train)):
  item = []
  _r = train.relationships[i]
  _o = train.gt_classes[i]
  for j in range(0, len(_r)):
    h = _o[_r[j][0]]
    t = _o[_r[j][1]]
    e = _r[j][2]
    key1 = (h,e,t)
    if key1 not in key_id: 
      id_key[max_id] = key1
      key_id[key1] = max_id
      max_id += 1
    item.append(key_id[key1])
  dataset.append(item)

cids = train.ind_to_classes
rids = train.ind_to_predicates
all_memes = []

def id_to_str(_id):
  key = id_key[_id]
  if len(key) == 2:
    pair = key
    l1, s1 = id_to_str(pair[0])
    l2, s2 = id_to_str(pair[1])
    return (l1 + l2, s1 + " & " + s2)
  else:
    return (1,"{}--{}-->{}".format(cids[key[0]], rids[key[1]], cids[key[2]]))

new_meme_score = {}
for p in range(0,25):
  print("iteration : {}".format(p)) 
  unigrams = defaultdict(float)
  bigrams = defaultdict(float) 
  unigrams_ori = defaultdict(float)
  T = 0
  T2 = 0
  for i in range(0, len(dataset)):
    item = dataset[i]
    for j in range(0, len(item)):
      key1 = item[j] 
      unigrams_ori[key1] += 1
      #T += 1
      for j2 in range(j+1 , len(item)):
        key2 = item[j2]
        if key1 > key2 : jkey = (key1, key2)
        else: jkey = (key2, key1)
        unigrams[key1] += 1
        unigrams[key2] += 1
        bigrams[jkey] += 1
        T2 += 1
  
  pmi = []
  for (jkey,val) in bigrams.items():
    pval = (val / T2) / ( (unigrams[jkey[0]]/ T2) * (unigrams[jkey[0]] / T2 )) 
    #print("{} {} {}".format(jkey, val, pval))
    if val > count_threshold and unigrams_ori[jkey[0]] > count_threshold and unigrams_ori[jkey[1]] > count_threshold and pval > pmi_threshold : 
      pmi.append( (pval , jkey, val) )
  #    new_memes.add(jkey)

  new_memes = set()
  pmi = sorted(pmi, key = lambda x: -x[0])
  new_meme_c = set()
  for (v,k, f) in pmi:
    #if k[0] in all_memes and k[1] in all_memes: continue 
    #if len( new_memes) > 1000: break
    if k[0] in new_meme_c or k[1] in new_meme_c: continue
    new_meme_c.add(k[0])
    new_meme_c.add(k[1])
    print("{} & {} \t {} \t {} \t {} \t {}".format(id_to_str(k[0]), id_to_str(k[1]), v, unigrams[k[0]], unigrams[k[1]], bigrams[k]))
    new_memes.add(k)
  #assign new ids to the memes
    new_meme_score[k] = v 
    #break
  for meme in new_memes:
    if meme in key_id: continue
    all_memes.append(max_id)
    id_key[max_id] = meme
    key_id[meme] = max_id
    max_id+=1
  print("{} memes discovered ".format(len(new_memes)))
  #go through and adjust the dataset
  new_dataset = []
  eliminated = 0
  for i in range(0,len(dataset)):
    item_save = dataset[i]
    item = item_save
    new_item = []
    #merges = {}
    while True:
     best = None
     best_score = 0
     for j in range(0, len(item)):
      key1 = item[j]
      for j2 in range(j+1 , len(item)):
        key2 = item[j2]
        if key1 > key2 : jkey = (key1, key2)
        else: jkey = (key2, key1)
        if jkey in new_meme_score and new_meme_score[jkey] > best_score: 
          best = (j, j2) 
          best_score = new_meme_score[jkey]
        #if jkey in key_id and j not in merges and j2 not in merges: 
        #  merges[j] = j2
        #  merges[j2] = j
     if best is not None:
      for j in range(0, len(item)):
        if j == best[0]: 
          key1 = item[j]
          key2 = item[best[1]]
          if key1 > key2 : jkey = (key1, key2)
          else: jkey = (key2, key1)
          new_item.append(key_id[jkey]) 
        elif j == best[1]: continue
        else: new_item.append(item[j])
      #break
      item = new_item
      new_item = [] 
     else:
      #print("done")
      new_item = item 
      break
    #for j in range(0, len(item)):
    #  if j not in merges: new_item.append(item[j])
    #  elif j < merges[j]: 
    #    key1 = item[j]
    #    key2 = item[merges[j]]
    #   if key1 > key2 : jkey = (key1, key2)
    #    else: jkey = (key2, key1)
    #    new_item.append(key_id[jkey])  
    eliminated += len(item_save) - len(new_item)
    new_dataset.append(new_item)
  print ("{} total eliminated".format(eliminated))
  dataset = new_dataset

meme_freq = defaultdict(float)

def increment_recursive(i):
  #meme = id_key[i]
  if i in all_memes:
    meme_freq[i] += 1
    key1 = id_key[i][0]
    key2 = id_key[i][1]
    increment_recursive(key1)
    increment_recursive(key2)

def meme_length(i):
  if i in all_memes:
    return meme_length(id_key[i][0]) + meme_length(id_key[i][1])
  else: 
    return 1

#compute statistics of memes
for i in range(0,len(dataset)):
  item = dataset[i]
  for j in range(0, len(item)):
    increment_recursive(item[j])
  
for meme in all_memes:
  print ("{} {}".format( id_to_str(meme), meme_freq[meme]))

T = 0 
T2 = 0
n_images = defaultdict(float)
n_edges = defaultdict(float)
for item in dataset:
  meme_lengths = []
  for j in range(0, len(item)):
    meme_lengths.append(meme_length(item[j]))
  n_images[max(meme_lengths)] += 1
  #for l in meme_lengths: n_images[l] +=1
  T += 1

for item in dataset:
  for j in range(0, len(item)):
    l = meme_length(item[j])
    n_edges[l] += l
    T2 += l

for (k,v) in n_images.items():
  print("{} {}".format(k, v/T))
print("---")
for (k,v) in n_edges.items():
  print("{} {}".format(k, v/T2))




  
