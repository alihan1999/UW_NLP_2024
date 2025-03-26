from collections import defaultdict
from tqdm import tqdm
import pickle

data = []
with open("./data/BPE-data.txt") as file:
    for line in file:
        data.append(line)
train_data = data[:4000]
test_data = data[4000:]

class BPE:

    def __init__(self,text):

        self.text=text
        self.merges = defaultdict(str)
        self.word_count = defaultdict(int)
        self.splits = defaultdict(list)

        v = []
        
        for r in text:
            words = r.split(" ")
            words = [words[0]]+[" "+w for w in words[1:]]
            for w in words:
                self.word_count[w]+=1
            for c in r:
                if c not in v:
                    v.append(c)
        
        self.vocab = v.copy() + ["<STOP>"]
        self.vocab.sort()
       
        
        for word in self.word_count.keys():
            self.splits[word] = [l for l in word]

        
    def _most_frequent_pair(self):

        pair_freq = defaultdict(int)

        for word in self.word_count.keys():
            l = self.splits[word]
            for i in range(len(l)-1):
                conc = (l[i],l[i+1])
                pair_freq[conc]+=1
        
        m = sorted(pair_freq.items(),key=lambda x:(x[1],x[0]))[-1]
        merge = m[0]
        freq = m[1]
        return merge[0],merge[1],freq
    
    def _merge(self,m):
        
        for word in self.word_count.keys():
            split=self.splits[word]
          
            i=0
            while i < len(split)-1:
                if split[i]+split[i+1]==m:
                    split = split[:i]+[m]+split[i+2:]
                else:
                    i+=1
            self.splits[word]=split
    
    def train(self):
        freq = 3
        i=1
        while freq>2:

            m1,m2,freq= self._most_frequent_pair()
            n = m1+m2
            self.vocab.append(n)
            self.merges[(m1,m2)]=n
            self._merge(n)


            i+=1

    def apply(self,inp,merges=None):

        if merges:
            with open(merges,"rb") as file:
                self.merges = pickle.load(file)["merges"]
            print("loaded successfully")
        

        
        tokens = []

        for r in inp:
            line_sep= [r.split(" ")[0]] + [" "+w for w in r.split(" ")[1:]]
            splits = [[l for l in w] for w in line_sep]
            
            
            for idx,split in enumerate(splits):
              
                for p,r in self.merges.items():
                    i = 0
                    while i < len(split)-1:
                        if split[i]+split[i+1] == r:
                            
                            split = split[:i] + [r] + split[i+2:]
                            
                        else:
                            i+=1
                splits[idx]=split

            splits = sum(splits,[])
            tokens.append(splits)
        return tokens     
                  

