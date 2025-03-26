import pickle
from collections import Counter
import argparse
import pandas as pd

"""
counts = None

with open("./data/1b_benchmark.train.tokens","r") as f:
    data = f.read()
    tokens = data.split()
    counts = Counter(tokens)


def preprocess(window_length=1):

    token_count = dict()
    words=[]
    

    with open("./data/1b_benchmark.train.tokens","r") as file:
      
        for line in file:
            
            line_split = line.split()        
            
            for i in range(len(line_split)):
                if counts[line_split[i]]<=2:
                    line_split[i]="<unk>"
                  
                

            for _ in range(window_length-1):
                line_split.insert(0,"<START>")

            line_split.append("<STOP>")

            sent = [" ".join(line_split[i:i+window_length]) for i in range(len(line_split)-window_length+1)]
          
            
            for token in sent:
                if not token in token_count.keys():
                    token_count[token]=1
                else:
                    token_count[token]+=1
        
    with open(f"{window_length}-gram.pkl","wb") as file:
        pickle.dump(token_count,file)
"""

train_len = 0
dev_len = 0
test_len = 0


with open("./data/1b_benchmark.train.tokens","r") as f:
    for line in f:
        train_len+=1

with open("./data/1b_benchmark.dev.tokens","r") as f:
    for line in f:
        dev_len+=1

with open("./data/1b_benchmark.test.tokens","r") as f:
    for line in f:
        test_len+=1


UNI=None
BI = None
TRI = None


with open("./data/1-gram.pkl","rb") as f:
    UNI = pickle.load(f)

with open("./data/2-gram.pkl","rb") as f:
    BI = pickle.load(f)

with open("./data/3-gram.pkl","rb") as f:
    TRI = pickle.load(f)

total = sum(UNI.values())




def unigram(token,smooth=False,k=0):
    if token not in UNI.keys():
        token="<unk>"

    if not smooth:
        return UNI[token]/total
    else:
        v = len(UNI.keys())
        return UNI[token]+k/total+v*k

def bigram(prev,nxt,smooth=False,k=0):

    if prev not in UNI.keys():
        prev = "<unk>"
    if nxt not in UNI.keys():
        nxt = "<unk>"


    conc = prev+" "+nxt

    if smooth:
        v = len(UNI.keys())
        denum = UNI.get(prev,0) + v*k 
        num = BI.get(conc,0) + k
        return num/denum
        

    if conc not in BI.keys():
        return unigram(nxt)
    else:
        num = BI[conc]

    denum = UNI[prev]

    res = num/denum


    return res

def trigram(prev1,prev2,nxt,smooth=False,k=0):
    
    if nxt not in UNI.keys():
        nxt = "<unk>"

    if prev1 not in UNI.keys():
        prev1 = "<unk>"

    if prev2 not in UNI.keys():
        prev2 = "<unk>"
    
    conc = prev1+" "+prev2+" "+nxt
    pr = prev1+" "+prev2

    if smooth:
        v = len(UNI.keys())
        num = BI.get(pr,0)+k
        denum = TRI.get(conc,0)+v*k
        res = num/denum
        return res

    if pr not in BI.keys() or conc not in TRI.keys():
        return 0.5*(bigram(prev2,nxt)+bigram(prev1,nxt))

    num = TRI[conc]
    denum = BI[pr]
    

    return num/denum

class interpolate:

    def __init__(self,c1,c2,c3):
        self.c1=c1
        self.c2=c2
        self.c3=c3
    
    def __call__(self,prev1,prev2,nxt):
        return self.c3*trigram(prev1,prev2,nxt)+self.c2*bigram(prev2,nxt)+self.c1*unigram(nxt)


def per_inter(model,mode="train"):
    per = 0

    with open(f"./data/1b_benchmark.{mode}.tokens","r") as file:
    
            for line in file:

                line_split = line.split()
                line_split.insert(0,"<START>")
                line_split.insert(0,"<START>")
                line_split.append("<STOP>")
                N = len(line_split)
                res = 1
                for i in range(len(line_split)-2):
                    res*=(1/model(line_split[i],line_split[i+1],line_split[i+2]))**(1/N)
                if mode =="dev":
                    per += res/dev_len
                elif mode=="train":
                    per += res/train_len
                else:
                    per += res/test_len

    return per
    

def perplexity(model="unigram",mode="train"):

    per = 0
    
    if model=="unigram":

        with open(f"./data/1b_benchmark.{mode}.tokens","r") as file:
        
            for line in file:

                line_split = line.split()
                line_split.append("<STOP>")
                N = len(line_split)
                res = 1
                for w in line_split:
                    res*=(1/unigram(w))**(1/N)

                if mode =="dev":
                    per += res/dev_len
                elif mode=="train":
                    per += res/train_len
                else:
                    per += res/test_len


    elif model=="bigram":

        with open(f"./data/1b_benchmark.{mode}.tokens","r") as file:
        
            for line in file:

                line_split = line.split()
                line_split.insert(0,"<START>")
                line_split.append("<STOP>")
                N = len(line_split)
                res = 1
                for i in range(len(line_split)-1):
                    res*=(1/bigram(line_split[i],line_split[i+1]))**(1/N)
                if mode =="dev":
                    per += res/dev_len
                elif mode=="train":
                    per += res/train_len
                else:
                    per += res/test_len
    
        
    else:
    
        with open(f"./data/1b_benchmark.{mode}.tokens","r") as file:
        
            for line in file:

                line_split = line.split()
                line_split.insert(0,"<START>")
                line_split.insert(0,"<START>")
                line_split.append("<STOP>")
                N = len(line_split)
                res = 1
                for i in range(len(line_split)-2):
                    res*=(1/trigram(line_split[i],line_split[i+1],line_split[i+2]))**(1/N)
                if mode =="dev":
                    per += res/dev_len
                elif mode=="train":
                    per += res/train_len
                else:
                    per += res/test_len
   
    return per

def perplexity_laplace(model="unigram",mode="train",k=0):

    per = 0
    
    if model=="unigram":

        with open(f"./data/1b_benchmark.{mode}.tokens","r") as file:
        
            for line in file:

                line_split = line.split()
                line_split.append("<STOP>")
                N = len(line_split)
                res = 1
                for w in line_split:
                    res*=(1/unigram(w,True,k))**(1/N)
                if mode =="dev":
                    per += res/dev_len
                elif mode=="train":
                    per += res/train_len
                else:
                    per += res/test_len
        
   
                

    elif model=="bigram":

        with open(f"./data/1b_benchmark.{mode}.tokens","r") as file:
        
            for line in file:

                line_split = line.split()
                line_split.insert(0,"<START>")
                line_split.append("<STOP>")
                N = len(line_split)
                res = 1
                for i in range(len(line_split)-1):
                    res*=(1/bigram(line_split[i],line_split[i+1],True,k))**(1/N)
                if mode =="dev":
                    per += res/dev_len
                elif mode=="train":
                    per += res/train_len
                else:
                    per += res/test_len
    
    else:
      
        with open(f"./data/1b_benchmark.{mode}.tokens","r") as file:
        
            for line in file:

                line_split = line.split()
                line_split.insert(0,"<START>")
                line_split.insert(0,"<START>")
                line_split.append("<STOP>")
                N = len(line_split)
                res = 1
                for i in range(len(line_split)-2):
                    res*=(1/trigram(line_split[i],line_split[i+1],line_split[i+2],True,k))**(1/N)
                if mode =="dev":
                    per += res/dev_len
                elif mode=="train":
                    per += res/train_len
                else:
                    per += res/test_len
    
    return per

def inter_experiment(mode="dev"):

    C = [[1/3,1/3,1/3],[6/10,3/10,1/10],[3/10,6/10,1/10],[3/10,1/10,6/10],[0.1,0.3,0.6],[6/10,1/10,3/10]]
    
    res=dict()
    c1=[]
    c2=[]
    c3=[]
    pp =[]

    for c in C:
        model = interpolate(c[0],c[1],c[2])
        c1.append(round(c[0],2))
        c2.append(round(c[1],2))
        c3.append(round(c[2],2))
        r = per_inter(model,mode)
        pp.append(round(r,2))
        
    
    res['lambda_1']=c1
    res["lambda_2"]=c2
    res["lambda_3"]=c3
    res["perplexity"]=pp

    df = pd.DataFrame.from_dict(res)
    df.to_csv(f"{mode}_interpolate.csv",index=False)

def main():
    
    parser = argparse.ArgumentParser("perplexity calculator")
    parser.add_argument("--model",type=str,default="unigram",choices=["unigram","bigram","trigram"],help="choose n-gram model")
    parser.add_argument("--dataset",type=str,default="dev",choices=["train","dev","test"],help="choose dataset (train, dev or test)")
    parser.add_argument("-s","--smoothing",action="store_true")
    parser.add_argument("-k",type=float,default=0.0)

    args = parser.parse_args()


    if args.smoothing:
        result = perplexity_laplace(args.model,args.dataset,args.k)
        print(f"model:{args.model}, dataset:{args.dataset}, k={args.k}, perplexity:{result}")

    else:
        result = perplexity(args.model,args.dataset)
        print(f"model:{args.model}, dataset:{args.dataset}, perplexity:{result}")

if __name__ == "__main__":
  
    main()

