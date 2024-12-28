import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()



CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
        test_sentence[i]
    )
    for i in range(CONTEXT_SIZE, len(test_sentence))
]

## creating a vocabulory
vocab=set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}

class Ngram(nn.Module):
    def __init__(self,vocab_size,embedding_size,context_size):
        super(Ngram,self).__init__()
        self.embed1=nn.Embedding(vocab_size,embedding_size)
        self.lin1=nn.Linear(context_size*embedding_size,128)
        self.lin2=nn.Linear(128,vocab_size)
        #self.Relu=F.relu()

    def forward(self,x):
        x=self.embed1(x)
        x=x.view((1, -1))
        #x=self.lin1(x)
        x = F.relu(self.lin1(x))
        #x=self.Relu(x)
        x=self.lin2(x)

        return x

losses=[]
criterion=torch.nn.CrossEntropyLoss()
model=Ngram(len(vocab),EMBEDDING_DIM,CONTEXT_SIZE)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss=0
    for context,target in ngrams:
        context_idxs=torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        model.zero_grad()
        preds=model(context_idxs)
        loss=criterion(preds,torch.tensor([word_to_ix[target]],dtype=torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    losses.append(total_loss)

print(losses)
