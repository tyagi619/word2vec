with open('dataset/wikiBillionChars/enwik9_cleaned','r') as f:
    sentences = f.read()

sentences = sentences.replace('\n',' ')
sentences = sentences.split('.')


sentences_new = []
sentences_new.append('sentence_index sentence')
for i,sentence in enumerate(sentences):
    sentence_t = sentence.strip()
    sentence_t = ' '.join(sentence_t.split(' '))
    if len(sentence_t) == 0:
        continue

    if len(sentence.split(' ')) < 5:
        continue

    sentence_t = f'{i} {sentence_t}'
    sentences_new.append(sentence_t)

print(len(sentences_new))
sentences_new = sentences_new[:int(0.1 * len(sentences_new))]
print(len(sentences_new))
sentences_new = '\n'.join(sentences_new)

with open('dataset/wikiBillionChars/datasetSentences.txt','w') as f:
    f.write(sentences_new)
