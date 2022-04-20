from glob import glob
from tqdm import tqdm
import re
import os
#import kss # sentence separator

email_regex = re.compile(r"\S*@\S*\s?", flags=re.UNICODE)
url_regex = re.compile(r"^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$", flags=re.UNICODE)
sep_regex = re.compile(r"[()\";:<>{}`+=|,]", flags=re.UNICODE)
thousand_comma_regex = re.compile(r"(\d),(\d)", flags=re.UNICODE)
char_regex = re.compile(r"[a-zA-Z0-9가-힣.\$\%\&\-\~ ]+", flags=re.UNICODE)
latin_regex = re.compile(r"[a-zA-Z]", flags=re.UNICODE)
number_regex = re.compile(r"\d", flags=re.UNICODE)
numbering_regex = re.compile(r"[a-zA-Z\d][\.] ", flags=re.UNICODE)
sentence_sep_regex = re.compile(r"(?![0-9a-zA-Z])[\.\!\?(\n)](?![0-9a-zA-Z])", flags=re.UNICODE)
latin_hangul_regex = re.compile(r"([a-zA-Z]+)([가-힣]+)", flags=re.UNICODE)
url_regex = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", flags=re.UNICODE)
html_regex = re.compile(r"&[a-z0-9#;]{1,5}", flags=re.UNICODE)
style_regex = re.compile(r"style=\"(.*)\"", flags=re.UNICODE)
css_regex = re.compile(r"(valign|float|width|height)=([0-9a-zA-Z%])+", flags=re.UNICODE)

latin_numbering_regex = re.compile(r"[\d][\.] ", flags=re.UNICODE)
latin_abbr_regex = re.compile(r"(^|[\s])(mr|mrs|ms|miss|dr|mon|tue|tues|wed|thu|thur|fri|sat|sun|jan|feb|mar|apr|may|jun|julaug|sept|oct|nov|dec)\.", flags=re.UNICODE)
latin_ampm_regex = re.compile(r"(a|p)(\.)(m)(\.)", flags=re.UNICODE)

sc = []
hc = []
path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(path, 'special_character.txt'), 'r') as fs:
    for line in fs:
        sc.append(line.split())

def preprocess(text, lang='ko', separate=True, remove_non_char=True, use_ratio_filter=True):
    text = text.lower()
    if lang == 'ko':
        for c in sc:
            text = text.replace(c[0], c[1].replace('\n',''))
        for c in hc:
            text = text.replace(c[0], c[1].replace('\n',''))
    text = remove_parenthesis(text)
    text = before_split(text, lang)
    if separate == True:
        sentences = sentence_separate(text)
    else:
        sentences = [text]
    text = []
    
    for sen in sentences:        
        if remove_non_char:
            sen = after_split(sen)

        if len(sen) == 0:
            continue

        if use_ratio_filter:
            if not ratio_filter(sen, latin_th=0.4 if lang=='ko' else 1):        
                continue

        text.append(sen)

    return text

def sentence_separate(text):
    return re.split(sentence_sep_regex, text) # based on punctuation
    #return kss.split_sentences(text) # more robust rules based on pattern

def remove_leading_sym(line):
    def remove_one(line):
        if len(line) < 1:
            return line
        leading = line.split()[0].replace('.', '')
        if re.match(r"[a-zA-Z0-9]", leading):
            return ' '.join(line.split()[1:])
        else:
            return line

    while 1:
        s = remove_one(line)
        if s == line:
            break
        else:
            line = s
    return line

def ratio_filter(s, latin_th=0.4, space_min=0.05, space_max=0.4, number_th=0.5):
    latin_ratio = len(re.findall(latin_regex, s))/len(s.replace(' ', ''))
    space_ratio = len(re.findall(" ", s)) / len(s)
    number_ratio = len(re.findall(number_regex, s))/len(s.replace(' ', ''))
    if (latin_ratio <= latin_th and
        space_ratio <= space_max and
        space_ratio >= space_min and
        number_ratio <= number_th):
        return True
    return False

def before_split(text, lang):
    text = re.sub(email_regex, '', text)
    text = re.sub(url_regex, '', text)
    if lang == 'ko':
        text = re.sub(numbering_regex, '', text)
    else:
        text = re.sub(latin_numbering_regex, '', text)
        text = re.sub(latin_abbr_regex, r'\1\2', text)
        text = re.sub(latin_ampm_regex, r'\1\3', text)
    text = re.sub(html_regex, '', text)
    text = re.sub(style_regex, '', text)
    text = re.sub(css_regex, '', text)
    text = re.sub(thousand_comma_regex, r"\1\2", text)
    text = re.sub(sep_regex, ' ', text)
    text = re.sub(' +', ' ', text)
#    text = re.sub(latin_hangul_regex, r"\1 \2", text)
    if lang == 'ko':
        text = re.sub(r"\w{12,}", '', text).strip()
    else:
        text = re.sub(r"\w{20,}", '', text).strip()

    return text

def after_split(text):
    text = re.findall(char_regex, text)
    text = ' '.join(''.join(text).split())

    return text

def remove_parenthesis(s):

    pts = [('(', ')'), ('[', ']'), ('{', '}'), ('《', '》'), ('||', '||'), ('<', '>')]

    def remove_one(t, pt):
        begin = t.find(pt[0])
        if begin == -1:
            return t
        end = t[begin:].find(pt[1])
        if end == -1:
            return t
        end += begin
        next_begin = t[begin+1:].find(pt[0])
        if next_begin < 0:
            return t[:begin]+t[end+1:]
        next_begin += begin + 1
        if next_begin < end:
            return t[:next_begin] + remove_one(t[next_begin:], pt)
        return t[:begin]+t[end+1:]

    for pt in pts:
        i = 0
        while 1:
            i+=1
            t = remove_one(s, pt)
            if t == s:  break
            s = t
    return s



if __name__ == '__main__':
    text="Washington: The US military disabled scores of aircraft and armored vehicles as well as a high-tech rocket defense system at the Kabul airport before it left Monday, a US general said."
    print('text',text)
    processed_text = preprocess(text,lang="en")
    print('processed_text',processed_text)
