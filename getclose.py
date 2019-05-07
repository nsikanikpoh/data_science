kw = ['tiger','lion','elephant','black cat','dog']
s = ["I saw a tyger","There are 2 lyons","I mispelled Kat","bulldogs"]

def find_similar_word(s, kw, thr=0.5):
    from difflib import SequenceMatcher
    out = []
    for i in s:
        f = False
        for j in i.split():
            for k in kw:
                if SequenceMatcher(a=j, b=k).ratio() > thr:
                    out.append(k)
                    f = True
                if f:
                    break
            if f:
                break
        else:
            out.append(None)    
    return out


print(find_similar_word(s, kw))